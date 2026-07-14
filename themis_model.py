import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor

# from transformers import BitsAndBytesConfig
# now all the model imports
from transformers import InstructBlipVisionModel, CLIPVisionModel, AutoModel

import torch.nn as nn

#torch.set_default_device("cuda")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        return torch.matmul(attn, x)


class Themis(nn.Module):
    def __init__(self, img_embed_model, lm_model, is_pythia=False, merge_tokens=None):
        super(Themis, self).__init__()
        self.img_embed_model = img_embed_model
        self.lm_model = lm_model

        self.hinner_dim = self.lm_model.config.hidden_size
        self.emb = self.lm_model.model.embed_tokens  # self.lm_model.base_model.wte
        self.lm_head = self.lm_model.lm_head
        self.h = self.lm_model.model.layers  # self.lm_model.base_model.h

        self.drop = nn.Dropout(0.1)
        # print(self.img_embed_model.config)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(self.img_embed_model.config.hidden_size),
            nn.Linear(
                self.img_embed_model.config.hidden_size, self.hinner_dim * 2, bias=False
            ),
            nn.GELU(),
            nn.LayerNorm(self.hinner_dim * 2),
            nn.Linear(self.hinner_dim * 2, self.hinner_dim, bias=False),
            nn.Tanh(),
        )

        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.hinner_dim))
        self.merge_tokens = merge_tokens
        if merge_tokens is not None:
            n_patches = merge_tokens
            self.patch_merger = PatchMerger(self.hinner_dim, n_patches)
        self.layernorm = nn.LayerNorm(self.hinner_dim)

    def forward(self, images=None, texts=None):
        #reshape the images
        
        if images is not None:
            if isinstance(images, torch.Tensor):
                images = {"pixel_values": images}
            if len(images["pixel_values"].shape) == 4:
                images["pixel_values"] = images["pixel_values"].unsqueeze(1)
            b,k, c, h, w = images["pixel_values"].shape
            images["pixel_values"] = images["pixel_values"].reshape(b*k, c, h, w)
            #get the image and text embeddings
            image_features = self.img_embed_model(**images)
            image_features = image_features.last_hidden_state
            image_features = self.image_proj(image_features)
            #print(image_features.shape)
        
        if texts is not None:
            text_embeds = self.emb(texts["input_ids"])
            text_embeds = text_embeds.view(text_embeds.shape[0], text_embeds.shape[-2],  text_embeds.shape[-1])
        
        #print(text_embeds.shape)
        #add the cls token to the text embeddings
        #cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        
        #stack the image features and text features
        if images is not None and texts is not None:
            x = torch.cat((image_features, text_embeds), dim=1)
            if self.merge_tokens is not None:
                x = self.patch_merger(x)
        elif images is not None:
            x = image_features
        elif texts is not None:
            x = text_embeds
        else:    
            raise ValueError("At least one of images or texts must be not None")

        #pass through the module list h
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        for i in range(len(self.h)):
            x = self.h[i](x, position_ids=position_ids)[0]
        x = x.mean(dim=1)
        
        #print(x.shape)
        #pass through the lm head
        logits = self.lm_head(x)
        #logits *= -1
        scores = torch.sigmoid(logits)

        return scores, logits

    def per_layer_features(self, images=None, texts=None):
        """Mean-pooled hidden state AFTER each transformer layer, shape
        [batch, num_layers, hidden]. Mirrors forward() up to the layer loop;
        used as the input to per-layer linear probes. forward() is untouched.
        """
        if images is not None:
            if isinstance(images, torch.Tensor):
                images = {"pixel_values": images}
            if len(images["pixel_values"].shape) == 4:
                images["pixel_values"] = images["pixel_values"].unsqueeze(1)
            b, k, c, h, w = images["pixel_values"].shape
            images["pixel_values"] = images["pixel_values"].reshape(b * k, c, h, w)
            image_features = self.img_embed_model(**images).last_hidden_state
            image_features = self.image_proj(image_features)

        if texts is not None:
            text_embeds = self.emb(texts["input_ids"])
            text_embeds = text_embeds.view(text_embeds.shape[0], text_embeds.shape[-2], text_embeds.shape[-1])

        if images is not None and texts is not None:
            x = torch.cat((image_features, text_embeds), dim=1)
            if self.merge_tokens is not None:
                x = self.patch_merger(x)
        elif images is not None:
            x = image_features
        elif texts is not None:
            x = text_embeds
        else:
            raise ValueError("At least one of images or texts must be not None")

        feats = []
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        for i in range(len(self.h)):
            x = self.h[i](x, position_ids=position_ids)[0]
            feats.append(x.mean(dim=1))
        return torch.stack(feats, dim=1)  # [batch, num_layers, hidden]

    def per_layer_logits(self, images=None, texts=None):
        """Final classification head applied to each layer's pooled hidden state
        (a 'logit lens'), shape [batch, num_layers]."""
        feats = self.per_layer_features(images=images, texts=texts)  # [B, L, H]
        B, L, H = feats.shape
        return self.lm_head(feats.reshape(B * L, H)).reshape(B, L)


def get_Themis(
    name_llm="distilgpt2",
    name_img_embed="openai/clip-vit-base-patch32",
    use_lora=False,
    is_pythia=False,
    lora_alpha=8,
    lora_r=16,
    lora_dropout=0.2,
    merge_tokens=None,
    device="cuda:0"
):
    # model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name_llm, device_map={"": str(device)}, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(name_llm, trust_remote_code=True)

    # prompt = "Hey, are you conscious? Can you talk to me?"
    # inputs = tokenizer(prompt, return_tensors="pt")

    # # Generate
    # generate_ids = model.generate(inputs.input_ids, max_length=2048)
    # print(model.config)
    # print(generate_ids.shape)
    # exit()
    if "clip" in name_img_embed:
        img_embed = CLIPVisionModel.from_pretrained(
            name_img_embed, device_map={"": str(device)}, trust_remote_code=True
        )
    elif "instruct" in name_img_embed:
        img_embed = InstructBlipVisionModel.from_pretrained(
            name_img_embed, device_map={"": str(device)}, trust_remote_code=True
        )
    else:
        img_embed = AutoModel.from_pretrained(
            name_img_embed, device_map={"": str(device)}, trust_remote_code=True
        )
    processor = AutoImageProcessor.from_pretrained(
        name_img_embed, trust_remote_code=True, device_map={"": str(device)}
    )

    # add a model head for classification
    # change the model head to a classification head

    new_head = torch.nn.Sequential(
        torch.nn.LayerNorm(model.config.hidden_size),
        torch.nn.Linear(model.config.hidden_size, 1, bias=False),
        #torch.nn.Sigmoid(),
    )

    # freeze the model
    for param in img_embed.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    # print list of parameters names and size
    # for name, param in model.named_parameters():
    #    print(name, param.size())
    if use_lora:
        # target_modules = ["embed_tokens","query_key_value","fc1","fc2"]
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
            "embed_tokens",
        ]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model = model.base_model.model
    # print(model)
    model.lm_head = new_head
    themis = Themis(img_embed, model, is_pythia=is_pythia, merge_tokens=merge_tokens)

    # search parameter with cls in the name if found unfreeze and print unfreezed parameter cls
    for name, param in themis.named_parameters():
        if "cls" in name:
            param.requires_grad = True
            # print("Themis unfreeze cls token: ", name)

    # print("Themis: pre unfreeze norm layers")
    # print_trainable_parameters(themis)
    # unfreeze the norm layers
    for name, param in themis.named_parameters():
        if "norm" in name:
            param.requires_grad = True

    # print("Themis: post unfreeze norm layers")
    # print_trainable_parameters(themis)

    return themis, tokenizer, processor
