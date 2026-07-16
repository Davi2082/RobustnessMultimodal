from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re

from configuration import RESPONSES_EXPECTED

# access_token = ''
# access_token = os.environ.get("HF_TOKEN")

RE_MULTIPLE_NEWLINES = re.compile(r"\n+")

models = {"OLDGEMMA": "google/gemma-1.1-2b-it", "LLAMA1B": "meta-llama/Llama-3.2-1B-Instruct",
          "LLAMA3B": "meta-llama/Llama-3.2-3B-Instruct",
          "LLAMA8B": "meta-llama/Llama-3.1-8B-Instruct", "GEMMA2B": "google/gemma-2-2b-it",
          "GEMMA9B": "google/gemma-2-9b-it", "OLMO7B": "allenai/OLMo-7B-0724-Instruct-hf"
          }


class Rephraser:
    def __init__(self, model, device, command):
        pretrained_model = models[model]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16, token=access_token)
        self.command = command
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def prepare_prompt(input_text, command):
        commands = {"REPHRASE": "Rephrase the provided input text.",
                    "PARAPHRASE": "Paraphrase the provided input text.",
                    "SIMPLIFY": "Simplify the provided input text.",
                    "FORMAL": "Rewrite the provided input text in a more formal style.",
                    "INFORMAL": "Rewrite the provided input text in a less formal style.",
                    "CHANGE": "Make changes to the provided input text."}
        prompt = commands[
                     command] + " You can add, remove or replace individual words or punctuation characters, but " + (
                     "try to " if command == "CHANGE" else "keep the changes to the minimum to ") + "preserve the original meaning. " + \
                 f"Return exactly {RESPONSES_EXPECTED} different rephrasings, separated by newline. Do not generate any text except the reformulations.\nINPUT:\n" + input_text + "\nOUTPUT:\n"
        return prompt
    
    @staticmethod
    def unpack_answer(output_text, input_text):
        output_text = RE_MULTIPLE_NEWLINES.sub("\n", output_text)
        response = output_text[output_text.find('OUTPUT:\n'):]
        for end_marker in ["<eos>", "<|eot_id|>", "```<|end_of_text|>", "<end_of_turn>", "<|endoftext|>"]:
            if response.endswith(end_marker):
                response = response[0:(-(len(end_marker)))]
        responses = response.split("\n")
        responses = [response for response in responses if not (
                response.strip() in ['', 'OUTPUT:', 'INPUT:', input_text] or response.startswith(
            'Rephrasing ') or response.startswith('Here are '))]
        responses = [response.lstrip('1234567890-').lstrip('.').lstrip() for response in responses]
        if len(responses) != RESPONSES_EXPECTED:
            # print("ERROR: Not received the expected number of responses after parsing the input. Received " + str(len(responses)))
            pass
        return responses
    
    def rephrase(self, input_text):
        if len(input_text) < 5:
            return []

        prompt = self.prepare_prompt(input_text, self.command)
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoding = self.tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        input_len = encoding["input_ids"].shape[1]

        src_len = len(self.tokenizer(input_text)["input_ids"])
        max_new_tokens = max(48, int(src_len * 2.5))

        with torch.no_grad():
            outputs = self.model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.92,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        variants = []
        for sequence in outputs:
            generated = sequence[input_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip().strip('"').strip()
            if text and text.lower() != input_text.strip().lower():
                variants.append(text)
        return variants
