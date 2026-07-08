from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re

access_token = 'YOUR_HUGGINGFACE_TOKEN_HERE'
RESPONSES_EXPECTED = 5
RE_MULTIPLE_NEWLINES = re.compile(r"\n+")

models = {"OLDGEMMA": "google/gemma-1.1-2b-it", "LLAMA1B": "meta-llama/Llama-3.2-1B-Instruct",
          "LLAMA3B": "meta-llama/Llama-3.2-3B-Instruct",
          "LLAMA8B": "meta-llama/Llama-3.1-8B-Instruct", "GEMMA2B": "google/gemma-2-2b-it",
          "GEMMA9B": "google/gemma-2-9b-it", "OLMO7B": "allenai/OLMo-7B-0724-Instruct-hf",
          # Models without QwenTokenizer, so without upgrading transformer library
          "YI34B": "01-ai/Yi-34B-Chat",
          "MISTRAL7B": "mistralai/Mistral-7B-Instruct-v0.2",
          "HERMES7B": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"}


class Rephraser:
    def __init__(self, model, device, command):
        pretrained_model = models[model]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16,
                                                          token=access_token)
        self.command = command
        self.model.to(device)
    
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
                 "Return five different rephrasings, separated by newline. Do not generate any text except the reformulations.\nINPUT:\n" + input_text + "\nOUTPUT:\n"
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
            print("ERROR: Not received the expected number of responses after parsing the input.")
        return responses
    
    def rephrase(self, input_text):
        if len(input_text) < 5:
            return []
        new_tokens = int(len(self.tokenizer(input_text)['input_ids']) * RESPONSES_EXPECTED * 1.2)
        prompt = self.prepare_prompt(input_text, self.command)
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        input_ids = {key: input_ids[key].to(self.device) for key in input_ids}
        outputs = self.model.generate(**input_ids, max_new_tokens=new_tokens)
        outputs = outputs.to(torch.device("cpu"))
        output_text = self.tokenizer.decode(outputs[0])
        variants = self.unpack_answer(output_text, input_text)
        return variants
