import torch
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor


class LlavaNext():
    def __init__(self, model_ids = "llava-hf/llava-next-72b-hf", cuda="cpu", max_new_tokens=1024, temperature=0):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,)
        
        self.processor = LlavaNextProcessor.from_pretrained(model_ids)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_ids, quantization_config=quantization_config, device_map=cuda)
        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.cuda = cuda
        self.max_new_token = max_new_tokens
        self.temperature = temperature

    
    def __call__(self, image) -> torch.Any:
        chat1 = [
            {"role": "system",
            "content": [
                {"type" : "text", 
                "text": "You are the most powerful image classifer for classfying road sign. You will be given the image of road sign. Then you will classify the image into one of the correct answers. You can make a guess if it is reasonable. Answer without any explaination and in the format of \"Answer: Category\". The answer is only one of {\"stop\", \"cross walk\", \"no entry\", \"pedestrian crossing\", \"speed limit\", \"traffic light\", \"yield\"}"}]},

            {"role": "user",
            "content": [
                {"type": "image"},
                #  {"type": "text", "text": "What is this sign from all of these answers, {\"stop\", \"cross walk\", \"no entry\", \"pedestrian crossing\", \"speed limit\", \"traffic light\", \"yield\"}, ? Answer without explanation."}
            ]}
        ]
        prompt = self.processor.apply_chat_template(chat1, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=prompt, padding=True, return_tensors="pt").to(self.cuda)

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_token, temperature=self.temperature)

        text = self.processor.decode(output[0], skip_special_tokens=True)
        keyword = "assistant" # Expected this format from the model
        output = text[text.rfind(keyword) + len(keyword): ].strip()
        return output
        