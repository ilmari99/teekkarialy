from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
import pandas as pd

class LanguageModel:
    def __init__(self, model_name, model=None, tokenizer=None):
        if model:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt, temperature=0.5, max_new_tokens=50):
        print(f"PROMPT --------------------------------------------\n{prompt}\n--------------------------------------------")
        input = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(
            **input,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
        )
        out = self.tokenizer.decode(output[0], skip_special_tokens=False)
        return out
    
    def get_only_new_tokens(self, prompt, temperature=0.5, max_new_tokens=50):
        output = self.generate(prompt, temperature, max_new_tokens)
        out = output[len(prompt):]
        print(f"Generated --------------------------------------------\n{out}\n--------------------------------------------")
        return out
    
    def get_only_next_line(self, prompt, temperature=0.5, max_new_tokens=50):
        output = self.get_only_new_tokens(prompt, temperature, max_new_tokens)
        return output.split('\n')[0]
    
    def get_only_until_token(self, prompt, temperature=0.5, max_new_tokens=50, token='[MEG]'):
        output = self.get_only_new_tokens(prompt, temperature, max_new_tokens)
        return output.split(token)[0] + token



    
if __name__ == '__main__':
    pass



