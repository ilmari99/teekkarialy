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

    def create_profile_for_user(self, username, messages, prev_knowledge = None):
        """
        Create a profile for a user based on a sequence of messages.

        Parameters
        username: str
            The username, for who to create the profile for.
        messages: str
            The messages to create the profile from.
        prev_knowledge: str
            The previous knowledge to use when creating the profile. I.e. the previous profile.
        """
        if prev_knowledge:
            curr_profile = prev_knowledge
        else:
            curr_profile = f"{username} on opiskelija Lappeenrannan teknillisessä yliopistossa."
        # Create the prompt
        prompt = f"Tehtävänäsi on kirjoittaa n. 50 sanan analyysi {username}:sta taustatiedon, sekä lyhyen viesti keskustelun perusteella hyödyntäen persoonallisuus psykologiaa sekä stereotypioita.\n\n"
        prompt += f"Taustatieto: \"{curr_profile}\"\n\n"
        prompt += "Lyhyt viesti ketju:\n\n\"" + messages + "\""
        prompt += f"\n\nPersoonallisuus analyysi {username}:sta:"
        print(f"Prompt:\n{prompt}\n")
        # Generate the new profile
        new_profile = self.get_only_new_tokens(prompt, temperature=0.8, max_new_tokens=100)
        return new_profile

    def generate(self, prompt, temperature=0.7, max_new_tokens=50):
        input = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(
            **input,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def get_only_new_tokens(self, prompt, temperature=0.7, max_new_tokens=50):
        output = self.generate(prompt, temperature, max_new_tokens)
        return output[len(prompt):]
    
    def get_only_next_line(self, prompt, temperature=0.7, max_new_tokens=50):
        output = self.get_only_new_tokens(prompt, temperature, max_new_tokens)
        return output.split('\n')[0]



    
if __name__ == '__main__':
    pass



