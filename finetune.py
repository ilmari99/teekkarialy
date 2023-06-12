"""
This dataset reads a csv file of sequential messages from a Telegram chat.
The data contains each messages sender and the message itself.

This file contains a convenient class, TelegramDataset, for reading the data and using the data to fine tune a language model.

The goal is to use this dataset to finetune a GPT-3 model, to generate text that resembles the chat history.
Chat history is represented like: "sender: message\nsender: message\n...".

The model is finetuned to predict the next message, given the previous n messages, by making the prompt: "sender: message\nsender: message\n...GPT:"
"""
import json
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import random
import numpy as np
import os
#from LanguageModel import LanguageModel

class TelegramDataset(Dataset):
    def __init__(self, filepath, model, tokenizer, max_length=256):
        self.filepath = filepath
        self.model = model
        self.seq_len = 10
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Messages contain a sequence of sender-message pairs
        self.messages = pd.read_csv(self.filepath,sep = ";", encoding="utf-8").iloc[0:100,:]
        # Drop if the message has over 80 characters
        self.messages = self.messages[self.messages['message'].str.len() < 80]
        self.messages = self.messages.reset_index(drop=True)
        # Change 30% of the senders to "GPT"
        #self.messages['sender'] = self.messages['sender'].apply(lambda x: "GPT" if random.random() < 0.3 else x)
        print(f"Number of messages: {len(self.messages)}")
        self.prompts = self.create_prompts()
        random.shuffle(self.prompts)
        print(f"Number of prompts: {len(self.prompts)}")

    def create_prompts(self):
        """ Combine the messages into prompts and targets. """
        prompts = []
        targets = []
        for i in range(len(self.messages) - self.seq_len):
            prompt = ""
            for j in range(self.seq_len):
                prompt += f"{self.messages['sender'][i+j]}:{self.messages['message'][i+j]}\n"
            prompt += "GPT:"
            #target_sender = self.messages['sender'][i+self.seq_len]
            # Change the target senders name to "GPT" in the prompt
            #prompt = prompt.replace(target_sender, "GPT")
            #target = prompt + self.messages['message'][i+self.seq_len]
            prompt = prompt + self.messages['message'][i+self.seq_len]
            prompts.append(prompt)
            #targets.append(target)
        return prompts#, targets

    def __len__(self):
        return len(self.prompts) - self.seq_len
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        # Encode the prompt, and pad/truncate it to a fixed length
        # The truncating is done from the left, so the last message is always included
        input = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        return {"input_ids": input['input_ids'].squeeze(), "labels": input['input_ids'].squeeze()}


def test_model(model, tokenizer, ds):
    """ Test the model on the dataset. """
    # Get a random sample
    idx = random.randint(0, len(ds)-1)
    sample = ds[idx]
    # Get the prompt
    prompt = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    try:
        prompt_split = prompt.split("GPT:")
    except:
        print(f"No 'GPT:' in prompt: {prompt}")
        return
    prompt = prompt_split[0] + "GPT:"
    target = prompt_split[1]
    # Get the predicted next message
    prompt_tensor = tokenizer(prompt, return_tensors='pt')['input_ids']
    output = model.generate(prompt_tensor, do_sample=True, temperature=0.6, max_new_tokens=50, no_repeat_ngram_size=2)
    output = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
    output = output[len(prompt):].split('\n')[0]
    # Print the results
    print(f"Prompt: {prompt}")
    print(f"Target: {target}")
    print(f"Output: {output}")
    print()


def load_model(model_name):
    """ Load model and tokenizer from the results directory. """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

    
if __name__ == "__main__":
    # Load the dataset
    model_name = 'TurkuNLP/gpt3-finnish-large'
    model_name = "gpt3-finetuned"
    model, tokenizer = load_model(model_name)
    dataset = TelegramDataset("_chat_history.csv", model, tokenizer)
    dataset_test = TelegramDataset("_chat_history_test.csv", model, tokenizer)


    for i in range(4):
        test_model(model, tokenizer, dataset_test)
    # Train the model
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        #eval_dataset=dataset,
        #data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("./gpt3-finetuned-telegram")
    # Save the tokenizer
    tokenizer.save_pretrained("./gpt3-finetuned-telegram")
    # Save the dataset
    torch.save(dataset, "./gpt3-finetuned-telegram/dataset.pt")
    print("Models saved.")
    for i in range(4):
        test_model(model, tokenizer, dataset_test)

    


