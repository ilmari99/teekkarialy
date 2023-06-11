"""
This dataset reads a csv file of sequential messages from a Telegram chat.
The data contains each messages sender and the message itself.

This file contains a convenient class, TelegramDataset, for reading the data and using the data to fine tune a language model.

The goal is to use this dataset to finetune a GPT-3 model, to generate text that resembles the chat history.
Chat history is represented like: "sender: message\nsender: message\n...".

The model is finetuned to predict the next message, given the previous n messages, by making the prompt: "sender: message\nsender: message\n...GPT:"
"""
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import random
import numpy as np
import os
from LanguageModel import LanguageModel

class TelegramDataset(Dataset):
    def __init__(self, filepath, model, tokenizer, max_length=200):
        self.filepath = filepath
        self.model = model
        self.seq_len = 8
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Messages contain a sequence of sender-message pairs
        self.messages = pd.read_csv(self.filepath,sep = ";", encoding="utf-8")
        # Drop if the message has over 80 characters
        self.messages = self.messages[self.messages['message'].str.len() < 80]
        self.messages = self.messages.reset_index(drop=True)
        # Change 30% of the senders to "GPT"
        self.messages['sender'] = self.messages['sender'].apply(lambda x: "GPT" if random.random() < 0.3 else x)
        print(f"Number of messages: {len(self.messages)}")
        self.prompts, self.targets = self.create_prompts()
        print(f"Number of prompts: {len(self.prompts)}")
        print(f"Number of targets: {len(self.targets)}")

    def create_prompts(self):
        """ Combine the messages into prompts and targets. """
        prompts = []
        targets = []
        for i in range(len(self.messages) - self.seq_len):
            prompt = ""
            for j in range(self.seq_len):
                prompt += f"{self.messages['sender'][i+j]}:{self.messages['message'][i+j]}\n"
            prompt += "GPT:"
            target_sender = self.messages['sender'][i+self.seq_len]
            # Change the target senders name to "GPT" in the prompt
            #prompt = prompt.replace(target_sender, "GPT")
            target = prompt + self.messages['message'][i+self.seq_len]
            prompts.append(prompt)
            targets.append(target)
        return prompts, targets

    def __len__(self):
        return len(self.messages) - self.seq_len
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        # Encode the prompt, and pad/truncate it to a fixed length
        # The truncating is done from the left, so the last message is always included
        input = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        # Get the next message
        next_message = self.targets[idx]
        # Encode the next message to a fixed length
        target = self.tokenizer(next_message, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        # Get the labels
        labels = target['input_ids']

        return {"input_ids": input['input_ids'].squeeze(), "labels": labels.squeeze()}


def test_model(model, tokenizer, ds):
    """ Test the model on the dataset. """
    # Get a random sample
    idx = random.randint(0, len(ds))
    sample = ds[idx]
    # Get the prompt
    prompt = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    # Get the target
    target = tokenizer.decode(sample['labels'], skip_special_tokens=True)
    target = target[len(prompt):]
    # Get the predicted next message
    output = model.generate(sample['input_ids'].unsqueeze(0), do_sample=True, temperature=0.6, max_new_tokens=20, no_repeat_ngram_size=2)
    output = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
    output = output[len(prompt):].split('\n')[0]
    # Print the results
    print(f"Prompt: {prompt}")
    print(f"Target: {target}")
    print(f"Output: {output}")
    print()

def evaluate_model(model, tokenizer,ds):
    """ Calculate the loss on the dataset."""
    # Get the dataloader
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    # Calculate the loss
    loss = 0
    print(f"Number of batches: {len(dl)}")
    for batch_n, batch in enumerate(dl):
        start = time.time()
        print(f"Evaluating batch {batch_n}")
        input_ids = batch['input_ids']
        labels = batch['labels']
        outputs = model(input_ids, labels=labels)
        loss += outputs.loss
        print(f"Batch loss: {outputs.loss}")
        print(f"Time taken: {time.time() - start}")
    return loss / len(dl)

    
if __name__ == "__main__":
    # Load the dataset
    model_name = 'TurkuNLP/gpt3-finnish-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(type(tokenizer))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset = TelegramDataset("_chat_history.csv", model, tokenizer)
    dataset_test = TelegramDataset("_chat_history_test.csv", model, tokenizer)


    for i in range(4):
        test_model(model, tokenizer, dataset_test)
    # Evaluate the model
    #loss = evaluate_model(model, tokenizer, dataset_test)
    #print(f"Loss: {loss}")
    # Train the model
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=8,   # batch size per device during training
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset,         # training dataset
        #eval_dataset=dataset,             # evaluation dataset
        #data_collator=data_collator,         # define collator to format data
    )
    trainer.train()
    trainer.save_model("./gpt3-finetuned-telegram")
    #Evaluate the model
    loss = evaluate_model(model, tokenizer, dataset_test)
    print(f"Loss: {loss}")

    # Save the tokenizer
    tokenizer.save_pretrained("./gpt3-finetuned-telegram")
    # Save the model
    model.save_pretrained("./gpt3-finetuned-telegram")
    # Save the dataset
    torch.save(dataset, "./gpt3-finetuned-telegram/dataset.pt")

    


