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
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import random
import numpy as np
import os
#os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] =

class TelegramDataset(Dataset):
    def __init__(self, filepath, model, tokenizer, max_length=256, seq_len=10, overlap = 1):
        self.filepath = filepath
        self.model = model
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.overlap = overlap
        self.max_length = max_length
        # Messages contain a sequence of sender-message pairs
        self.messages = pd.read_csv(self.filepath,sep = ";", encoding="utf-8")
        # Drop if the message has over 80 characters
        self.messages = self.messages[self.messages['message'].str.len() < 120]
        self.messages = self.messages.reset_index(drop=True)
        # Change 30% of the senders to "GPT"
        #self.messages['sender'] = self.messages['sender'].apply(lambda x: "GPT" if random.random() < 0.3 else x)
        print(f"Number of messages: {len(self.messages)}")
        self.prompts = self.create_prompts(overlap=self.overlap)
        random.shuffle(self.prompts)
        print(f"Number of prompts: {len(self.prompts)}")

    def create_prompts(self, overlap = 1):
        """ Combine the messages into prompts, where the prompt is the previous seq_len messages, with the last message sender changed to "GPT".
        """
        # Create a list of prompts, where each prompt is a sequence of messages
        prompts = []
        # Maintain a list of the last seq_len-1 messages
        last_seq_len_messages = self.messages.iloc[0:self.seq_len,:].reset_index(drop=True)
        # Loop through messages starting from the seq_len'th message,
        # since the first seq_len-1 messages are in the last_seq_len_messages list
        # Create a prompt, and update the last_seq_len_messages list
        for i in range(self.seq_len, len(self.messages)):
            prompt = ""
            # Get all messages except the last one
            for j in range(self.seq_len):
                prompt += f"{last_seq_len_messages['sender'][j]}:{last_seq_len_messages['message'][j]}\n"
            # Change j to be te index of the last message in last_seq_len_messages window
            #j += 1
            actual_sender = last_seq_len_messages['sender'][j]
            #actual_message = last_seq_len_messages['message'][j]
            #print(f"Actual sender: {actual_sender}")
            #prompt += actual_sender + ":"+last_seq_len_messages['message'][self.seq_len-1]
            actual_sender = last_seq_len_messages['sender'][j]
            prompt = prompt.replace(f"{actual_sender}:", "GPT:")
            prompts.append(prompt)
            # Update the last_seq_len_messages list by removing the first message, and appending the next
            last_seq_len_messages = last_seq_len_messages.iloc[1:,:].reset_index(drop=True)
            last_seq_len_messages = pd.concat([last_seq_len_messages, self.messages.iloc[i,:].to_frame().transpose()], axis=0).reset_index(drop=True)
        overlap = self.seq_len if overlap is None else overlap
        # Only pick every overlap'th prompt
        prompts = prompts[::overlap]
        return prompts
            
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
        # Get the target, which is the last message after "GPT:"
        # Find the last "GPT:" in the prompt
        prompt_split = prompt.rsplit("GPT:", 1)
    except:
        print(f"No 'GPT:' in prompt: {prompt}")
        return
    prompt = prompt_split[0] + "GPT:"
    try:
        target = prompt_split[1]
    except IndexError:
        print(f"No target in prompt: {prompt}")
        return
    # Get the predicted next message
    pre_prompt = "GPT on hauska ja vitsikäs Teekkari-botti;\n"
    prompt = pre_prompt + prompt
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
    #model_name = 'TurkuNLP/gpt3-finnish-small'
    #model_name = "gpt3-finetuned-telegram"
    model_name = "gpt3-xl-finetuned-3"
    model, tokenizer = load_model(model_name)
    dataset = TelegramDataset("_chat_history.csv", model, tokenizer, max_length=256, seq_len=15, overlap=9)
    dataset_test = TelegramDataset("_chat_history_test.csv", model, tokenizer, max_length=256, seq_len=15, overlap=9)
    
    # Clean up the cache
    torch.cuda.empty_cache()
    # Reserve all the GPU memory
    #torch.cuda.set_per_process_memory_fraction(0.99, None)
    
    for i in range(6):
        pass
        test_model(model, tokenizer, dataset_test)
    exit()
    # Train the model
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        #save_steps=200,
        #save_total_limit=2,
        #fp16=True,
        #fp16_opt_level='O0',
        
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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

    


