"""
This dataset reads a csv file of sequential messages from a Telegram chat, in the format:
id;time;from;text;reply_to_message_id
```
1189;21:12;Pauli lut;Saisko tldr kokouksest😂;
1190;21:13;Taru Haimi;Tervetuloa hallitukseen 😎;
1191;21:13;Joona;Tää 😄;1190
1192;21:13;Taru Haimi;(image or file);
1193;21:13;Taru Haimi;Siinä kokoonpano;1192
1194;22:49;Akseli Suutari;"Onnittelut viel @tartza 
Piti lähtä treeneihi kesken kokouksen";
1195;23:18;Taru Haimi;Kiitoskiitos 😎;
```

This data is used to finetune a GPT-3 model, and the resulting model is used to generate new messages based on previous messages.

The model is tuned by giving it segments of the chat history (first id starting from random -10000-10000).
In a segment:
- no sender is changed to "GPT"
- The sender of the last message is changed to "GPT"
- All occurrences of the last sender in the segment are changed to "GPT"
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import TextDataset, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import random
import numpy as np
import os
from LanguageModel import LanguageModel

def load_dataset(path, test_split_frac = 0.1):
    """ Load a dataset from a csv file.
    """
    # Load the data
    messages = pd.read_csv(path,sep = ";", encoding="utf-8")
    messages = messages.reset_index(drop=True)
    # Split the data into train and test sets
    train_size = int(len(messages) * (1-test_split_frac))
    if test_split_frac == 0:
        test_dataset = pd.DataFrame()
    else:
        test_size = len(messages) - train_size
        train_dataset, test_dataset = train_test_split(messages, test_size=test_size, shuffle = False)
    return train_dataset, test_dataset

def load_model(model_name = "TurkuNLP/gpt3-finnish-small"):
    """ Load a pretrained GPT model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    new_tokens = ["[FS]", "[MSG]","[MEG]"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    # Check if the tokenizer has padding token
    if tokenizer.pad_token is None:
        #add padding token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #Resize the model's embedding layer to include the new tokens
    old_num_tokens = model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embedding layer from {old_num_tokens} to {len(tokenizer)} tokens.")
    return model, tokenizer

class ChatHistoryDataset(Dataset):
    """ Takes in a dataframe of sequential messages from a Telegram chat,
    and batches them into segments, which are used to finetune a GPT-3 model.
    Each message has the following: id;time;from;text;reply_to_message_id
    """
    def __init__(self, df, tokenizer, filter_messages_over_chars = 200, max_tokens_in_segment = 350, segment_len = 10, replace_sender_with = "FinGPT_bot"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_tokens_in_segment = max_tokens_in_segment
        self.segment_len = segment_len
        self.replace_sender_with = replace_sender_with

        # Drop if the message has over filter_messages_over_chars characters
        self.parsed_df = self.df[self.df["text"].str.len() < filter_messages_over_chars]
        print(f"Number of messages after dropping messages with over {filter_messages_over_chars} characters: {len(self.parsed_df)}")

        self.segments = self._create_segments()

    def _create_segments(self):
        """ Create len(df) / segment_len dataframes where each dataframe has segment_len messages.
        In each dataframe, all the occurrences of the last sender are changed to "GPT", and
        """
        segments = []
        for i in range(0, len(self.parsed_df), self.segment_len):
            segment_df = self.parsed_df.iloc[i:i+self.segment_len]
            segment_df = self._prepare_segment(segment_df)
            segments.append(segment_df)
        return segments

    def segment_as_prompt(self, segment_df):
        """ Convert a segment dataframe to a string, that looks like:
        ```
        [MSG]1189[FS]21:12[FS]GPT[FS]Saisko tldr kokouksest😂[FS][MEG]
        [MSG]1190[FS]21:13[FS]Taru Haimi[FS]Tervetuloa hallitukseen 😎[FS][MEG]
        [MSG]1191[FS]21:13[FS]GPT[FS]Tää 😄[FS]1190[MEG]
        ```
        """
        prompt = ""
        for _, row in segment_df.iterrows():
            prompt += f"[MSG]{row['id']}[FS]{row['time']}[FS]{row['from']}[FS]{row['text']}[FS]{row['reply_to_message_id']}[MEG]\n"
        return prompt

    def _prepare_segment(self, segment_df):
        """ Prepare a segment of messages by changing the sender of the last message to "GPT" and all the occurrences of the last sender to "GPT".
        Also change the indexes.
        """
        segment_df = segment_df.copy()
        last_sender = segment_df.iloc[-1]["from"]
        # Change all the occurrences of the last sender to "GPT"
        segment_df["from"] = segment_df["from"].replace(last_sender, self.replace_sender_with)

        # Change id to start from a random number -1000 - 1000 and change all 'reply_to_message_id' to match the new ids
        start_random_id = random.randint(-1000,1000)
        old_ids = segment_df["id"].values
        new_ids = list(range(start_random_id, start_random_id + len(segment_df)))
        id_map = {old_id: new_id for old_id, new_id in zip(old_ids, new_ids)}
        id_map[pd.NA] = pd.NA
        # Change the 'id' column
        segment_df["id"] = segment_df["id"].apply(lambda x: id_map[x])

        # Change the 'reply_to_message_id' column
        segment_df["reply_to_message_id"] = segment_df["reply_to_message_id"].apply(lambda x: id_map.get(x, ""))

        return segment_df

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        text = self.segment_as_prompt(self.segments[idx])
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_tokens_in_segment, padding="max_length")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def test_on_prompts(model, tokenizer, dataset, num_prompts = 10, pre_prompt = ""):
    """ Test the model on random prompts from the dataset.
    Print the prompt, model's response, and the actual response (last message in the segment).
    """
    lang_model = LanguageModel(model_name="GPT", model=model, tokenizer=tokenizer)
    for i in range(num_prompts):
        idx = random.randint(0, len(dataset)-1)
        # Get the segment_df
        segment_df = dataset.segments[idx]
        # Remove and store the last message
        last_message = segment_df.iloc[-1]
        segment_df.drop(segment_df.tail(1).index, inplace=True)
        
        middle_prompt = dataset.segment_as_prompt(segment_df)
        
        # Add the last message to the prompt up to the username
        #prompt = pre_prompt + middle_prompt + "\n" + str(last_message["id"]) + ";" + last_message["time"] + ";" + str(last_message["from"]) + ";"
        prompt = pre_prompt + middle_prompt + "[MSG]" + str(last_message["id"]) + "[FS]" + last_message["time"] + "[FS]" + str(last_message["from"]) + "[FS]"
        
        print(f"Prompt {idx}\n----------------------------------- \n", prompt, "\n-----------------------------------")
        pred_resp = lang_model.get_only_new_tokens(prompt, temperature=0.7, max_new_tokens=50)
        true_resp = last_message["text"] + "[FS]" + str(last_message["reply_to_message_id"]) + "[MEG]"
        print(f"Predicted response: {pred_resp}")
        print(f"True response: {true_resp}")
        
        
    

if __name__ == "__main__":
    #MODEL_NAME = "TurkuNLP/gpt3-finnish-small"
    MODEL_NAME = "gpt3-xl-finetuned-v4"
    #MODEL_NAME = "finetuned_model"
    SAVE_PATH = "./finetuned_model"
    TEST_RUN = False
    RUN_EVALS = [True, False]
    
    # reset backend
    torch.cuda.empty_cache()
    train_data_df, test_data_df = load_dataset("_chat_history.csv")
    if TEST_RUN:
        train_data_df = train_data_df[:100]
        test_data_df = test_data_df[:100]
    
    model, tokenizer = load_model(MODEL_NAME)
    
    train_ds = ChatHistoryDataset(train_data_df, tokenizer)
    test_ds = ChatHistoryDataset(test_data_df, tokenizer)
    
    if RUN_EVALS[0]:
        # Test model on random prompts
        pre_prompt = f"Nimeni on FinGPT_bot ja olen hauska ja ystävällinen Teekkari tekoäly LUT:sta. Harrastan komiikkaa ja koodausta.\n"
        test_on_prompts(model, tokenizer, test_ds, num_prompts=5, pre_prompt=pre_prompt)
    exit()
    
    print(f"Number of segments in train: {len(train_ds)}")
    print(f"Number of segments in test: {len(test_ds)}")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        do_eval=True,
        do_train=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )
    
    trainer.train()
    # Save model and tokenizer
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
    if RUN_EVALS[1]:
        # Move everything to the same device before inference
        model.to("cuda")
        # Test model on random prompts
        test_on_prompts(model, tokenizer, test_ds, num_prompts=5, pre_prompt="")
    
    
    
    
    
    
    
    