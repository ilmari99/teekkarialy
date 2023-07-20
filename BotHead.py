import json
import os
from typing import List, Tuple
import requests
from LanguageModel import LanguageModel
import telebot
import random
from utils import parse_username, data_datestamp_to_time, get_curr_time, parse_message_text, unix_time_to_ISO
import time
import pandas as pd
from telebot.types import Message


""" The telegram bots head.

The app defines a bot.
When a message arrives, the handlers are called in order until a handler defined by the bot handles the message.

The handlers make actions through the bot.


"""

class BotHead:
    def __init__(self, model_name, access_token, n_messages = 10, tg_name="FinGPT_bot"):
        self.model_name = model_name
        self.access_token = access_token
        self.n_messages = n_messages
        self.tg_name = tg_name
        self.last_messages : dict[int, pd.DataFrame] = {}
        pre_prompt = f"Nimeni on {self.tg_name} ja olen hauska ja ystävällinen Teekkari tekoäly LUT:sta. Harrastan komiikkaa ja koodausta."
        self.pre_prompt = pre_prompt
        self.tg_bot = telebot.TeleBot(self.access_token)
        self.lang_model = LanguageModel(self.model_name)
        print("Bot initialized")
        
    def _init_last_messages(self, chat_id):
        """ Initialize the last_messages dict, by putting an empty dataframe for the chat_id.
        """
        self.last_messages[chat_id] = pd.DataFrame(columns=["id", "time", "from", "text", "reply_to_message_id"])
        # Create a csv file for the chat
        os.makedirs("ChatDatas/", exist_ok=True)
        # Write headers
        with open("ChatDatas/" + str(chat_id) + ".csv", "a") as f:
            f.write("id;time;from;text;reply_to_message_id\n")
        return
            
        
    def dataframe_to_prompt(self, df, add_pre_prompt=True, add_post_prompt=True):
        """ Convert a chat history dataframe to a prompt.
        [MSG]<id>[FS]<time>[FS]<sender>[FS]<message>[FS]<reply_to_message_id>[MEG]
        """
        prompt = self.pre_prompt if add_pre_prompt else ""
        for i, row in df.iterrows():
            prompt += "[MSG]" + str(row["id"]) + "[FS]" + row["time"] + "[FS]" + str(row["from"]) + "[FS]" + row["text"] + "[FS]" + str(row["reply_to_message_id"]) + "[MEG]\n"
            #f"{row['id']};{row['time']};{row['from']};{row['text']};{row['reply_to_message_id']}\n"
        if add_post_prompt:
            new_msg_id = df.iloc[-1]["id"] + 1
            new_msg_sent_time = get_curr_time()
            sender = self.tg_name
            prompt += "[MSG]" + str(new_msg_id) + "[FS]" + new_msg_sent_time + "[FS]" + sender + "[FS]"
            #str(new_msg_id) + ";" + new_msg_sent_time + ";" + sender + ";"
        return prompt
        
    def parse_username(self, username):
        return parse_username(username)
        
    def run(self):
        self.tg_bot.infinity_polling()
        
    def store_item(self, message):
        """ Store the sent message
        """
        msg_id = message.message_id
        send_time = data_datestamp_to_time(unix_time_to_ISO(message.date))
        sender = self.parse_username(message.from_user.username)
        message_text = parse_message_text(message.text)
        reply_id = message.reply_to_message.message_id if message.reply_to_message else None
        sticker_emoji = message.sticker.emoji if message.sticker else None
        
        # From 'read_tg_export.py'
        # If text is empty and sticker_emoji is not, change text to sticker_emoji
        message_text = sticker_emoji if message_text == "" and sticker_emoji != "" else message_text
        # If text (and sticker_emoji) is empty, change text to "(image or file)"
        message_text = "(image or file)" if message_text == "" else message_text
        
        # Check if known chat
        if not message.chat.id in self.last_messages:
            self._init_last_messages(message.chat.id)
            
        # Concatenate the new message (or changed version) to the dataframe
        new_df = pd.DataFrame([[msg_id, send_time, sender, message_text, reply_id]], columns=["id", "time", "from", "text", "reply_to_message_id"])
        # Save to last_messages and to the csv file
        new_df.to_csv("ChatDatas/" + str(message.chat.id) + ".csv", mode="a", header=False, index=False, sep=";")
        self.last_messages[message.chat.id] = pd.concat([self.last_messages[message.chat.id], new_df], ignore_index=True)
        #print(f"Stored message\n {self.last_messages[message.chat.id].iloc[-1]}")
        # Remove the oldest message if there are too many
        self.last_messages[message.chat.id] = self.last_messages[message.chat.id].tail(self.n_messages)
        
    def get_n_tokens(self, text):
        """ Calculate the number of tokens in text
        """
        return len(self.lang_model.tokenizer(text)["input_ids"])
        
    def send_message_wrapper(self, chat_id, message_text, reply_to_message_id=None, max_send_tries=2):
        try:
            tries = 0
            success = False
            while not success:
                try:
                    sent_msg = self.tg_bot.send_message(chat_id, message_text, reply_to_message_id=reply_to_message_id, allow_sending_without_reply=True)
                    success = True
                except Exception as e:
                    success = False
                tries += 1
                if tries > max_send_tries:
                    print(f"Sending message {message_text} failed.")
                    raise Exception("Sending message failed.")
        except Exception as e:
            return False
        self.store_item(sent_msg)
        self.tg_name = self.parse_username(sent_msg.from_user.username)
        return True
    
    def parse_response(self, text_reply : str) -> List[Tuple[str, int]]:
        """ Parse the response from the model.
        The first line is parsed as the reply, and the rest are sent separately as long as the sender is "GPT".
        When the first sender is not "GPT", the rest of the messages are not sent.
        ```
        <message>[FS]<reply_to_message_id>[MEG]
        [MSG]id[FS]time[FS]sender[FS]message[FS]reply_to_message_id[MEG]
        ...
        ```       
        
        Return a list of tuples, where each tuple is (message, reply_to_message_id).
        If the reply field is empty, reply_to_message_id is None.
        """
        send_messages = []
        # Split the response string
        replies = text_reply.split("[MEG]")
        # Parse the first reply
        first_reply = replies.pop(0).split("[FS]")
        first_reply_text = first_reply[0]
        first_reply_to_id = first_reply[1]
        send_messages.append((first_reply_text, first_reply_to_id))
        try:
            for reply in replies:
                reply.replace("[MSG]", "")
                split_reply = reply.split("[FS]")
                reply_id = split_reply[0]
                reply_time = split_reply[1]
                reply_sender = split_reply[2]
                reply_text = split_reply[3]
                reply_to_id = split_reply[4]
                if reply_sender != self.tg_name:
                    break
                send_messages.append((reply_text, reply_to_id))
        except Exception as e:
            pass
        return send_messages
    
    
    def create_replies(self, chat_id):
        """ create a list of messages (text,id) to send based on the latest messages.
        """
        # Get the last messages
        if not chat_id in self.last_messages:
            self._init_last_messages(chat_id)
        last_messages = self.last_messages[chat_id]
        # Convert to prompt
        prompt = self.dataframe_to_prompt(last_messages)
        print(f"Prompt\n----------------------------------- \n", prompt, "\n-----------------------------------")
        # Get the response
        response = self.lang_model.get_only_new_tokens(prompt, temperature=0.6, max_new_tokens=120)
        print(f"Response\n----------------------------------- \n", response, "\n-----------------------------------")
        responses = self.parse_response(response)
        return responses
        for msg, reply_id in responses:
            success = self.send_message_wrapper(chat_id, msg, reply_to_message_id=reply_id)
            if not success:
                # Try 2 times
                success = self.send_message_wrapper(chat_id, msg, reply_to_message_id=reply_id)
                if not success:
                    print(f"Sending message {msg} failed.")
                    break

    
    
    
    
