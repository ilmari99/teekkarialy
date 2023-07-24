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
from message_conversion import message_2_df_row, message_df_row_2_string, message_string_2_df_row
from utils import BEGIN_MSG_DELIMITER, END_MSG_DELIMITER, SEPARATOR


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
        self.pre_prompt = pre_prompt + "\n"
        self.tg_bot = telebot.TeleBot(self.access_token)
        self.lang_model = LanguageModel(self.model_name)
        print("Bot initialized")
        
    def _init_last_messages(self, chat_id):
        """ Initialize the last_messages dict, by putting an empty dataframe for the chat_id.
        """
        # Create a csv file for the chat
        os.makedirs("ChatDatas/", exist_ok=True)
        if not os.path.exists("ChatDatas/" + str(chat_id) + ".csv"):
            # Write headers
            with open("ChatDatas/" + str(chat_id) + ".csv", "a") as f:
                f.write("id;time;from;text;reply_to_message_id\n")
            self.last_messages[chat_id] = pd.DataFrame(columns=["id", "time", "from", "text", "reply_to_message_id"])
        else:
            self.last_messages[chat_id] = pd.read_csv("ChatDatas/" + str(chat_id) + ".csv", sep=";")
            # Get at most n_messages messages
            self.last_messages[chat_id] = self.last_messages[chat_id].tail(self.n_messages)
        return
    
    
    def dataframe_to_prompt(self, df, add_pre_prompt=True, add_post_prompt=True):
        """ Convert a chat history dataframe to a prompt.
        [MSG]<id>[FS]<time>[FS]<sender>[FS]<message>[FS]<reply_to_message_id>[MEG]
        """
        prompt = self.pre_prompt if add_pre_prompt else ""
        for i, row in df.iterrows():
            prompt += message_df_row_2_string(row) + "\n"
        if add_post_prompt:
            new_msg_id = df.iloc[-1]["id"] + 1
            new_msg_sent_time = get_curr_time()
            sender = self.tg_name
            series_to_convert = pd.Series({"id": new_msg_id, "time": new_msg_sent_time, "from": sender, "text": "", "reply_to_message_id": None})
            prompt += message_df_row_2_string(series_to_convert, only_prompt_string=True)
        return prompt
        
    def parse_username(self, username):
        return parse_username(username)
        
    def run(self):
        self.tg_bot.infinity_polling()
        
    def store_item(self, message):
        """ Store the sent message
        """
        new_df = message_2_df_row(message)
        # Check if known chat
        if not message.chat.id in self.last_messages:
            self._init_last_messages(message.chat.id)
            
        # Save to last_messages and to the csv file
        new_df.to_csv("ChatDatas/" + str(message.chat.id) + ".csv", mode="a", header=False, index=False, sep=";")
        self.last_messages[message.chat.id] = pd.concat([self.last_messages[message.chat.id], new_df], ignore_index=True)
        # Remove the oldest message as long as the prompt is too long
        # TODO: change threshold to variable
        while self.get_n_tokens(self.dataframe_to_prompt(self.last_messages[message.chat.id])) > 450:
            self.last_messages[message.chat.id].drop(self.last_messages[message.chat.id].head(1).index, inplace=True)
        
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
        Return a list of tuples, where each tuple is (message, reply_to_message_id).
        If the reply field is empty, reply_to_message_id is None.
        """
        send_messages = []
        # Split the response string
        replies = text_reply.split(END_MSG_DELIMITER)
        for reply in replies:
            reply += END_MSG_DELIMITER
        # Parse the first reply
        first_reply = replies.pop(0).split(SEPARATOR) if SEPARATOR in replies[0] else None
        if first_reply is None:
            return []
        first_reply_text = first_reply[0]
        first_reply_to_id = first_reply[1]
        send_messages.append((first_reply_text, first_reply_to_id))
        try:
            for reply in replies:
                reply_series = message_string_2_df_row(reply)
                if reply_series is None:
                    break
                # If the sender is not GPT, stop sending messages, or if the message is '(image or file)'
                if reply_series["from"] == self.tg_name and reply_series["text"] != "(image or file)":
                    send_messages.append((reply_series["text"], reply_series["reply_to_message_id"]))
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
        response = self.lang_model.get_only_new_tokens(prompt, temperature=0.3, max_new_tokens=80)
        print(f"Response\n----------------------------------- \n", response, "\n-----------------------------------")
        responses = self.parse_response(response)
        return responses

    
    
    
    
