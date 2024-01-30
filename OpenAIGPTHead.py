import copy
import json
import os
import time
from typing import List, Tuple
import warnings
import pandas as pd
from BotHead import BotHead
from utils import END_MSG_DELIMITER, get_curr_time
from message_conversion import message_df_row_2_openai_json
import telebot
from LanguageModel import OpenAIChatModel
import ast
import re


class GPTBotHead(BotHead):
    """ This is an model, that instead of a local model, uses the OpenAI API to generate text.
    """
    
    def __init__(self, model_name, access_token, max_n_tokens = 2048, tg_name="Comic"):
        
        self.start_time = time.time()
        self.model_name = model_name
        self.access_token = access_token
        self.max_n_tokens = max_n_tokens
        self.tg_name = tg_name
        self.last_messages : dict[int, pd.DataFrame] = {}
        
        self.trigger_phrases = ["a", "i", "t", "n", "e"]
        
        self.bot_info = ("Moi! Olen GPT3.5 malliin pohjautuva tekoäly: "
                         "Teekkariäly. Minut on koulutettu keskustelemaan "
                         "teekkarien kanssa, ja yritän jatkaa keskustelua "
                         "luontevasti. Minä luen aina viestit, jos minulle saapuu viesti jossa on jotain seuraavista: "
                         f"{self.trigger_phrases}."
                         " Toimin myös Komennoilla:\n"
                         "\t- /gpt-help - Tietoa komennoista"
                         "\t- /vitsi [mistä] - Lähetä vitsi. 'mistä' voi specifioida mistä "
                         "aiheesta vitsi halutaan. Esim. '/vitsi Teemu'."
                         "\nMeillä tulee varmana olemaan hauskaa!"
        )

        self.pre_prompt = "Olet osana Laskennallisen tekniikan (teekkarien) keskustelua. Koita sopeutua mahdollisimman hyvin joukkoon, äläkä kerro olevasi tekoäly."
        
        self.post_prompt = ""
        
        self.mandatory_post_prompt = ("Jos et kuulu keskusteluun "
                                      "tai sinulla ei ole mitään sanottavaa, "
                                      "vastaa pelkästään 'pass'. "
                                      "Jos tahdot vastata johonkin tiettyyn "
                                      "viestiin, lisää vastauksesi alkuun @<message-id>.")
        
        self.tg_bot = telebot.TeleBot(self.access_token)
        with open("__openai_api_key.txt", "r") as f:
            self.lang_model = OpenAIChatModel(f.read().strip(), model=self.model_name, model_kwargs={"temperature" : 0.5, "frequency_penalty" : 0.5, "presence_penalty" : -0.3})
        print("Bot initialized")
        
    def get_joke_prompt(self, msg):
        # Joke is triggered by a message '/vitsi (aihe)'
        subject = msg.text.split(" ")[1:]
        if len(subject) == 0:
            return "Kerro random vitsi, joka mieluusti liittyy chatin historiaan, teekkariuteen, tai on muuten vaan hauska."
        return "Kerro vitsi aiheesta " + " ".join(subject)
        
    def get_n_tokens(self, text):
        """ Get the approximate number of tokens in the text.
        """
        l = len(text) // 3
        return l
    
    def parse_username(self, username):
        uname = super().parse_username(username)
        # Remove non alphanumeric characters and limit to 60 characters
        # Convert to ascii
        uname = "".join([c for c in uname if c.isalnum()])[:60]
        convert_map = {"ä" : "a", "ö" : "o", "å" : "a", "Ä" : "A", "Ö" : "O", "Å" : "A"}
        uname = uname.translate(str.maketrans(convert_map))
        # Convert the remaining characters to ascii
        uname = uname.encode("ascii", errors="ignore").decode()
        return uname

            
    
    def dataframe_to_prompt(self, df, add_pre_prompt=True, add_system_post_prompt = True) -> str:
        """ Convert a chat history dataframe to a prompt.
        The dataframe has the columns "id", "time", "from", "text", "reply_to_message_id",
        and rows represent previous messages.
        This function converts the dataframe to a string, that is used as a prompt for the API.
        The prompt has a json-like format:
        {"messages": [
            if add_pre_prompt:
                {"role" : "system", "content" : <pre_prompt>},
            {"role" : "user", "name" : "Jaakko", "content" : "{'id': 1,'time': '2021-05-01 12:00:00', 'text': 'Terve', 'reply_to_message_id': None}"},
            {"role" : "user", "name" : "Mikko", "content" : "{'id': 2,'time': '2021-05-01 12:01:32', 'text': 'Moroo', 'reply_to_message_id': 1}"},
            {"role" : "user", "name" : "Jaakko", "content" : "{'id': 3,'time': 2021-05-01 12:02:00, 'text': 'Mitä kuuluu?', 'reply_to_message_id': None}"},
            {"role" : "assistant", "name" : <bot_name>, "content" : "{'id': 4,'time': 2021-05-01 12:02:30, 'text': 'Terve vaan!', 'reply_to_message_id': 3}"},
        }
        From names, the whitespaces are removed.
        """
        prompt = {"messages": []}
        if add_pre_prompt:
            prompt["messages"].append({"role" : "system", "content" : self.pre_prompt})
        for _, row in df.iterrows():
            # If name is "FinGPTbot", change it to 'Teekkarialy' and role to 'assistant'
            from_ = self.parse_username(row["from"])
            if from_ in ["FinGPTbot", self.tg_name]:
                row["from"] = self.tg_name
                row["role"] = "assistant"
                prompt["messages"].append({"role" : "assistant", "name" : self.parse_username(self.tg_name), "content" : row["text"]})
            else:
                prompt["messages"].append({"role" : "user", "name" : self.parse_username(row["from"]), "content" : str(row.to_dict())})
        if add_system_post_prompt:
            prompt["messages"].append({"role" : "system", "content" : f"{self.post_prompt}{self.mandatory_post_prompt}"})
        # Return as a string, but keep äö etc.
        s = json.dumps(prompt, ensure_ascii=False)
        return s
    
    def make_joke(self, msg):
        """ Generate a joke, and send it.
        """
        # Get the prompt
        prompt = self.dataframe_to_prompt(self.last_messages[msg.chat.id], add_system_post_prompt=False)
        prompt = json.loads(prompt)
        # Prompt to make a joke
        joke_prompt = self.get_joke_prompt(msg)
        prompt["messages"].append({"role" : "system", "content" : joke_prompt})
        
        response = self.lang_model.generate_response(prompt).choices[0].message.content
        return response
    
    
    def create_replies(self, chat_id):
        """ create a list of messages (text,id) to send based on the latest messages.
        """
        # Get the last messages
        if not chat_id in self.last_messages:
            self._init_last_messages(chat_id)
        self.remove_trailing_last_messages(chat_id)
        last_messages = self.last_messages[chat_id]
        # Convert to prompt
        prompt_str = self.dataframe_to_prompt(last_messages)
        #print(prompt_str)
        # Load the prompt string, and keep äö etc.
        prompt = json.loads(prompt_str)
        print(f"Prompt\n----------------------------------- \n", json.dumps(prompt, indent=4, ensure_ascii=False), "\n-----------------------------------")
        # Get the response
        response = self.lang_model.generate_response(prompt).choices[0].message.content
        print(f"Response\n----------------------------------- \n", response, "\n-----------------------------------")
        responses = self.parse_response(response)
        # We should only have one response
        if len(responses) > 1:
            raise Exception("Too many responses.")
        # Check if the first four letters of the response are "pass"
        if responses and len(responses[0][0]) >= 4 and responses[0][0][:4].lower() == "pass":
            responses = []
        return responses
    
    def parse_response(self, text_reply : str) -> List[Tuple[str, int]]:
        """ Parse the response from the model.
        The response is a string, with optionally '@<id>' at the beginning of the message.
        """
        reply_to_id = None
        if len(text_reply) < 1:
            return []
        if text_reply[0] == "@":
            msg_begin_idx = text_reply.find(" ")
            try:
                reply_to_id = int(text_reply[1:msg_begin_idx])
            except ValueError:
                warnings.warn(f"Message ID is supposed to be an integer, but got {text_reply[1:msg_begin_idx]}.")
                reply_to_id = None
            text_reply = text_reply[msg_begin_idx+1:]
        return [(text_reply, reply_to_id)]

