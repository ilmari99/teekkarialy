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
from LanguageModel import OpenAILanguageModel
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
        
        pre_prompt = (f"Nimeni on {self.tg_name} ja olen Teekkariäly. "
                           "Seuraan keskustelua,ja yritän olla hauska. "
                           "En ota juttuja liian vakavasti, vaan olen aina valmis vitsailemaan mistä tahansa,"
                           "jopa vähän kyseenalaisista aiheista."
        )
        self.pre_prompt = (f"Nimesi on {self.tg_name} ja olet Teekkariäly. "
                      "Seuraat keskustelua "
                      "ja olet hauska. Et ota juttuja liian vakavasti, "
                      "vaan olet aina valmis vitsailemaan mistä tahansa, "
                      "jopa kyseenalaisista aiheista."
        )
        self.pre_prompt = ""
        
        self.post_prompt = ""
        
        mandatory_post_prompt = ("Jos en kuulu keskusteluun "
                                      "tai minulla ei ole mitään sanottavaa, "
                                      "vastaan pelkästään 'pass'. "
                                      "Jos tahdon vastata johonkin tiettyyn "
                                      "viestiin, lisään vastaukseni alkuun @<id>."
        )
        self.mandatory_post_prompt = ("Jos et kuulu keskusteluun "
                                      "tai sinulla ei ole mitään sanottavaa, "
                                      "vastaa pelkästään 'pass'. "
                                      "Jos tahdot vastata johonkin tiettyyn "
                                      "viestiin, lisää vastauksesi alkuun @<message-id>.")
        
        self.tg_bot = telebot.TeleBot(self.access_token)
        with open("__openai_api_key.txt", "r") as f:
            self.lang_model = OpenAILanguageModel(f.read(), model = self.model_name)
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
            if from_ == "FinGPTbot":  
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
        if responses[0][0][:4].lower() == "pass":
            responses = []
        return responses
    
    def parse_response(self, text_reply : str) -> List[Tuple[str, int]]:
        """ Parse the response from the model.
        The response is a string, with optionally '@<id>' at the beginning of the message.
        """
        reply_to_id = None
        if text_reply[0] == "@":
            msg_begin_idx = text_reply.find(" ")
            try:
                reply_to_id = int(text_reply[1:msg_begin_idx])
            except ValueError:
                warnings.warn(f"Message ID is supposed to be an integer, but got {text_reply[1:msg_begin_idx]}.")
                reply_to_id = None
            text_reply = text_reply[msg_begin_idx+1:]
        return [(text_reply, reply_to_id)]

def remove_illegal_quotes(msg_for_json):
    # The string is e.g. "{"id": 1,"time": "2021-05-01 12:00:00", "text": "Terve", "reply_to_message_id": None}"
    # Check if there are any nested quotes in values/keys 
    # (Remove for example {"text" : "Then he said "Hello" to me."}), where you would substitute "Hello" with 'Hello'
    # Change all double quotes inside the inner curly brackets to single quotes
    # ({"text" : "Then he said 'Hello' to me."})
    
    # Find the first inner curly bracket
    first_inner_curly_bracket_idx = msg_for_json.find("{", 1)
    # Find the last inner curly bracket
    last_inner_curly_bracket_idx = msg_for_json.rfind("}")
    sub_str = msg_for_json[first_inner_curly_bracket_idx:last_inner_curly_bracket_idx]
    # Find all double quotes inside the inner curly brackets
    double_quote_idxs = [m.start() for m in re.finditer("\"", sub_str)]
    
    # Change all double quotes to single quotes
    for idx in double_quote_idxs:
        msg_for_json = msg_for_json[:idx] + "'" + msg_for_json[idx+1:]
    return msg_for_json

    

    

if __name__ == "__main__":
    df = pd.read_csv("combined_chat_history.csv", sep=";")
    bot = GPTBotHead("gpt-3.5-turbo", "TOKEN", 2048, "Teekkariäly")
    
    bot.lang_model.client.fine_tuning.jobs.create(model = "gpt-3.5-turbo",
                                                  training_file="file-1RS1q64yFWWeJD1GIyK7p7Id",
                                                  validation_file="file-JP7ehTGFtKItQkUtnZ7XU8BW",
                                                  hyperparameters={"n_epochs" : 1},
                                                  suffix = "lateksii"
                                                  )
    exit()
    # Format the messages to batches of 10 messages,
    # And convert them to json:
    # {"messages": [
    #     {"role" : "user", "name" : "Jaakko", "content" : "{'id': 1,'time': '2021-05-01 12:00:00', 'text': 'Terve', 'reply_to_message_id': None}"},
    #     {"role" : "user", "name" : "Mikko", "content" : "{'id': 2,'time': '2021-05-01 12:01:32', 'text': 'Moroo', 'reply_to_message_id': 1}"},
    #]}
    # Then, change the last message role to "assistant", and the content to "@<reply_to_message_id> content["text"]""
    # We then save these examples in jsonl format.
    
    # Convert the dataframe to json
    json_str = bot.dataframe_to_prompt(df, add_pre_prompt=False, add_system_post_prompt=False)
    
    # Load the json
    json_dict = json.loads(json_str)
    num_failed_messages = 0
    # Change every 10th message to match the format, and save to jsonl
    jsonl = []
    for i in range(0,len(json_dict["messages"]),10):
        # Make sequences of 10 messages
        message_sequence = []
        for j in range(i,i+10):
            if j >= len(json_dict["messages"]):
                break
            message_sequence.append(copy.copy(json_dict["messages"][j]))
        # If the sequence is too short, skip it
        if len(message_sequence) < 10:
            continue
        # Change the last message to assistant
        sender = message_sequence[-1]["name"]
        messages_sent_by_last_person = [m for m in message_sequence if m["name"] == sender]
        for message in messages_sent_by_last_person:
            sender = message["name"]
            #print(f"Original message: {message}")
            message["role"] = "assistant"
            # {'id': 2, 'time': '23:45', 'from': 'Anonym', 'text': '(image or file)', 'reply_to_message_id': nan}
            msg_for_json = message["content"].replace("\"", "'").replace("nan", "null").replace("'", "\"")
            # Check if there are any nested quotes in values/keys 
            # (Remove for example {"text" : "Then he said "Hello" to me."}), where you would substitute "Hello" with \"Hello\"
            msg_for_json = remove_illegal_quotes(msg_for_json)
            try:
                msg_json = json.loads(msg_for_json)
            except json.decoder.JSONDecodeError as e:
                num_failed_messages += 1
                print(f"Error in message {msg_for_json}")
                print(e)
                continue
            try:
                reply_to_id = int(msg_json["reply_to_message_id"])
            except TypeError:
                reply_to_id = None
            if reply_to_id is not None:
                message["content"] = f"@{reply_to_id} {msg_json['text']}"
            else:
                message["content"] = msg_json["text"]
            # Create a json with all the messages
            del message["name"]
        # Add a preprompt
        message_sequence.insert(0, {"role" : "system", "content" : "Olet osana Laskennallisen tekniikan (teekkarien) keskustelua. Koita sopeutua mahdollisimman hyvin joukkoon, äläkä kerro olevasi tekoäly."})
        message_sequence_json = {"messages": message_sequence}
        jsonl.append(message_sequence_json)
            
    print(f"Number of failed messages: {num_failed_messages}")
    # Shuffle and save 20% to validation set
    import random
    random.shuffle(jsonl)
    n_validation = len(jsonl) // 5
    jsonl_train = jsonl[:-n_validation]
    jsonl_validation = jsonl[-n_validation:]
    # Save to jsonl
    with open("openai_finetune_train.jsonl", "w") as f:
        for json_dict in jsonl_train:
            json_str = json.dumps(json_dict, ensure_ascii=False)
            f.write(json_str + "\n")
    with open("openai_finetune_validation.jsonl", "w") as f:
        for json_dict in jsonl_validation:
            json_str = json.dumps(json_dict, ensure_ascii=False)
            f.write(json_str + "\n")
        
    
    
        