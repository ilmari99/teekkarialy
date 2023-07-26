import time
import datetime
from typing import Dict
from telebot.types import Message, User
import pandas as pd
import utils
import json
from utils import BEGIN_MSG_DELIMITER, END_MSG_DELIMITER, SEPARATOR

def message_2_message_json(message : Message) -> Dict:
    """ Convert a Message object to a message json.
    """
    return message.json

def message_string_2_df_row(message_string : str) -> pd.Series:
    """ Convert a message string to a row of a dataframe.
    Ideally:
    All message info is between BEGIN_MSG_DELIMITER and END_MSG_DELIMITER.
    This info is then split with SEPARATOR.
    
    Truthfully:
    Since the message is created by an AI, it may not follow the ideal format.
    Hence:
    If message cannot be parsed, return None.
    """
    for delimiter in [BEGIN_MSG_DELIMITER, END_MSG_DELIMITER]:
        if delimiter not in message_string:
            return None
    # Remove the delimiters
    message_string = message_string.replace(BEGIN_MSG_DELIMITER, "").replace(END_MSG_DELIMITER, "")
    # Split the message
    message_split = message_string.split(SEPARATOR)
    # The message split must have 4 (if not a reply) or 5 (if a reply) elements
    if len(message_split) not in [4,5]:
        return None
    message_info = {}
    message_info["id"] = message_split[0]
    message_info["time"] = message_split[1]
    message_info["from"] = message_split[2]
    message_info["text"] = message_split[3]
    message_info["reply_to_message_id"] = message_split[4] if len(message_split) == 5 else None
    return pd.Series(message_info)
    
    

def message_2_string(message : Message) -> str:
    """ Convert a Message object to a string that represents the message.
    This is used to input a series of messages to the model.
    """
    msg_string = BEGIN_MSG_DELIMITER + str(message.id) \
    + SEPARATOR + utils.unix_time_to_ISO(message.date)  \
    + SEPARATOR + utils.parse_username(message.from_user.username)  \
    + SEPARATOR + utils.parse_message_text(message.text)    \
    + SEPARATOR + str(message.reply_to_message.id)  \
    + END_MSG_DELIMITER
    return msg_string

def message_json_2_df_row(message_json : Dict):
    """ Convert a message json to a row of a dataframe.
    """
    msg_id = message_json["id"]
    send_time = utils.data_datestamp_to_time(message_json["date"])
    sender = utils.parse_username(message_json.get("from",""))
    message_text = utils.parse_message_text(message_json.get("text",""))
    reply_id = message_json.get("reply_to_message_id",None)
    sticker_emoji = message_json.get("sticker_emoji","")
    
    # From 'read_tg_export.py'
    # If text is empty and sticker_emoji is not, change text to sticker_emoji
    message_text = sticker_emoji if message_text == "" and sticker_emoji != "" else message_text
    # If text (and sticker_emoji) is empty, change text to "(image or file)"
    message_text = "(image or file)" if message_text == "" else message_text
    
    return pd.Series({"id": msg_id, "time": send_time, "from": sender, "text": message_text, "reply_to_message_id": reply_id})
    

def message_df_row_2_string(row : pd.Series,only_prompt_string = False) -> str:
    """ Convert a row of a message dataframe to a string that represents the message.
    This is used to input a series of messages to the model.
    """
    msg_string = BEGIN_MSG_DELIMITER + str(row["id"]) + SEPARATOR + str(row["time"]) + SEPARATOR + str(row["from"]) + SEPARATOR
    if only_prompt_string:
        return msg_string
    msg_string += str(row["text"]) + SEPARATOR + str(row["reply_to_message_id"]) + END_MSG_DELIMITER
    return msg_string

def message_2_df_row(message : Message) -> pd.Series:
    """ Convert a Message object to a row of a dataframe.
    """
    msg_id = message.message_id
    send_time = utils.data_datestamp_to_time(utils.unix_time_to_ISO(message.date))
    sender = utils.parse_username(message.from_user.username)
    reply_id = message.reply_to_message.message_id if message.reply_to_message else ""
    sticker_emoji = message.sticker.emoji if message.sticker else None
    message_text = utils.parse_message_text(message.text) if message.text else ""
    
    # From 'read_tg_export.py'
    # If text is empty and sticker_emoji is not, change text to sticker_emoji
    message_text = sticker_emoji if message_text == "" and sticker_emoji is not None else message_text
    # If text (and sticker_emoji) is empty, change text to "(image or file)"
    message_text = "(image or file)" if message_text == "" else message_text
    
    return pd.DataFrame({"id": msg_id, "time": send_time, "from": sender, "text": message_text, "reply_to_message_id": reply_id}, index=[0])