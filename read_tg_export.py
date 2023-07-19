""" Read a chat history exported from Telegram, and use it to fine tune a GPT-3 model. """
import json
import pandas as pd
from utils import parse_username, data_datestamp_to_time, get_curr_time, parse_message_text

def read_chat_history(filepath):
    """ Read a chat history json exported from Telegram.
    Return a list of messages.
    """
    with open(filepath, 'r',encoding="utf-8") as f:
        data = json.load(f)
    messages = data['messages']
    return messages

    

if __name__ == "__main__":
    path = "_private_data.json"
    messages = read_chat_history(path)
    print(f"Number of messages: {len(messages)}")
    # only take date, sender and message
    df = pd.DataFrame(messages)[["id", "date", "from", "text","sticker_emoji", "reply_to_message_id"]]
    # Set datatypes
    df["id"] = df["id"].astype(int)
    # Set dtype of 'reply_to_message_id' to int if not Nan
    df["reply_to_message_id"] = df["reply_to_message_id"].astype(pd.Int64Dtype())
    
    # Convert date format
    df["date"] = df["date"].apply(data_datestamp_to_time)
    # Convert message texts
    df["text"] = df["text"].apply(parse_message_text)
    
    # If text is empty and sticker_emoji is not, change text to sticker_emoji
    df["text"] = df.apply(lambda x: x["sticker_emoji"] if x["text"] == "" and x["sticker_emoji"] != "" else x["text"], axis=1)
    
    # If text (and sticker_emoji) is empty, change text to "(image or file)"
    df["text"] = df["text"].apply(lambda x: "(image or file)" if pd.isna(x) or x == "" else x)
    
    # Pop sticker_emoji column, since it's not needed anymore
    df.pop("sticker_emoji")
    
    # Drop all rows with sender Nan (should only be the first row)
    df.dropna(subset=["from"], inplace=True)
    
    # rename date to time
    df.rename(columns={"date": "time"}, inplace=True)
    #Show dtypes
    print(df.dtypes)
    
    df.to_csv("_chat_history.csv", index=False, encoding="utf-8", sep=";")