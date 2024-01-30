""" Read a chat history exported from Telegram, and use it to fine tune a GPT-3 model. """
import json
import os
import pandas as pd
from utils import parse_username, data_datestamp_to_time, get_curr_time, parse_message_text
from message_conversion import message_json_2_df_row

def read_chat_history(filepath):
    """ Read a chat history json exported from Telegram.
    Return a list of messages.
    """
    with open(filepath, 'r',encoding="utf-8") as f:
        data = json.load(f)
    messages = data['messages']
    return messages

    

if __name__ == "__main__":
    data_folder = "data"
    # Each folder in data folder contains a 'result.json' file
    # containing the chat history
    for i,folder in enumerate(os.listdir(data_folder)):
        if "OldData" in folder.lower():
            continue
        print(f"Reading chat history from {folder}...")
        path = os.path.join(data_folder, folder, "result.json")
        
        messages = read_chat_history(path)[1:]
        print(f"Number of messages: {len(messages)}")
        
        print("First message:")
        print(messages[0])
        messages = [message_json_2_df_row(m) for m in messages]
        df = pd.DataFrame(messages)
        print(f"Shape of dataframe: {df.shape}")
        
        # Write to csv
        if i == 0:
            df.to_csv("combined_chat_history.csv", index=False,mode="w",header=True, sep=";")
        else:
            df.to_csv("combined_chat_history.csv", index=False,mode="a",header=False, sep=";")