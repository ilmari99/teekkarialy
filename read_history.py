""" Read a chat history exported from Telegram, and use it to fine tune a GPT-3 model. """
import json
import pandas as pd

def read_chat_history(filepath):
    """ Read a chat history json exported from Telegram. """
    with open(filepath, 'r',encoding="utf-8") as f:
        data = json.load(f)
    messages = data['messages']
    return messages

def parse_username(username):
    """ Parse the username from a Telegram username. """
    split_chars = [".", "!", "?", " ", "_", "-"]
    for split_char in split_chars:
        username = username.split(split_char)[0]
    return username

def parse_names_and_messages(messages):
    """ Parse the names and messages from the json. """
    names_parsed = []
    messages_parsed = []
    for message in messages:
        if 'from' not in message:
            print("No from in message")
            continue
        message_text = ""
        # Message is either the text of the message, or the emoji of the sticker
        if 'text_entities' in message and message['text_entities']:
            for entity in message['text_entities']:
                if entity['type'] == 'plain':
                    message_text = entity['text']
                    break
        elif 'sticker_emoji' in message:
            message_text = message['sticker_emoji']
        if message_text:
            messages_parsed.append(message_text)
            uname = parse_username(message['from'])
            names_parsed.append(uname)
    return names_parsed, messages_parsed

if __name__ == "__main__":
    path = "__chat_export__.json"
    messages = read_chat_history(path)
    print(f"Number of messages: {len(messages)}")
    names, messages = parse_names_and_messages(messages)
    print(f"Read {len(messages)} messages from {len(names)} users.")
    # Save to file
    df = pd.DataFrame({'sender': names, 'message': messages})
    df.to_csv("_chat_history.csv", index=False, encoding="utf-8", sep=";")