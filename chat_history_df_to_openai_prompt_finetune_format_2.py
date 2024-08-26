import pandas as pd
from OpenAIGPTHead import GPTBotHead
import json
import copy
import re
import argparse
import random
import datetime

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
    parser = argparse.ArgumentParser(description="Convert chat history dataframe to OpenAI prompt finetune format.")
    # All arguments are optional
    parser.add_argument("--dataframe_path", type=str, default="combined_chat_history.csv", help="Path to the chat history dataframe.")
    parser.add_argument("--message_sequence_length", type=int, default=15, help="Number of messages in each sequence.")
    parser.add_argument("--train_test_split_fraction", type=float, default=0.2, help="Fraction of the data to use for validation.")
    args = parser.parse_args()
    
    system_prompt = ("You are part of a Telegram group chat between Computational Engineering students from Finland.",
                     "Keskustelu on suomeksi, ja opiskelijat ovat Teekkareita Lappeenrannan teknillisestä yliopistosta.",
                     "Keskustelu on kavereiden keskistä, ja aiheet vaihtelevat laidasta laitaan.",
                     "Koita sulautua joukkoon, äläkä paljasta että olet Tekoäly.",
                     "Käytä Teekkarihuumoria jos se sopii tilanteeseen.",
                     "Jos tahdot vastata johonkin aiempaan viestiin, lisää '@<id> ' ennen tekstiä.",
    )
    system_prompt = {"role": "system", "content": " ".join(system_prompt)}
    
    dataframe_path = args.dataframe_path
    message_sequence_length = args.message_sequence_length
    train_test_split_fraction = args.train_test_split_fraction
    
    df = pd.read_csv(dataframe_path, sep=";")
    bot = GPTBotHead("gpt-4o-mini", "token", 2048, "Teekkariäly")
    # Format the messages to batches of message_sequence_length messages,
    # And convert them to json:
    # ...
    # Then, change the last message role to "assistant", and the content to "@<reply_to_message_id> content["text"]""
    # We then save these examples in jsonl format.
    
    # Convert the dataframe to json
    json_str = bot.dataframe_to_prompt(df, add_pre_prompt=False, add_system_post_prompt=False)
    
    # Load the json
    json_dict = json.loads(json_str)
    num_failed_messages = 0
    num_discarded_sequences = 0
    # Change every message_sequence_length-th message to match the format, and save to jsonl
    jsonl = []
    for i in range(0,len(json_dict["messages"]) - message_sequence_length, message_sequence_length):
        # Make sequences of message_sequence_length messages
        message_sequence = []
        prev_id = None
        # hh:mm
        seq_start_time = datetime.datetime.strptime(df.iloc[i]["time"], "%H:%M")
        for j in range(i,i+message_sequence_length):
            curr_id = df.iloc[j]["id"]
            curr_time = datetime.datetime.strptime(df.iloc[j]["time"], "%H:%M")
            #print(f"i: {i}, j: {j}, curr_id: {curr_id}, prev_id: {prev_id}")
            #if j >= len(json_dict["messages"]):
            #    break
            # If the messages are not part of the same chat, then well save a shorter sequence
            if prev_id is not None and abs(curr_id - prev_id) > 4:
                print(f"Breaking sequence (current length {len(message_sequence)}), because the message IDs are too much apart: curr_id: {curr_id}, prev_id: {prev_id}")
                break
            # If the time difference between the first and the current message is over 1 hour, then we save a shorter sequence
            if (curr_time - seq_start_time).seconds > 3600:
                print(f"Breaking sequence (current length {len(message_sequence)}), because the time difference is too large: curr_time: {curr_time}, seq_start_time: {seq_start_time}")
                break
            prev_id = curr_id
            message_sequence.append(copy.copy(json_dict["messages"][j]))
        # If the sequence is too short, skip it
        if len(message_sequence) < 10:
            num_discarded_sequences += 1
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
            # Remove the name attribute for assistant messages
            del message["name"]
        # Add a preprompt
        message_sequence.insert(0, system_prompt)
        message_sequence_json = {"messages": message_sequence}
        jsonl.append(message_sequence_json)
            
    print(f"Number of failed messages: {num_failed_messages}")
    print(f"Number of discarded sequences: {num_discarded_sequences}")
    random.shuffle(jsonl)
    n_validation = int(len(jsonl) * train_test_split_fraction)
    jsonl_train = jsonl[:-n_validation]
    jsonl_validation = jsonl[-n_validation:]
    # Save to jsonl
    with open("openai_finetune_train_2.jsonl", "w") as f:
        for json_dict in jsonl_train:
            json_str = json.dumps(json_dict, ensure_ascii=False)
            f.write(json_str + "\n")
    with open("openai_finetune_validation_2.jsonl", "w") as f:
        for json_dict in jsonl_validation:
            json_str = json.dumps(json_dict, ensure_ascii=False)
            f.write(json_str + "\n")