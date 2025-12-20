import argparse
import datetime
import json
import random

import numpy as np
import pandas as pd
import re

from OpenAIGPTHead import GPTBotHead


def _not_nan(value) -> bool:
    return value is not None and (not isinstance(value, float) or not np.isnan(value))


_ASCII_CONVERT_MAP = {"ä": "a", "ö": "o", "å": "a", "Ä": "A", "Ö": "O", "Å": "A"}


def normalize_sender(sender: str) -> str:
    """Normalize sender name to lowercase ASCII alphanumerics for comparisons."""
    if sender is None:
        return ""
    s = str(sender).strip().translate(str.maketrans(_ASCII_CONVERT_MAP))
    s = s.encode("ascii", errors="ignore").decode()
    return "".join(ch for ch in s if ch.isalnum()).lower()


def format_message_content(row_dict: dict) -> str:
    # Convert a message row to the "user" content format expected by the model.
    reply_val = row_dict.get("reply_to_message_id")
    reply_id = int(reply_val) if _not_nan(reply_val) else None
    payload = {
        "id": int(row_dict["id"]),
        "time": row_dict["time"],
        "text": row_dict["text"],
        "reply_to_message_id": reply_id,
    }
    return str(payload)


def approx_message_chars(row_dict: dict) -> int:
    """Rough character budget for a row, using worst-case of user-format vs assistant-format."""
    text = str(row_dict["text"])
    reply_val = row_dict.get("reply_to_message_id")
    if _not_nan(reply_val):
        try:
            reply_len = len(str(int(reply_val))) + 2  # '@' + space
        except (ValueError, TypeError):
            reply_len = 0
    else:
        reply_len = 0

    user_repr = {
        "id": int(row_dict["id"]),
        "time": row_dict["time"],
        "text": row_dict["text"],
        "reply_to_message_id": int(reply_val) if _not_nan(reply_val) else None,
    }
    user_content_len = len(str(user_repr))
    assistant_len = reply_len + len(text)
    return max(user_content_len, assistant_len)


def parse_time_hhmm(t_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(t_str, "%H:%M")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert chat history dataframe to OpenAI prompt finetune format.")
    parser.add_argument("--dataframe_path", type=str, default="combined_chat_history.csv", help="Path to the chat history dataframe.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Approximate target tokens per example (chars = tokens * chars_per_token).")
    parser.add_argument("--chars_per_token", type=int, default=3, help="Approximate characters per token for budgeting.")
    parser.add_argument("--min_messages", type=int, default=4, help="Minimum messages per example (system + turns).")
    parser.add_argument("--train_test_split_fraction", type=float, default=0.05, help="Fraction of the data to use for validation.")
    parser.add_argument("--stride", type=int, default=2, help="Sliding window stride. 1 keeps all overlapping contexts.")
    args = parser.parse_args()

    system_prompt = {
        "role": "system",
        "content": (
            "You are a participant in a Finnish Telegram group chat of Computational Engineering students at LUT. "
            "Write naturally in Finnish with relaxed Teekkari humor when it fits. Blend in as a peer, don't reveal or hint you're an AI. "
            "Keep responses concise, like a real chat message. If replying to a specific earlier message, prefix with @<id>. "
            "Honor the informal, friendly tone; avoid overly formal language."
        ),
    }

    df = pd.read_csv(args.dataframe_path, sep=";")
    bot = GPTBotHead("gpt-4.1-mini", "token", 4096, "Teekkariäly")

    records: list[dict] = df.to_dict(orient="records")
    parsed_times: list[datetime.datetime | None] = []
    sender_lower: list[str] = []
    sender_norm: list[str] = []
    approx_chars: list[int] = []
    user_payloads: list[str] = []

    for rec in records:
        sender_lower.append(str(rec.get("from", "")).strip().lower())
        sender_norm.append(normalize_sender(rec.get("from", "")))

        try:
            parsed_times.append(parse_time_hhmm(rec["time"]))
        except (ValueError, TypeError):
            parsed_times.append(None)

        approx_chars.append(approx_message_chars(rec))
        user_payloads.append(format_message_content(rec))

    jsonl = []
    num_discarded_sequences = 0

    target_chars = args.max_tokens * args.chars_per_token

    username_cache: dict[str, str] = {}

    for start_idx in range(0, len(records), args.stride):
        window_rows: list[int] = []
        char_budget = len(system_prompt["content"])
        discard_sequence = False

        start_time = None
        last_time = None
        last_id = None

        for idx in range(start_idx, len(records)):
            row_dict = records[idx]
            curr_time = parsed_times[idx]

            # Skip sequences containing the excluded participant.
            if sender_lower[idx] == "teekkarialy" or sender_norm[idx] == "teekkarialy":
                discard_sequence = True
                break

            # ID continuity check
            if last_id is not None and abs(int(row_dict["id"]) - int(last_id)) > 4:
                break

            # Time continuity check with midnight handling
            if curr_time is None:
                discard_sequence = True
                break

            if start_time is None:
                start_time = curr_time
                last_time = curr_time
            else:
                if curr_time < last_time:
                    curr_time += datetime.timedelta(days=1)
                if (curr_time - start_time).total_seconds() > 3600:
                    break
                last_time = curr_time

            msg_len = approx_chars[idx]
            if char_budget + msg_len > target_chars and len(window_rows) > 0:
                break

            char_budget += msg_len
            window_rows.append(idx)
            last_id = row_dict["id"]

        if discard_sequence or len(window_rows) < args.min_messages:
            num_discarded_sequences += 1
            continue

        # Build messages; all messages from the last sender are marked as assistant
        messages = []
        last_sender = records[window_rows[-1]]["from"]
        for idx in window_rows:
            row_dict = records[idx]
            sender = row_dict["from"]
            text = str(row_dict["text"])
            reply_id = row_dict.get("reply_to_message_id")
            is_last_sender = sender == last_sender

            if is_last_sender:
                if text.strip() == "(image or file)":
                    discard_sequence = True
                    break

                content = text
                if pd.notna(reply_id):
                    try:
                        r_id = int(reply_id)
                        content = f"@{r_id} {text}"
                    except ValueError:
                        content = text

                # Filter low-quality assistant replies
                trimmed = content.strip()
                low_info_reply = trimmed.lower() in {"joo", "okei"}
                starts_with_command = trimmed.startswith("/")
                has_url = bool(re.search(r"https?://|www\.\S+", trimmed))
                if len(trimmed) <= 2 or has_url or low_info_reply or starts_with_command:
                    discard_sequence = True
                    break
                messages.append({"role": "assistant", "content": content})
            else:
                if sender in username_cache:
                    name = username_cache[sender]
                else:
                    name = bot.parse_username(sender)
                    username_cache[sender] = name
                user_payload = user_payloads[idx]
                messages.append({"role": "user", "name": name, "content": user_payload})

        if discard_sequence:
            num_discarded_sequences += 1
            continue

        messages.insert(0, system_prompt)
        jsonl.append({"messages": messages})

    random.shuffle(jsonl)
    n_validation = int(len(jsonl) * args.train_test_split_fraction)
    if n_validation == 0 and len(jsonl) > 1:
        n_validation = 1

    jsonl_train = jsonl[:-n_validation] if n_validation > 0 else jsonl
    jsonl_validation = jsonl[-n_validation:] if n_validation > 0 else []

    print(f"Generated {len(jsonl)} sequences")
    print(f"Discarded {num_discarded_sequences} sequences due to continuity checks")
    print(f"Training samples: {len(jsonl_train)}")
    print(f"Validation samples: {len(jsonl_validation)}")

    with open("openai_finetune_train_2.jsonl", "w") as f:
        for json_dict in jsonl_train:
            f.write(json.dumps(json_dict, ensure_ascii=False) + "\n")

    with open("openai_finetune_validation_2.jsonl", "w") as f:
        for json_dict in jsonl_validation:
            f.write(json.dumps(json_dict, ensure_ascii=False) + "\n")