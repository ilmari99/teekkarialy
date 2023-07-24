import time
import datetime
from telebot.types import Message

TIME_FORMAT = "%H:%M"
BEGIN_MSG_DELIMITER = "[MSG]"
END_MSG_DELIMITER = "[MEG]"
SEPARATOR = "[FS]"


def parse_username(username):
    """ Parse a username from a Telegram username.
    """
    #spec_chars = [".", "!", "?", " ", "_", "-"]
    # Remove special characters
    #username = "".join([c for c in username if c not in spec_chars])
    if not username:
        return "Anonym"
    return username

def get_curr_time():
    """ Return the current time in the format HH:MM.
    """
    return time.strftime(TIME_FORMAT, time.localtime())

def unix_time_to_ISO(unix_time : int):
    """ Convert a unix timestamp to ISO format.
    """
    return datetime.datetime.fromtimestamp(unix_time).isoformat()

def data_datestamp_to_time(data : str):
    """ Convert a datestamp in the format 2021-11-01T00:08:32 (YYYY-MM-DDTHH:MM:SS) to TIME_FORMAT
    """
    return datetime.datetime.strptime(data, "%Y-%m-%dT%H:%M:%S").strftime(TIME_FORMAT)

def parse_message_text(message):
    """ Parse a message (list[dict | str] or string) and return a string.
    If the message is a list, join strings with spaces, and dictionaries on their 'text' attribute.
    """
    if isinstance(message, str):
        return message
    elif isinstance(message, list):
        return "".join([parse_message_text(m) for m in message])
    elif isinstance(message, dict):
        return parse_message_text(message['text'])
    else:
        raise TypeError(f"Message must be a string or a list of strings, not {type(message)}")

if __name__ == "__main__":
    timestamp = "2021-11-01T00:08:32"
    new_time = data_datestamp_to_time(timestamp)
    print(new_time)
    cur_ = get_curr_time()
    print(cur_)