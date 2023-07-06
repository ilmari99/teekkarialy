





def parse_username(username):
    """ Parse a username from a Telegram username.
    """
    spec_chars = [".", "!", "?", " ", "_", "-"]
    # Remove special characters
    username = "".join([c for c in username if c not in spec_chars])
    return username