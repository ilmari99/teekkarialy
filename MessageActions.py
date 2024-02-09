""" 
This file contains multiple message handlers. The handlers are placed in a list, and when a message is received
the handlers are looped through until a handler is found that handles the message.
"""

import random
from typing import Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from telebot.types import Message
from BotHead import BotHead
from utils import get_curr_time, unix_time_to_ISO, parse_username
if TYPE_CHECKING:
    from BotHead import BotHead
    
    
JOKE_BEGIN_PROMPTS = [
    "Keksin eile hauskan vitsin",
    "Tässä teille hauska Teekkari vitsi",
    "Ai että, tuli hauska vitsi mieleen",
    "Siis kuulin tän vitsin eilen",
    "Kotiseutukursioilla kuulin tän vitsin",
    ]
    
JOKE_BEGINS = [
    "Mikä on",
    "Mitä saa kun",
    "Kuinka monta",
    "Miksi",
    "Olipa kerran",
    "",
]

        
class BaseHandler(ABC):
    """ This is the base class for all handlers.
    This class is not meant to be used directly, and this is only used to define the interface for handlers.
    """
    def __init__(self, tg_bot : 'BotHead'):
        self.tg_bot = tg_bot
    
    def __call__(self, msg : Message) -> Any:
        """ This method is called when the handler is called as a function."""
        return self.handle(msg)
    
    def get_additional_message_prompt(self, msg : Message) -> str:
        """ Returns a string, that is used as pre-prompt for some message generation tasks.
        Use this like:
        self.get_additional_message_prompt(msg) + "Tässä hauska vitsi:"
        """
        return f"[MSG]{msg.message_id}[FS]{get_curr_time()}[FS]{self.tg_bot.tg_name}[FS]"
    
    def check_message_is_text(self, msg : Message) -> bool:
        """ Checks if the message is a text message.
        """
        return msg.content_type == "text"
    
    @abstractmethod
    def handle(self, msg : Message) -> Any:
        """ Takes in a message, and returns True if the message is handled, and False otherwise."""
        pass
    
class ReactWhenRespondedTo(BaseHandler):
    """ Reacts to message if someone replies to a message the bot sent.
    """
    def handle(self, msg : Message) -> Any:
        """ If the message is a reply to a message the bot sent, react to it.
        """
        if not self.check_message_is_text(msg):
            return False
        if msg.reply_to_message is None:
            return False
        if msg.reply_to_message.from_user.id != self.tg_bot.tg_bot.id:
            return False
        chat_id = msg.chat.id
        reply = self.tg_bot.create_replies(chat_id)
        num_tried = 0
        while not reply and num_tried < 3:
            reply = self.tg_bot.create_replies(chat_id)
            num_tried += 1
        reply_text = reply[0][0]
        self.tg_bot.send_message_wrapper(msg.chat.id, reply_text)
        return True

class RandomlyRespond(BaseHandler):
    """ Randomly respond to messages.
    """
    def __init__(self, tg_bot : 'BotHead', trigger_probability=0.1):
        super().__init__(tg_bot)
        self.trigger_probability = 0.1
        if hasattr(self.tg_bot, "trigger_probability"):
            self.trigger_probability = self.tg_bot.trigger_probability
        
    def handle(self, msg : Message) -> Any:
        """ If the message is from a new chat, introduce the bot.
        """
        chat_id = msg.chat.id
        if len(self.tg_bot.last_messages[chat_id]) == 0:
            return False
        if random.random() > self.trigger_probability:
            return False
        reply = self.tg_bot.create_replies(chat_id)
        num_tried = 0
        while not reply and num_tried < 4:
            reply = self.tg_bot.create_replies(chat_id)
            num_tried += 1
        reply_text = reply[0][0]
        self.tg_bot.send_message_wrapper(msg.chat.id, reply_text)
        return True
    
    
class OnFirstMessageInNewChat(BaseHandler):
    """ If a message is received from a new chat, introduce the bot.
    """
    def handle(self, msg : Message) -> Any:
        """ If the message is from a new chat, introduce the bot.
        """
        chat_id = msg.chat.id
        # Trigger this, if the received message is the first message in the chat
        if len(self.tg_bot.last_messages[chat_id]) > 1:
            return False

        # Introduce the bot in a polite but fun way
        message_part1 = ("Kiitos ryhmään pääsystä! Olen Teekkariuteen pohjautuva tekoäly. "
                         "Yritän jatkaa keskustelua samalla tyylillä, lisäten siihen huumoria, hehe."
        )
        # Check if self.tg_bot has method 'get_only_until_token'
        if not hasattr(self.tg_bot, "get_only_until_token"):
            self.tg_bot.send_message_wrapper(msg.chat.id, message_part1)
            return True
        
        message_part1 += "Tässä teille yksi vitsi: "
        joke_begin = self.get_random_joke_begin()
        prompt = self.get_additional_message_prompt(msg) + message_part1 + joke_begin
        # Generate a joke
        message_part2 = self.tg_bot.lang_model.get_only_until_token(prompt, temperature=0.5, max_new_tokens=70, token="[FS]").replace("[FS]", "")
        # send
        self.tg_bot.send_message_wrapper(msg.chat.id, message_part1 + message_part2 + joke_begin)
        return True
    
    def get_random_joke_begin(self):
        """ Returns a random joke beginning.
        """
        return random.choice(JOKE_BEGINS)
    
class GiveCommandInformation(BaseHandler):
    """ Give information about the supported commands.
    """
    def __init__(self, tg_bot: BotHead):
        super().__init__(tg_bot)
        self.bot_info = """
        Moi!
        Olen TurkuNLP:n kouluttamaan suomalaiseen GPT malliin pohjautuva tekoäly, johon on lisätty huumoria LUT:sta.
        Olen koulutettu keskustelun historialla, ja yritän jatkaa keskustelua samalla tyylillä, mutta lisäten siihen huumoria.
        
        Komennot:
        /gpt-help - Tietoa komennoista
        /vitsi [mistä] - Lähetä vitsi. 'mistä' voi specifioida mistä aiheesta vitsi halutaan. Esim. '/vitsi Teemu'
        """
        if hasattr(self.tg_bot, "bot_info"):
            self.bot_info = self.tg_bot.bot_info
    
    command = "/gpt-help"
    def handle(self, msg : Message) -> Any:
        """ If the message is "/gpt", give information about the bot.
        """
        input_msg_id = msg.message_id
        if msg.content_type != "text" or not msg.text.startswith(self.command):
            return False
        
        self.tg_bot.send_message_wrapper(msg.chat.id, self.bot_info, reply_to_message_id=input_msg_id)   
        return True
    
class MakeJoke(BaseHandler):
    """ Generates a joke and sends it.
    Triggered by the command '/vitsi'
    """
    command = "/vitsi"
    
    def get_random_joke_prompt(self):
        return random.choice(JOKE_BEGIN_PROMPTS)
    
    def get_random_joke_begin(self):
        return random.choice(JOKE_BEGINS)
    
    def handle(self, msg : Message) -> Any:
        """ If the message is "/vitsi [topic]", generate a joke and send it.
        """
        
        if msg.content_type != "text" or not msg.text.startswith(self.command):
            return False
        
        if hasattr(self.tg_bot, "make_joke"):
            joke = self.tg_bot.make_joke(msg)
            self.tg_bot.send_message_wrapper(msg.chat.id, joke, reply_to_message_id=msg.message_id)
            return True
        
        msg_text = msg.text.lower()
        input_msg_id = msg.message_id
        
        # Parse the joke topic
        joke_topic = msg_text.replace(self.command, "").strip()
        if joke_topic == "":
            joke_topic = None
        
        # Select a random joke prompt
        message_part1 = self.get_random_joke_prompt()
        # If a joke topic is specified, add it to the prompt
        if joke_topic:
            message_part1 += f" aiheesta '{joke_topic}'"
        message_part1 += ": "
        joke_begin = self.get_random_joke_begin()
        # Add a message beginning '[MSG]<id>[FS]<time>[FS]<sender>[FS]' to the prompt
        prompt = self.get_additional_message_prompt(msg) + message_part1 + joke_begin
        
        print(f"Prompt --------------------------------------------\n{prompt}\n--------------------------------------------")
        
        message_part2 = self.tg_bot.lang_model.get_only_until_token(prompt, temperature=0.5, max_new_tokens=70, token="[FS]").replace("[FS]", "")
        
        self.tg_bot.send_message_wrapper(msg.chat.id, joke_begin + message_part2, reply_to_message_id=input_msg_id)
        return True

class LMGenerateOnTriggerPhrase(BaseHandler):
    """ This is called if the message contains a trigger phrase.
    If a trigger is found, the bot sends a reply.
    """
    def __init__(self, tg_bot: BotHead):
        super().__init__(tg_bot)
        self.trigger_phrases = ["bot","gpt","?", "ai", "neuroverk", "tekoäly"]
        if hasattr(self.tg_bot, "trigger_phrases"):
            self.trigger_phrases = self.tg_bot.trigger_phrases

    def handle(self, msg: Message) -> Any:
        """ If the message contains a trigger phrase, send a reply.
        """
        if msg.content_type != "text":
            return False
        msg_text = msg.text.lower()
        if not any(trigger in msg_text for trigger in self.trigger_phrases):
            return False
        replies = self.tg_bot.create_replies(msg.chat.id)
        for reply, reply_to_id in replies:
            self.tg_bot.send_message_wrapper(msg.chat.id, reply, reply_to_message_id=reply_to_id)
        return True
    
    
    