""" 
This file contains multiple message handlers. The handlers are placed in a list, and when a message is received
the handlers are looped through until a handler is found that handles the message.
"""

import random
from typing import Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from telebot.types import Message
from utils import get_curr_time, unix_time_to_ISO, parse_username
if TYPE_CHECKING:
    from BotHead import BotHead


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
    
    @abstractmethod
    def handle(self, msg : Message) -> Any:
        """ Takes in a message, and returns True if the message is handled, and False otherwise."""
        pass
    
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
        message_part1 = "Kiitos ryhmään pääsystä! Olen Teekkariuteen pohjautuva tekoäly. Yritän jatkaa keskustelua samalla tyylillä, lisäten siihen huumoria, hehe. Tässä teille yksi vitsi:"
        prompt = self.get_additional_message_prompt(msg) + message_part1
        # Generate a joke
        message_part2 = self.tg_bot.lang_model.get_only_until_token(prompt, temperature=0.7, max_new_tokens=70, token="[FS]").replace("[FS]", "")
        # send
        self.tg_bot.send_message_wrapper(msg.chat.id, message_part1 + message_part2)
        return True
    
class GiveGenericInformation(BaseHandler):
    """ Give information about the bot.
    This is triggered by the command "/gpt".
    """
    command = "/gpt-info"
    def handle(self, msg : Message) -> Any:
        """ If the message is "/gpt", give information about the bot.
        """
        msg_text = msg.text.lower()
        input_msg_id = msg.message_id
        if msg_text != self.command:
            return False
        bot_info ="""
        Moi!
        Olen TurkuNLP:n kouluttamaan suomalaiseen GPT malliin pohjautuva tekoäly, johon on lisätty huumoria LUT:sta.
        Olen koulutettu keskustelun historialla, ja yritän jatkaa keskustelua samalla tyylillä, mutta lisäten siihen huumoria.
        """
        self.tg_bot.send_message_wrapper(msg.chat.id, bot_info, reply_to_message_id=input_msg_id)
        return True
    
class GiveCommandInformation(BaseHandler):
    """ Give information about the supported commands.
    """
    command = "/gpt-help"
    def handle(self, msg : Message) -> Any:
        """ If the message is "/gpt", give information about the bot.
        """
        msg_text = msg.text.lower()
        input_msg_id = msg.message_id
        if msg_text != self.command:
            return False
        bot_info ="""
        Moi!
        Olen TurkuNLP:n kouluttamaan suomalaiseen GPT malliin pohjautuva tekoäly, johon on lisätty huumoria LUT:sta.
        Olen koulutettu keskustelun historialla, ja yritän jatkaa keskustelua samalla tyylillä, mutta lisäten siihen huumoria.
        
        Komennot:
        /gpt-help - Tietoa komennoista
        /vitsi [mistä] - Lähetä vitsi. 'mistä' voi specifioida mistä aiheesta vitsi halutaan. Esim. '/vitsi Teemu'
        """
        
        self.tg_bot.send_message_wrapper(msg.chat.id, bot_info, reply_to_message_id=input_msg_id)   
        return True
    
class MakeJoke(BaseHandler):
    """ Generates a joke and sends it.
    Triggered by the command '/vitsi'
    """
    command = "/vitsi"
    
    def get_random_joke_prompt(self):
        possible_prompts = ["Keksin eile hauskan vitsin",
                            "Tässä teille hauska Teekkari vitsi",
                            "Ai että, tuli hauska vitsi mieleen",
                            "Siis kuulin tän vitsin eilen",
                            "Kotiseutukursioilla kuulin tän vitsin",
                            ]
        return random.choice(possible_prompts)
    
    def handle(self, msg : Message) -> Any:
        """ If the message is "/vitsi [topic]", generate a joke and send it.
        """
        msg_text = msg.text.lower()
        input_msg_id = msg.message_id
        if not msg_text.startswith(self.command):
            return False
        
        # Parse the joke topic
        joke_topic = msg_text.replace(self.command, "").strip()
        if joke_topic == "":
            joke_topic = None
        
        # Select a random joke prompt
        message_part1 = self.get_random_joke_prompt()
        # If a joke topic is specified, add it to the prompt
        if joke_topic:
            message_part1 += f" aiheesta '{joke_topic}'"
        message_part1 += ":"
        # Add a message beginning '[MSG]<id>[FS]<time>[FS]<sender>[FS]' to the prompt
        prompt = self.get_additional_message_prompt(msg) + message_part1
        
        message_part2 = self.tg_bot.lang_model.get_only_until_token(prompt, temperature=0.7, max_new_tokens=70, token="[FS]").replace("[FS]", "")
        
        self.tg_bot.send_message_wrapper(msg.chat.id, message_part2, reply_to_message_id=input_msg_id)
        return True

class LMGenerateOnTriggerPhrase(BaseHandler):
    """ This is called if the message contains a trigger phrase.
    If a trigger is found, the bot sends a reply.
    """
    trigger_phrases = ["bot","gpt","?", "ai", "neuroverk", "tekoäly"]
    def handle(self, msg: Message) -> Any:
        """ If the message contains a trigger phrase, send a reply.
        """
        msg_text = msg.text.lower()
        if not any(trigger in msg_text for trigger in self.trigger_phrases):
            return False
        replies = self.tg_bot.create_replies(msg.chat.id)
        for reply, reply_to_id in replies:
            self.tg_bot.send_message_wrapper(msg.chat.id, reply, reply_to_message_id=reply_to_id)
        return True
    
    
    