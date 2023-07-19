""" 
This file contains multiple message handlers. The handlers are placed in a list, and when a message is received
the handlers are looped through until a handler is found that handles the message.
"""

from typing import Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from telebot.types import Message
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
    
    @abstractmethod
    def handle(self, msg : Message) -> Any:
        """ Takes in a message, and returns True if the message is handled, and False otherwise."""
        pass


class TriggerPhrase(BaseHandler):
    """ This is called if the message contains a trigger phrase.
    If a trigger is found, the bot sends a reply.
    """
    trigger_phrases = ["bot","gpt","?", "ai", "neuroverk", "tekoäly"]
    def handle(self, msg: Message) -> Any:
        """ If the message contains a trigger phrase, send a reply.
        """
        msg_text = msg.text.lower()
        if any(trigger in msg_text for trigger in self.trigger_phrases):
            self.tg_bot.create_reply(msg.chat.id)
            return True
        return False
    
    
    