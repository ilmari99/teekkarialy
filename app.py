import random
from BotHead import BotHead
from MessageActions import TriggerPhrase

with open("_token.txt", "r") as f:
    TOKEN = f.read().strip()

MODEL_NAME = 'gpt3-xl-finetuned-v3'
N_MESSAGES = 10
CHAT_TYPES = ['group', 'supergroup', 'private', 'channel', 'bot' ]
bot = BotHead(MODEL_NAME, TOKEN, N_MESSAGES)

MESSAGE_ACTIONS = [TriggerPhrase]
MESSAGE_ACTIONS = [action(bot) for action in MESSAGE_ACTIONS]

# One function, which handles all incoming messages.
@bot.tg_bot.message_handler(func=lambda message: True)
def message_stack_handler(message):
    bot.store_item(message)
    for action in MESSAGE_ACTIONS:
        triggered = action(message)
        if triggered:
            break

bot.run()
