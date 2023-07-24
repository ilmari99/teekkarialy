import random
from BotHead import BotHead
from MessageActions import LMGenerateOnTriggerPhrase, MakeJoke, GiveCommandInformation, OnFirstMessageInNewChat

with open("_token.txt", "r") as f:
    TOKEN = f.read().strip()

#MODEL_NAME = 'gpt3-xl-finetuned-v3-2-2Epoch'
MODEL_NAME = "gpt3-xl-finetuned-bigdata-v1"
N_MESSAGES = 16
CHAT_TYPES = ['group', 'supergroup', 'private', 'channel', 'bot' ]
bot = BotHead(MODEL_NAME, TOKEN, N_MESSAGES)

MESSAGE_ACTIONS = [
    OnFirstMessageInNewChat,
    GiveCommandInformation,
    MakeJoke,
    LMGenerateOnTriggerPhrase,
]
MESSAGE_ACTIONS = [action(bot) for action in MESSAGE_ACTIONS]

# One function, which handles all incoming messages. Also messaages from other bots
@bot.tg_bot.message_handler(func=lambda message: True, content_types=["text","sticker","photo","audio","video","document"])
def message_stack_handler(message):
    bot.store_item(message)
    for action in MESSAGE_ACTIONS:
        triggered = action(message)
        if triggered not in [True, False]:
            print(f"Action {action} returned {triggered}. Actions must return True or False.")
            break
        if triggered:
            print(f"Action '{action.__class__.__name__}' triggered.")
            break

bot.run()
