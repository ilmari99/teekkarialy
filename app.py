import random
from BotHead import BotHead
from OpenAIGPTHead import GPTBotHead
from MessageActions import LMGenerateOnTriggerPhrase, MakeJoke, GiveCommandInformation, OnFirstMessageInNewChat, RandomlyRespond

with open("_token.txt", "r") as f:
    TOKEN = f.read().strip()

#MODEL_NAME = 'gpt3-xl-finetuned-v3-2-2Epoch'
MODEL_NAME = "gpt3-xl-finetuned-bigdata-v1"
MAX_NUM_TOKENS = 2048
CHAT_TYPES = ['group', 'supergroup', 'private', 'channel', 'bot' ]
ALLOWED_CHAT_IDS = [-1001630430176, 1455609782, 2071428449, -1001856493108]
#bot = BotHead(MODEL_NAME, TOKEN, N_MESSAGES)
bot = GPTBotHead("ft:gpt-3.5-turbo-0613:personal::8mnRuwgR", TOKEN, MAX_NUM_TOKENS, "Teekkariäly")

MESSAGE_ACTIONS = [
    OnFirstMessageInNewChat,
    GiveCommandInformation,
    MakeJoke,
    RandomlyRespond,
    LMGenerateOnTriggerPhrase,
]
MESSAGE_ACTIONS = [action(bot) for action in MESSAGE_ACTIONS]

# One function, which handles all incoming messages. Also messaages from other bots
@bot.tg_bot.message_handler(func=lambda message: True, content_types=["text","sticker","photo","audio","video","document"])
def message_stack_handler(message):
    bot.store_item(message)
    # Check if the message is from an allowed chat
    if message.chat.id not in ALLOWED_CHAT_IDS:
        return
    if message.date < bot.start_time:
        return
    for action in MESSAGE_ACTIONS:
        triggered = action(message)
        if triggered not in [True, False]:
            print(f"Action {action} returned {triggered}. Actions must return True or False.")
            break
        if triggered:
            print(f"Action '{action.__class__.__name__}' triggered.")
            break

bot.run()
