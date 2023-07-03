import random
from FinGPTelebot import FinGPTelebot

with open("_token.txt", "r") as f:
    TOKEN = f.read().strip()

MODEL_NAME = 'gpt3-xl-finetuned-2'
N_MESSAGES = 10
CHAT_TYPES = ['group', 'supergroup', 'private', 'channel', 'bot' ]
bot = FinGPTelebot(MODEL_NAME, TOKEN, N_MESSAGES, simulate_users = False)
pre_messages = [
    ("Eki3ki", "Hei gpt kerro mielipide gaussisista prosesseista"),
    ("GPT", "Ihan toimiva idea, mut en tiiä toimiiko käytännössä. Joku vois tuua mulleki nii pääsisin testaa 😅"),
    ("pauliant", "y(t) = sin(t) + e(t), jossa e(t) ~ N(0,1). siin on sulle gaussinen prosessi gpt, eiku testaamaan"),
    ("GPT", "Siis miten toi toimii? En oikee hiffaa tota :D"),
    ("pauliant", "Mood")
]
bot.pre_fill_messages(pre_messages)
@bot.bot.message_handler(commands=['start', 'help'], chat_types=CHAT_TYPES)
def give_info(message):
    bot.give_info(message.chat.id)

@bot.bot.message_handler(content_types=['sticker'], chat_types=CHAT_TYPES)
def sticker_input(message):
    bot.store_sticker(message)

@bot.bot.message_handler(content_types=['document'], chat_types=CHAT_TYPES)
def file_input(message):
    bot.store_file(message)


@bot.bot.message_handler(func=lambda message: True, content_types=['text'], chat_types=CHAT_TYPES)
def possibly_reply(message):
    bot.store_message(message)
    msg = message.text.lower()
    triggers = ["bot","gpt","?", "ai", "neuroverk", "tekoäly"]
    if any(trigger in msg for trigger in triggers):
        bot.send_reply(message.chat.id)
bot.run()
