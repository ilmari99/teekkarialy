import os
import random
import time
from BotHead import BotHead
from OpenAIGPTHead import GPTBotHead
from MessageActions import LMGenerateOnTriggerPhrase, MakeJoke, GiveCommandInformation, OnFirstMessageInNewChat, RandomlyRespond, ReactWhenRespondedTo, MessageWhenChatSilent
import multiprocessing
from telebot.types import Message
multiprocessing.set_start_method('fork', True)

with open("_token.txt", "r") as f:
    TOKEN = f.read().strip()

#MODEL_NAME = 'gpt3-xl-finetuned-v3-2-2Epoch'
MODEL_NAME = "gpt3-xl-finetuned-bigdata-v1"
MAX_NUM_TOKENS = 4096
CHAT_TYPES = ['group', 'supergroup', 'private', 'channel', 'bot' ]
REENGAGE_MEAN_TIME = 60*60*24 # 24 hours
REFRESH_TIME_INTERVAL = 30*60 # 30 minutes
ALLOWED_CHAT_IDS = [-1001630430176, 1455609782, 2071428449, -1001856493108]
#bot = BotHead(MODEL_NAME, TOKEN, N_MESSAGES)
#bot = GPTBotHead("ft:gpt-3.5-turbo-0613:personal::8mnRuwgR", TOKEN, MAX_NUM_TOKENS, "Teekkariäly")
bot = GPTBotHead("ft:gpt-4.1-mini-2025-04-14:personal:algebros-finetune:CosugOff", TOKEN, MAX_NUM_TOKENS, "Teekkariäly")

MESSAGE_ACTIONS = [
    OnFirstMessageInNewChat,
    GiveCommandInformation,
    MakeJoke,
    RandomlyRespond,
    LMGenerateOnTriggerPhrase,
    ReactWhenRespondedTo,
]
MESSAGE_ACTIONS = [action(bot) for action in MESSAGE_ACTIONS]
reengage_chat_action = MessageWhenChatSilent(bot)

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
        
def randomly_send_message(parent_pid: int):
    """Periodically re-engage silent chats with a random message.

    Uses the chat history file's modification time to measure silence so it
    survives restarts. Runs in a subprocess.
    """
    monitored_chat_ids = [-1001630430176]#, 1455609782]
    while True:
        # Exit if parent died (prevents orphaned worker when daemon=False)
        if os.getppid() != parent_pid:
            print("Reengage worker exiting because parent died", flush=True)
            break
        for chat_id in monitored_chat_ids:
            bot._init_last_messages(chat_id)
            messages = bot.last_messages.get(chat_id, [])
            print(f"Found {len(messages)} messages in chat {chat_id}", flush=True)
            if len(messages) == 0:
                continue

            chat_history_path = f"ChatDatas/{chat_id}.csv"
            try:
                last_activity_ts = os.path.getmtime(chat_history_path)
            except FileNotFoundError:
                # If the file vanished, skip this round
                continue

            last_message_row = messages.iloc[-1]
            # Construct a minimal Message for the handler
            last_message = Message(
                message_id=last_message_row["id"],
                from_user=last_message_row["from"],
                chat=chat_id,
                date=last_activity_ts,
                content_type="text",
                options={},
                json_string=last_message_row["text"],
            )

            time_since_last_message = time.time() - last_activity_ts
            print(f"Time since last message: {time_since_last_message}", flush=True)

            # Expected trigger time is REENGAGE_MEAN_TIME with ±50% jitter
            reengage_mean_time_with_jitter = REENGAGE_MEAN_TIME + random.uniform(-REENGAGE_MEAN_TIME / 2, REENGAGE_MEAN_TIME / 2)
            if time_since_last_message < reengage_mean_time_with_jitter:
                continue

            triggered = reengage_chat_action(last_message)
            if triggered:
                print(f"Action '{reengage_chat_action.__class__.__name__}' triggered.", flush=True)

        time.sleep(REFRESH_TIME_INTERVAL)

# Start the subprocess with parent PID tracking (daemon=False so it can spawn children)
p = multiprocessing.Process(target=randomly_send_message, args=(os.getpid(),), daemon=False)
p.start()

bot.run()