# Telegram Bot based on finnish GPT3 model

This repository contains a Telegram bot frame and a script for finetuning a GPT model on parsed Telegram chat data.
Your selected GPT model (and possibly finetuned) is then used by the Telegram bot to send messages to the chat. The bot is triggered by default if a message contains any of `['ai', 'gpt', 'tekoäl', '?', 'bot']`.
Upon triggering, the bot will try to predict the next sent message that it would send, and send it to the chat. For prompt, the bot receives a short description (pre-prompt), the last N(=10) messages in the chat, and finally an empty next message `GPT:`, after which the bot will start generating a continuation (what `GPT` says next).


the bot can be used in any chat type. If you want to finetune a model to a specific chat (recommended), export the Telegram chat history and run `read_history.py` (change the path), which parses the messages to a sequence of `sender;message` pairs.
Separate a train set by moving some of the last messages to another file. Move the created file to this repos root, and name them `_chat_history.csv` and `_chat_history_test.csv`.

Then you can run `finetune.py` to adjust the model to the chat. The finetuning requires a good computer or Google Colab. To launch, create a bot (by BotFather) and put your token to a file named `_token.txt`, and run `app.py`. Modify `app.py` to use your custom model, or use a pretrained model from Huggingface.

The models are quite heavy to run, but larger models are much better than smaller ones.

I've done this with the [Finnish GPT3 Large](https://turkunlp.org/gpt3-finnish), and finetuned it on 13000 messages in a 10 person groupchat, and its really fun!

Bot won't be published, since it has been trained on strictly private data.
