# Telegram Bot based on finnish GPT3 model

This repository contains a Telegram bot frame, that can use a GPT model to chat 1-to-1 or in a groupchat based on the previous messages. If you want to finetune a model to a specific chat, export the chat history, and run `read_history.py`, which parses the messages. Then you can run `finetune.py` to adjust the model to the groupchat. The finetuning requires a good computer or atleast Google Colab. To launch, create a bot (by BotFather), and run `app.py`.

I've done this with the [Finnish GPT3 Large](https://turkunlp.org/gpt3-finnish), and finetuned it on 10000 messages in a 10 person groupchat, and its really fun!

Bot won't be published, since it has been trained on strictly private data.
