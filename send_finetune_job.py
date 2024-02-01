from openai import OpenAI
import argparse

with open("__openai_api_key.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
openai = OpenAI(api_key=OPENAI_API_KEY)

def upload_files(paths):
    """ Upload files to OpenAI.
    """
    for path in paths:
        openai.files.create(file = open(path, "rb"), purpose = "fine-tune")
        print(f"Uploaded {path}")
    return


def send_finetune_job(model_id, train_file_id, validation_file_id, epochs = 2):
    """ Finetune a model with the given file ids.
    """
    # Create the fine-tuned model
    openai.fine_tuning.jobs.create(
        model=model_id,
        training_file= train_file_id,
        validation_file= validation_file_id,
        hyperparameters={
            "n_epochs": epochs,
        },
    )
    return


#upload_files(["/home/ilmari/python/TGBOT_FIN_GPT/openai_finetune_train.jsonl", "/home/ilmari/python/TGBOT_FIN_GPT/openai_finetune_validation.jsonl"])
#list_files = openai.files.list()
#print(list_files)
send_finetune_job(model_id="ft:gpt-3.5-turbo-0613:personal:lateksii:8mfr6nap",
                  train_file_id="file-vadI5L78cGM5IpB7kYis5xR4",
                  validation_file_id="file-EfDVpJyeT1jbvtrn6rrt3LNn",
                  epochs=2)


