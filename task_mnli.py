import numpy as np
import torch
import torch.nn as nn
import transformers
from nlp import load_dataset
from torch.utils.data import DataLoader
import logging
logging.basicConfig(level=logging.INFO)
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, AutoTokenizer
from transformers import AutoModel
from utils import *


import os
# os.environ['HF_HOME'] = "/fs/classhomes/spring2023/cmsc828a/c828a038/.cache/huggingface"
    
mnli_all_ds = load_dataset("multi_nli")
# mnli_all_ds = load_dataset('glue', 'mnli', split='train') # load_dataset("multi_nli", split="train")
from datasets import  DatasetDict,  load_dataset

ner_all_ds = load_dataset("Babelscape/wikineural")
dataset_dict = DatasetDict({"mnli": mnli_all_ds["train"], "ner": ner_all_ds["train_en"]})
tpr=50
prev_loss={"mnli_1": 10, "ner_1": 10, "mnli_2": 10, "ner_2": 10}
tasks = list(dataset_dict.keys())
# identifier on huggingface.co/models
model_name = "bert-base-cased"
max_iters = 2
batch_size = 32
PATH = 'models/model_ner_w1'
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#ner
dataset_dict["ner"] = dataset_dict["ner"].map(tokenize_and_align_labels,  remove_columns=dataset_dict["ner"].column_names, batched=True)

#mnli
dataset_dict["mnli"] = dataset_dict["mnli"].map(preprocess_function, remove_columns=dataset_dict["mnli"].column_names, batched=True)

dataset_dict["ner"].set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
dataset_dict["mnli"].set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer)
dataloader_dict = {key: DataLoader(dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size) for key, dataset in dataset_dict.items()}
model = CustomModel().to('cuda')
# task_sampler = TaskSampler(dataloader_dict=dataloader_dict, max_iters=max_iters, task_weights=[0.5, 0.5])
task_sampler = TaskSampler(dataloader_dict=dataloader_dict, task_weights=[1, 0])


optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,eps=1e-6)
criterion = nn.CrossEntropyLoss()
epoch_loss = []
for idx, (task, batch)in enumerate(task_sampler):
    batch = batch.to('cuda')
    optimizer.zero_grad() # clear gradients first
    torch.cuda.empty_cache() # releases all unoccupied cached memory 
    predictions = model(task, batch)
    label = batch["labels"]
    # DWA
    loss = criterion(predictions, label) #model loss-task specific
    print(task,loss)
    # if task == 'mnli':
    #     l_prev, l_curr = prev_loss['mnli_1'], loss.item()
    #     w=l_curr/l_prev
    #     w_other = prev_loss["ner_1"]/prev_loss["ner_2"]
    #     lmbd_task = 2*np.exp(w/tpr)/(np.exp(w/tpr)+np.exp(w_other/tpr)) # cal weight
    #     lmbd_task = min(lmbd_task, 1)
    #     task_sampler.set_task_weights([lmbd_task, 1-lmbd_task]) #update task_weights
    #     loss = lmbd_task*loss + (1-lmbd_task)*prev_loss["ner_1"]  #cal tasklearning loss
    #     prev_loss["mnli_2"] = prev_loss["mnli_1"] 
    #     prev_loss["mnli_1"] = l_curr #prev loss update

    # else:
    #     l_prev, l_curr = prev_loss['ner_1'], loss.item()
    #     w=l_curr/l_prev
    #     w_other = prev_loss["mnli_1"]/prev_loss["mnli_2"]
    #     lmbd_task = 2*np.exp(w/tpr)/(np.exp(w/tpr)+np.exp(w_other/tpr)) # cal weight
    #     lmbd_task = min(lmbd_task, 1)
    #     task_sampler.set_task_weights([1-lmbd_task, lmbd_task]) #update task_weights
    #     loss = lmbd_task*loss + (1-lmbd_task)*prev_loss["mnli_1"]  #cal tasklearning loss
    #     prev_loss["ner_2"] = prev_loss["ner_1"] 
    #     prev_loss["ner_1"] = l_curr #prev loss update
    loss.backward()
    #function check
    # print(task, task_sampler.get_task_weights(), prev_loss)
    optimizer.step()
    epoch_loss.append(loss.item())

    if idx % 500 == 0:
        torch.save(model.state_dict(), "{}.pt".format(PATH))
        with open("{}.txt".format(PATH), "w") as output:
            for item in epoch_loss:
                # write each item on a new line
                output.write("%s\n" % item)
            print('Done')
    if idx == 14000:
        break
# model_scripted.save('model_test.pt')
# torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, '/fs/classhomes/spring2023/cmsc828a/c828a038/')