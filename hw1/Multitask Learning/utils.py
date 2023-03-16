import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from nlp import load_dataset
from datasets import DatasetDict
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO)
from transformers import AutoModel
import transformers

sentence1_key, sentence2_key = ("premise", "hypothesis")
from sklearn.preprocessing import LabelEncoder
def preprocess_function(examples):
    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    # the tokenization args here are flexible, you can change them if you want (refer to the examples and documentation)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    result = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
    
    labels = []
    for i, label in enumerate(examples["label"]):
        if label == 0:
        #   labels.append(torch.tensor([1, 0, 0]))
          labels.append([1, 0, 0])
          
        elif label == 1:
        #   labels.append(torch.tensor([0, 1, 0]))
          labels.append([0, 1, 0])
          
        elif label == 2:
        #   labels.append(torch.tensor([0, 0, 1]))
          labels.append([0, 0, 1])
          

    result["label"] = labels
    return result


def tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", max_length=128, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_all_tokens = False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["label"] = labels

    return tokenized_inputs


class TaskSampler():
    def __init__(self, 
                *, 
                dataloader_dict: dict[str, DataLoader],
                task_weights=None,
                max_iters=None):
        
        assert dataloader_dict is not None, "Dataloader dictionary must be provided."

        self.dataloader_dict = dataloader_dict
        self.task_names = list(dataloader_dict.keys())
        self.dataloader_iterators = self._initialize_iterators()
        self.task_weights = task_weights if task_weights is not None else self._get_uniform_weights()
        self.max_iters = max_iters if max_iters is not None else float("inf")
        self.p = [0, 1] #[0.5, 0.5] # [0, 1] #init p ## {mnli,ner}
        
    # Initialization methods
    def _get_uniform_weights(self):
        # return {"mnli": 0, "ner": 0}
        return [1/len(self.task_names) for i in self.task_names]
    
    def _initialize_iterators(self):
        return {name:iter(dataloader) for name, dataloader in self.dataloader_dict.items()}
    
    # Weight getter and setter methods (NOTE can use these to dynamically set weights)
    def set_task_weights(self, task_weights):
        assert sum(self.task_weights) == 1, "Task weights must sum to 1."
        self.task_weights = task_weights
    
    def get_task_weights(self):
        return self.task_weights

    # Sampling logic
    def _sample_task(self, p):
        ## {mnli,ner}
        return np.random.choice(self.task_names, p=self.p)
    
    def get_sample_task(self):
        return self.p
    
    def set_sample_task(self, p):
        assert sum(self.p) == 1, "Task weights must sum to 1."
        self.p = p
        # self._sample_task
    
    def _sample_batch(self, task):
        try:
            return self.dataloader_iterators[task].__next__()
        except StopIteration:
            print(f"Restarting iterator for {task}")
            self.dataloader_iterators[task] = iter(self.dataloader_dict[task])
            return self.dataloader_iterators[task].__next__()
        except KeyError as e:
            print(e)
            raise KeyError("Task not in dataset dictionary.")
    
    # Iterable interface
    def __iter__(self):
        self.current_iter = 0
        return self
    
    def __next__(self):
        if self.current_iter >= self.max_iters:
            raise StopIteration
        else:
            self.current_iter += 1
        p = self.get_sample_task()
        task = self._sample_task(p)
        batch = self._sample_batch(task)
        return task, batch
    
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # labels_vocab = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        # labels_vocab_reverse = {v:k for k,v in labels_vocab.items()}
        self.base_model = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.clf_ner = nn.Linear(768, 128) 
        self.clf_mnli = nn.Linear(768, 3) 
        
    def forward(self, task, batch): #input_ids, attn_mask, task):
        # You write you new head here
        outputs = self.base_model(batch["input_ids"], attention_mask=batch['attention_mask'])
        outputs = self.dropout(outputs[0])
        
        if task == 'mnli':
            return self.clf_mnli(outputs)
        else:
            return self.clf_ner(outputs)


