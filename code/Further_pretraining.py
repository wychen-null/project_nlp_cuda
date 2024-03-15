import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForPreTraining
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    RandomSampler, 
    SequentialSampler
)
import math 
from transformers.optimization import (
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    roc_curve,
    auc,
    average_precision_score,
)
from scipy.special import softmax
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
torch.cuda.empty_cache()

model_name = "roberta-large"

num_labels = 2
device = torch.device("cuda")
device_ids=[0, 1]

tokenizer_name = model_name

batch_size = 16
warmup_ratio = 0.06
weight_decay=0.0
gradient_accumulation_steps = 1
num_train_epochs =100
learning_rate = 1e-05
adam_epsilon = 1e-08

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForPreTraining.from_pretrained(model_name)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.5
)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='../data/tweets/train.txt',
    block_size=512,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='../data/tweets/test.txt',
    block_size=512,
)

#t_total = (len(train_dataset)//batch_size+1)//gradient_accumulation_steps*num_train_epochs
t_total=19890 
print(t_total)
warmup_steps = math.ceil(t_total * warmup_ratio)
optimizer = AdamW(model.parameters(),lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)



training_args = TrainingArguments(
    output_dir="./roberta-retrained-0.5",
    evaluation_strategy = "epoch",
#    save_strategy  = "epoch",
#    warmup_ratio = warmup_ratio,
    learning_rate = learning_rate,
    overwrite_output_dir=True,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #eval_steps=400,
    #save_steps=400,
    save_total_limit=3,
    seed=1,
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    
)

model= nn.DataParallel(model)
#model.to(device)
trainer.train()

model.module.save_pretrained("./roberta_pretrained_fin_0.5_e1")