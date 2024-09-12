#!/usr/bin/env python
# coding: utf-8

# # Lightweight Fine-Tuning Project

# TODO: In this cell, describe your choices for each of the following
# 
# * PEFT technique: LoRA
# * Model: BERT
# * Evaluation approach: evaluate method
# * Fine-tuning dataset: cornell-movie-review-data/rotten_tomatoes

# ## Loading and Evaluating a Foundation Model
# 
# TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.

# In[1]:


from datasets import load_dataset
#Loading dataset and splitting it into train and test
splits = ["train", "test"]
dataset = {split: dataset for split, dataset in zip(splits, load_dataset("cornell-movie-review-data/rotten_tomatoes", split=splits))}

for split in splits:
    dataset[split] = dataset[split].shuffle(seed=42).select(range(50))

dataset


# In[2]:


from transformers import AutoTokenizer
#Loading tokenizer and applying it to both test and train datasets
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = {}
for split in splits:
    tokenized_datasets[split] = dataset[split].map(tokenize_function, batched=True)


# In[3]:


from transformers import AutoModelForSequenceClassification
#Loading the pretrained distilbert model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"}, #Labels represent positive and negative movie reviews
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)


# In[4]:


import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
#metrics used to calculate the accuracy of the models
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

#training arguments for the pretrained model
training_args = TrainingArguments(output_dir="test_trainer", learning_rate=2e-4, evaluation_strategy="epoch", num_train_epochs=1, per_device_train_batch_size=2,)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer = tokenizer),
    compute_metrics=compute_metrics,
)


# In[5]:


trainer.evaluate()


# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[6]:


from peft import LoraConfig, get_peft_model

#loading lora config with task_type, inference_mode, r, lora_alpha, lora_dropout, and target_modules parameters for 
#this chosen dataset
config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules = ['q_lin', 'v_lin'])


# In[7]:


#loading the peft model
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()


# In[8]:


import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
#training the lora model 
lora_training_args = TrainingArguments(output_dir="lora_trainer",
                                        evaluation_strategy='steps',
                                        learning_rate=2e-4,
                                        num_train_epochs=1,
                                        per_device_train_batch_size=2, )

lora_trainer = Trainer(
    model = lora_model,
    args = lora_training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["test"],
    tokenizer = tokenizer,
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer),
    compute_metrics = compute_metrics,
)

lora_trainer.train()


# In[9]:


lora_model.print_trainable_parameters()


# In[10]:


lora_model.save_pretrained("distilbert-base-peft")


# ## Performing Inference with a PEFT Model
# 
# TODO: In the cells below, load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.

# In[11]:


from peft import AutoPeftModelForSequenceClassification
#loading the peft model
peft_model = AutoPeftModelForSequenceClassification.from_pretrained("distilbert-base-peft")


# In[12]:


peft_training_args = TrainingArguments(output_dir="peft_trainer",
                                        evaluation_strategy='steps',
                                        learning_rate=2e-4,
                                        num_train_epochs=10,
                                        per_device_train_batch_size=2, )

#training the peft model for evaluation and comparison
peft_trainer = Trainer(
    model = peft_model,
    args = peft_training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["test"],
    tokenizer = tokenizer,
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer),
    compute_metrics = compute_metrics,
)

peft_trainer.train()


# In[13]:


#peft model shows a substantial increase in accuracy compared to the pretrained model
peft_trainer.evaluate()


# In[14]:


trainer.evaluate()


# In[ ]:




