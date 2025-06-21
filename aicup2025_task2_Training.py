
import os
import sys
sys.path.append('/content/drive/MyDrive/aicup')
import json
import torch
import random
import zipfile
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, Audio, load_dataset, Features, Value
from aicup import (DataCollatorSpeechSeq2SeqWithPadding,
      transcribe_with_timestamps,
      collate_batch_with_prompt_template,
      generate_annotated_audio_transcribe_parallel,OpenDeidBatchSampler)

from huggingface_hub import login
login("add key")

print(torch.cuda.is_available())
def set_torch_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benckmark = False
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_torch_seed()
"""# Task 2"""

print("take file")
task2_train_answer = r"D:\Watashi No Folder\aicup\audio_dataset\TRAINGING DATASET_1PHASE\Training_Dataset_01\task2_answer.txt"
task2_train_transcribe = r"D:\Watashi No Folder\aicup\audio_dataset\TRAINGING DATASET_1PHASE\Training_Dataset_01\task1_answer.txt"
task2_train_data = r'D:\Watashi No Folder\aicup\audio_dataset\TRAINGING DATASET_1PHASE\Training_Dataset_01\task2_train.tsv'
generate_annotated_audio_transcribe_parallel(task2_train_answer, task2_train_transcribe, task2_train_data, num_processes=4)

"""## Read dataset"""

print("read dataset")
from datasets import load_dataset, Features, Value

task2_data = load_dataset("csv",data_files=task2_train_data, delimiter='\t',
  features = Features({'fid': Value('string'),'content': Value('string'),'label':Value('string')}),
  column_names=['fid','content','label'])

task2_data

from collections import Counter


ctr = Counter()

for i in task2_data['train']:
  phi_labelwvalue = i['label'].split("\\n")
  phi_label = [j.split(":")[0] for j in phi_labelwvalue]
  ctr.update(phi_label)

print(ctr)

"""## Model imoprt"""
print("import model")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


task2_model_name = "meta-llama/Llama-3.2-1B-Instruct"
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
sep = '\n\n####\n\n'

special_tokens_dict = {
    'eos_token': eos,
    'bos_token': bos,
    'pad_token': pad,
    'sep_token': sep
}

print("tokenizer")
tokenizer = AutoTokenizer.from_pretrained(task2_model_name, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'left'
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("BitsAndBytes")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("AutoModelForCausalLM")
model = AutoModelForCausalLM.from_pretrained(
    task2_model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
print("resize")
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.sep_token_id = tokenizer.sep_token_id

model = prepare_model_for_kbit_training(model)

print("lora config")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj","k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print(f"{tokenizer.pad_token}: {tokenizer.pad_token_id}")

def collate_batch_with_prompt_template(batch, tokenizer, template =
  "<|endoftext|> __CONTENT__\n\n####\n\n__LABEL__ <|END|>", IGNORED_PAD_IDX = -100):

  texts = [template.replace("__LABEL__", data['label']).replace("__CONTENT__",
    data['content']) for data in list(batch)]
  encoded_seq = tokenizer(texts, padding=True)

  indexed_tks = torch.tensor(encoded_seq['input_ids'])
  attention_mask = torch.tensor(encoded_seq['attention_mask'])
  encoded_label = torch.tensor(encoded_seq['input_ids'])
  encoded_label[encoded_label == tokenizer.pad_token_id] = IGNORED_PAD_IDX

  return indexed_tks, encoded_label, attention_mask

from torch.utils.data import Dataset, DataLoader


train_data = list(task2_data['train'])
train_dataloader = DataLoader(train_data, batch_size=2, shuffle=False, collate_fn=lambda batch: collate_batch_with_prompt_template(batch, tokenizer))
titer = iter(train_dataloader)
tks, labels, masks= next(titer)
print(tks.shape)
print(tks)
print(labels)
print(masks)
print(tokenizer.decode(tks[0]))
print(tokenizer.decode(tks[1]))
next(iter(titer))

results = tokenizer(
    [f"{bos} Yeah, I imagine it would — sorry, go ahead. So it's supposed to work immediately, right? Yep. So we'll see if I'm productive tomorrow. I hope I'm productive today. I've actually been trying to plan. If I do the titles today, then I can do my laundry tomorrow. Right. I probably could bring my computer and do titles while I'm doing my laundry. If I was — but I won't do that.{sep}DATE:tomorrow\nDATE:today\nDate:today {eos}",
    f"{bos} I imagine it{sep}PHI:Null {eos}"]
    ,padding=True
)
print(results['attention_mask'][0])
print(results['attention_mask'][1])
print(tokenizer.decode(results['input_ids'][0]))
print(tokenizer.decode(results['input_ids'][1]))

BATCH_SIZE = 8
bucket_train_dataloader = DataLoader(train_data,
  batch_sampler=OpenDeidBatchSampler(train_data, BATCH_SIZE),
  collate_fn=lambda batch: collate_batch_with_prompt_template(batch, tokenizer),
  pin_memory=True)

model.print_trainable_parameters()

"""## Training"""
print("pre-training")
from transformers import get_scheduler
from tqdm import tqdm
import torch
import os


EPOCHS = 15
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-4
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 4
output_dir = r"D:\Watashi No Folder\aicup\qlora-llama"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(bucket_train_dataloader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=total_steps,
)
best_loss = float("inf")
early_stop_counter = 0

print("training")
for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    progress_bar = tqdm(enumerate(bucket_train_dataloader), total=len(bucket_train_dataloader), desc=f"Epoch {epoch+1}")

    for step, (seqs, labels, masks) in progress_bar:
        input_ids = seqs.to(device)
        labels = labels.to(device)
        attention_mask = masks.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(bucket_train_dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()

        progress_bar.set_postfix(
            loss=loss.item() * GRADIENT_ACCUMULATION_STEPS,
            lr=lr_scheduler.get_last_lr()[0]
        )

    avg_loss = total_loss / len(bucket_train_dataloader)
    print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")

    # ==== save best model ====
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
        best_path = os.path.join(output_dir, "best_adapter")
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        print(f"Best model saved at {best_path} with loss {avg_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stop patience: {early_stop_counter}/{EARLY_STOP_PATIENCE}")

    # ==== Early Stopping ====
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch+1}. Best loss: {best_loss:.4f}")
        break

print("save model")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)