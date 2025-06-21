
import os
import sys
sys.path.append('/content/drive/MyDrive/aicup')
import json
import torch
import random
import librosa
import zipfile
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, Audio, load_dataset, Features, Value
from aicup import (DataCollatorSpeechSeq2SeqWithPadding,
      transcribe_with_timestamps,
      collate_batch_with_prompt_template,
      generate_annotated_audio_transcribe_parallel,OpenDeidBatchSampler)
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
)

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
import torch
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
print("pretrained")
# config = AutoConfig.from_pretrained(
#     task2_model_name,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.pad_token_id,
#     sep_token_id=tokenizer.sep_token_id,
#     output_hidden_states=False
# )

print("AutoModelForCausalLM")
model = AutoModelForCausalLM.from_pretrained(
    task2_model_name,
    quantization_config=bnb_config,
    # config=config,
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
  r"<|endoftext|> __CONTENT__\n\n####\n\n__LABEL__ <|END|>", IGNORED_PAD_IDX = -100):

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Load finetune model"""
print("Load finetune model")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoConfig
from peft import PeftModel

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
sep = '\n\n####\n\n'
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
adapter_path = r"D:\Watashi No Folder\aicup\qlora-llama"

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.padding_side = 'left'

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, trust_remote_code=True)
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval().cuda()

"""## Output result"""
print("output result")
task2_valid_data = r'D:\Watashi No Folder\aicup\try\task1_answer.txt'
valid_data = load_dataset("csv", data_files=task2_valid_data, delimiter='\t',
  features = Features({'fid': Value('string'),'content': Value('string')}),
  column_names=['fid', 'content'])
valid_list = list(valid_data['train'])
valid_data

prompt = """<|endoftext|>
Please extract PHI entities from the following text using the format `KEY: VALUE`, one per line.

⚠️ Only use keys from this list:
PATIENT, DOCTOR, USERNAME, FAMILYNAME, PROFESSION,
ROOM, DEPARTMENT, HOSPITAL, ORGANIZATION, STREET, CITY,
DISTRICT, COUNTY, STATE, COUNTRY, ZIP, LOCATION-OTHER,
AGE, DATE, TIME, DURATION, SET,
PHONE, FAX, EMAIL, URL, IPADDR,
SSN, MEDICALRECORD, HEALTHPLAN, ACCOUNT,
LICENSE, VEHICLE, DEVICE, BIOID, IDNUM, OTHER

Do not invent new keys.
Do not tag general words like “he”, “she”, “they”, “family”, “sorry”, or vague expressions.
Only extract explicit names, locations, times, or identifiers that can directly reveal identity.

__CONTENT__

####

"""


with open(r'D:\Watashi No Folder\aicup\try\task1_answer_timestamps.json', 'r', encoding='utf-8') as file:
  audio_timestamps = json.load(file)

from collections import defaultdict
import re


train_phi_category = ['PATIENT', 'DOCTOR', 'USERNAME','FAMILYNAME','PROFESSION',
             'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY',
             'DISTRICT','COUNTY','STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
             'AGE',
             'DATE', 'TIME', 'DURATION', 'SET',
             'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
             'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT',
             'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM',
             'OTHER']

def get_anno_format(infos, audio_timestamps):
  if "\\n" in infos:
    infos = infos.replace("\\n", "\n")
  anno_list = []
  phi_dict = defaultdict(list)
  print("info:",infos)
  print("\n")
  for line in infos.split("\n"):
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    print("key:",key)
    print("\n")
    print("value:",value)
    if key in train_phi_category and value:
      # print(key)
      phi_dict[key].append(value)
  # print(phi_dict)
  remaining_timestamps = audio_timestamps.copy()
  used_indices = set()

  for phi_key, phi_values in phi_dict.items():
    for phi_value in phi_values:
      phi_tokens = phi_value.lower().strip().split()

      for i in range(len(remaining_timestamps) - len(phi_tokens) + 1):
        if any((i + j) in used_indices for j in range(len(phi_tokens))):
          continue

        match = True
        for j, phi_token in enumerate(phi_tokens):
          tsd_word = remaining_timestamps[i + j]['word'].replace("Ġ", "").replace("▁", "").strip().lower()
          if tsd_word != phi_token:
            match = False
            break

        if match:
          anno_list.append({
              "phi": phi_key,
              "st_time": remaining_timestamps[i]['start'],
              "ed_time": remaining_timestamps[i + len(phi_tokens) - 1]['end'],
              "entity": phi_value
          })
          for j in range(len(phi_tokens)):
            used_indices.add(i + j)
          break
  return anno_list

def aicup_predict(model, tokenizer, _input, audio_timestamps, template = "<|endoftext|> __CONTENT__\n\n####\n\n"):
  seeds = [template.replace("__CONTENT__", data['content']) for data in _input]
  sep = tokenizer.sep_token
  eos = tokenizer.eos_token
  pad = tokenizer.pad_token
  model.eval()
  device = model.device
  texts = tokenizer(seeds, return_tensors = 'pt', padding=True).to(device)
  outputs = []

  with torch.no_grad():
    output_tokens = model.generate(**texts, max_new_tokens=128, do_sample=  True,
    temperature=0.8,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id)
    preds = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    print("preds:",preds[0])
    for idx , pred in enumerate(preds):
      if "Null" in pred:
        continue
      phi_infos = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
      # print("phi_info:",phi_infos)
      print(_input)
      annotations = get_anno_format(phi_infos,audio_timestamps[_input[idx]['fid']]['segments'])
      for annotation in annotations:
        outputs.append(f'{_input[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_time"]}\t{annotation["ed_time"]}\t{annotation["entity"]}')
  return outputs

import torch
from tqdm import tqdm

BATCH_SIZE = 6
with open(r"D:\Watashi No Folder\aicup\try\task2_answer.txt",'w',encoding='utf8') as f:
  for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
    with torch.no_grad():
      data = valid_list[i:i+BATCH_SIZE]
      # print(data)
      # print("\n")
      outputs = aicup_predict(model, tokenizer, data, audio_timestamps)
      print(f"[DEBUG] Batch {i} outputs:", outputs)
      for o in outputs:
        f.write(o)
        f.write('\n')