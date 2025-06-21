import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
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
from transformers import pipeline
# from transformers.models.whisper import WhisperTimeStampLogitsProcessor
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

t1_train_audio_folder = r"D:\Watashi No Folder\aicup\audio_dataset\TRAINGING DATASET_1PHASE\Validation_Dataset\audio"
# t1_train_transcription_file = r"D:\Watashi No Folder\aicup\audio_dataset\TRAINGING DATASET_1PHASE\Training_Dataset_01\task1_answer.txt"
transcripts, dataset_list = {}, []

model_name = "openai/whisper-medium"  # 或者你訓練好的模型路徑
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



# 建立 pipeline
pipe = pipeline(model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, task="automatic-speech-recognition", device=device)

for file in sorted(os.listdir(t1_train_audio_folder)):
  if file.endswith(".wav"):
    try:
      file_path = os.path.join(t1_train_audio_folder, file)
      audio_array, sr = librosa.load(file_path, sr=16000)
      dataset_list.append({"audio":
                 {
                  "path":file_path,
                  "array":audio_array,
                  "sampling_rate":sr
                 }})
    except Exception as e:
      print(e)
      print(f"Can't read {file_path}:{e}")

dataset = Dataset.from_pandas(pd.DataFrame(dataset_list))

output_file = r"D:\Watashi No Folder\aicup\try\task1_answer.txt"
json_output_file = r"D:\Watashi No Folder\aicup\try\task1_answer_timestamps.json"
results = []
audio_timestamp = {}

counter = 0
for item in dataset:
    audio = item['audio']['array']
    audio_np = np.array(audio, dtype=np.float32)  # ✅ 這行是關鍵

    pipe_result = pipe(audio_np, return_timestamps="word")
    file_name = item['audio']['path'].split("\\")[-1].split(".")[0]

    # print(item['audio']['path'].split("\\")[-1].split(".")[0], pipe_result['text'])
    timestamp = []
    for word in pipe_result["chunks"]:
        timestamp.append({
            "word": word['text'],
            "start": word['timestamp'][0],
            "end": word['timestamp'][1] if word["timestamp"][1] is not None else word['timestamp'][0] + 0.3
        })
    # print(timestamp)
    audio_timestamp[file_name] = {
        "text": pipe_result["text"],
        "segments": timestamp
    }

    # print(audio_timestamp)
    counter = counter +1
    print("Processing:",counter,"/",len(dataset))
    results.append((file_name, pipe_result['text']))

with open(output_file, "w", encoding="utf-8") as f:
    for file_name, text in results:
        f.write(f"{file_name}\t{text}\n")

with open(json_output_file, "w", encoding="utf-8") as f:
  json.dump(audio_timestamp, f, ensure_ascii=False)



