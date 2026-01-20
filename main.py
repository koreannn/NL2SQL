import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
from utils import load_config, load_json
from train import CustomTrainer
from dataset import Dataset, data_split

LABELS = ["O", "B-COL", "I-COL", "B-VALUE", "I-VALUE"] # B-COL: (컬럼 이름)시작 토큰, B-VALUE: ()시작 토큰
LABEL2ID = {l: i for i, l in enumerate(LABELS)} # {0: "O", 1: "B-COL", 2: "I-COL", ..., 4: "I-VALUE"}
ID2LABEL = {i: l for l, i in LABEL2ID.items()} # {"O": 0, 1: "B-COL", ..., 4: "I-VALUE"}

config = load_config("config.yaml")
dataset_path = config["data"]["train_labeled_data_path"] + "/TEXT_NL2SQL_label_publicdata_climate.json"
data = load_json(dataset_path)
train_data, valid_data = data_split(data)
model_name = config["model"]["model_name"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = Dataset(train_data, tokenizer, max_length = config["model"]["train_param"]["max_length"])
valid_dataset = Dataset(valid_data, tokenizer, max_length = config["model"]["train_param"]["max_length"])
data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id = -100)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels = len(LABELS),
    id2label = ID2LABEL,
    label2id = LABEL2ID,
)

trainer = CustomTrainer(train_dataset, valid_dataset, model, tokenizer, data_collator)
trainer.train()