from __future__ import annotations
from dataset import LABEL2ID
from utils import load_config
from transformers import (  
            Trainer,
            TrainingArguments,
        )
import torch
import numpy as np

config = load_config("config.yaml")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true, y_pred = [], []
    for pred, label in zip(preds, labels):
        for pi, li in zip(pred, label):
            if li == -100:
                continue
            y_true.append(li)
            y_pred.append(pi)

    # Token-level micro F1 excluding "O"
    o_id = LABEL2ID["O"]
    mask = [t != o_id for t in y_true]
    if sum(mask) == 0:
        return {"f1_micro_no_o": 0.0, "acc": float(np.mean(np.array(y_true) == np.array(y_pred)))}

    yt = np.array(y_true)[mask]
    yp = np.array(y_pred)[mask]

    classes = [LABEL2ID["B-COL"], LABEL2ID["I-COL"], LABEL2ID["B-VALUE"], LABEL2ID["I-VALUE"]]
    tp = fp = fn = 0
    for c in classes:
        tp += int(np.sum((yp == c) & (yt == c)))
        fp += int(np.sum((yp == c) & (yt != c)))
        fn += int(np.sum((yp != c) & (yt == c)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return {"f1_micro_no_o": float(f1), "acc": acc}

class CustomTrainer:
    def __init__(self, train_dataset, valid_dataset, model, tokenizer, data_collator):
        self.train_args = TrainingArguments(
            output_dir = config["model"]["train_param"]["output_dir"],
            learning_rate = config["model"]["train_param"]["lr"],
            per_device_train_batch_size = config["model"]["train_param"]["batch_size"],
            per_device_eval_batch_size = config["model"]["train_param"]["eval_batch_size"],
            num_train_epochs = config["model"]["train_param"]["epoch"],
            weight_decay = config["model"]["train_param"]["weight_decay"],
            eval_steps = config["model"]["train_param"]["eval_strategy"],
            save_steps = config["model"]["train_param"]["save_steps"],
            logging_steps = config["model"]["train_param"]["logging_steps"],
            save_total_limit = config["model"]["train_param"]["save_total_limit"],
            fp16 = config["model"]["train_param"].get("fp16", torch.cuda.is_available()),
            report_to = config["model"]["train_param"]["report_to"],
        )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = Trainer(
            model = model,
            args = self.train_args,
            train_dataset = self.train_dataset,
            eval_dataset = self.valid_dataset,
            tokenizer = self.tokenizer,
            data_collator = data_collator,
            compute_metrics = compute_metrics,
        )
        self.output_dir = config["model"]["train_param"]["output_dir"]
        
    def train(self):
        self.trainer.train()
        self.trainer.evaluate()
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        return
