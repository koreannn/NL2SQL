from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils import load_config, load_json
from dataset import Dataset
from loguru import logger
from collections import Counter
import json
import torch
import argparse

config = load_config("config.yaml")

def predict(sentence: str, model_dir: str, tokenizer):
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[i] for i in pred_ids]
    return tokens, labels


def group_labeled_spans(tokens: list[str], labels: list[str], tokenizer) -> dict[str, list[str]]:
    """
    Group contiguous BIO spans into text chunks.
    Returns {"COL": [...], "VALUE": [...]}.
    """
    entities: dict[str, list[str]] = {"COL": [], "VALUE": []}
    current_tokens: list[str] = []
    current_type: str | None = None

    def flush():
        nonlocal current_tokens, current_type
        if current_type and current_tokens:
            text = tokenizer.convert_tokens_to_string(current_tokens).strip()
            if text:
                entities[current_type].append(text)
        current_tokens = []
        current_type = None

    for tok, lab in zip(tokens, labels):
        if tok in tokenizer.all_special_tokens:
            flush()
            continue

        if lab.startswith("B-"):
            flush()
            span_type = lab[2:]
            if span_type in entities:
                current_type = span_type
                current_tokens = [tok]
            else:
                current_type = None
        elif lab.startswith("I-"):
            span_type = lab[2:]
            if current_type == span_type:
                current_tokens.append(tok)
            elif span_type in entities:
                flush()
                current_type = span_type
                current_tokens = [tok]
            else:
                flush()
        else:
            flush()

    flush()
    return entities


def main(argv: list[str] | None = None):
    model_dir = config["model"]["train_param"]["output_dir"]
    model_name = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_path = config["data"]["test_labeled_data_path"] + "/TEXT_NL2SQL_label_publicdata_climate.json"
    annotation_path = config["data"]["test_annotation_path"] + "/publicdata_climate" + "_db_annotation.json"
    test_data = load_json(dataset_path)
    annotation = load_json(annotation_path)
    
    test_utterances = [item["utterance"] for item in test_data]
    values = [[value["token"] for value in item["values"]] for item in test_data]
    cols = [[col["token"] for col in item["cols"]] for item in test_data]
    
    annotation_by_db_id: dict[str, dict] = {item["db_id"]: item for item in annotation}
    db_to_col_index: dict[str, dict[str, int]] = {}
    db_to_col_original_index: dict[str, dict[str, int]] = {}
    db_to_table_name: dict[str, str] = {}
    db_to_col_display: dict[str, list[str]] = {}
    for db_id, item in annotation_by_db_id.items():
        col_names = [name for (_tbl, name) in item["column_names"][1:]]
        col_names_original = [name for (_tbl, name) in item["column_names_original"][1:]]
        table_name = item["table_names_original"][0] if item["table_names_original"] else db_id
        db_to_col_index[db_id] = {name: idx for idx, name in enumerate(col_names)}
        db_to_col_original_index[db_id] = {name: idx for idx, name in enumerate(col_names_original)}
        db_to_table_name[db_id] = table_name
        db_to_col_display[db_id] = [
            f"{orig}({ko})" for orig, ko in zip(col_names_original, col_names)
        ]
    
    col_index = []
    val_index = []
    output_rows: list[dict[str, str]] = []
    
    # cols 맞춘 횟수, values 맞춘 횟수
    cols_count, values_count = 0, 0
    for i in range(len(test_utterances)):
        curr_tokens, curr_labels = predict(test_utterances[i], model_dir = model_dir, tokenizer = tokenizer)
        grouped = group_labeled_spans(curr_tokens, curr_labels, tokenizer)
        
        # 각 자연어에 대해 토큰 단위로 태깅 확인할 떄
        # for tok, lab in zip(curr_tokens, curr_labels): 
        #     print(f"{tok:5s} -> {lab}")
        
        # 각 iter마다 로그 찍어보기용
        logger.info(f"자연어 문장: {test_utterances[i]}")
        logger.info(f"예측한 col 단어: {grouped["COL"]}")
        logger.info(f"정답 col 단어: {cols[i]}")
        logger.info(f"예측한 values 단어: {grouped["VALUE"]}")
        logger.info(f"정답 values 단어: {values[i]}")
        print()
        
        # cols 점수 계산
        if (Counter(grouped["COL"]) == Counter(cols[i])):
            cols_count += 1
        # values 점수 계산
        if (Counter(grouped["VALUE"]) == Counter(values[i])):
            values_count += 1
        
        
        # cols/values 텍스트와 컬럼 인덱스 매칭
        db_id = test_data[i]["db_id"]
        col_map = db_to_col_index.get(db_id, {})
        col_pairs = [(text, col_map[text] + 1) for text in cols[i] if text in col_map]
        val_pairs = [(text, col_map[text] + 1) for text in values[i] if text in col_map]
        col_index.append(col_pairs)
        val_index.append(val_pairs)

        table_name = db_to_table_name.get(db_id, db_id)
        col_display_map = db_to_col_display.get(db_id, [])
        # Use predicted COL/ VALUE spans only
        pred_cols = grouped["COL"]
        pred_vals = grouped["VALUE"]
        pred_col_pairs = [(text, col_map[text]) for text in pred_cols if text in col_map]
        pred_val_pairs = [(text, col_map[text]) for text in pred_vals if text in col_map]
        pred_pairs = pred_col_pairs + pred_val_pairs
        pred_pairs.sort(key=lambda x: x[1])
        seen_idx = set()
        pred_col_display = []
        for _text, idx in pred_pairs:
            if idx in seen_idx:
                continue
            if idx < len(col_display_map):
                pred_col_display.append(col_display_map[idx])
                seen_idx.add(idx)
        columns_display = ", ".join(pred_col_display)
        value_display = ", ".join(pred_vals)
        schema = (
            f"table {table_name} columns {columns_display}"
            f" values {value_display}"
        )
        output_rows.append(
            {
                "translate_to_sql": "translate to sql",
                "question": test_utterances[i],
                "schema": schema,
            }
        )
        
        if i == 20:
            break
    output_path = "./inference_schema.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    logger.info(f"inference json saved: {output_path}")
    
    cols_score = cols_count / len(cols)
    values_score = values_count / len(values)
    logger.info(f"cols 점수: {cols_score}")
    logger.info(f"values 점수: {values_score}")


if __name__ == "__main__":
    main()