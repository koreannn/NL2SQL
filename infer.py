from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import argparse

def predict(sentence: str, model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[i] for i in pred_ids]
    return tokens, labels, tokenizer


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
    parser = argparse.ArgumentParser(description="Inference for NL2SQL span tagger (COL/VALUE)")
    parser.add_argument(
        "--model-dir",
        default="./span_tagger_output",
        help="학습된 모델 디렉토리(Trainer save_model 결과).",
    )
    parser.add_argument(
        "--sentence",
        default="작업 일시가 2021년인 약국의 이름과 주소를 찾아줘",
        help="추론할 문장",
    )
    args = parser.parse_args()

    tokens, labels, tokenizer = predict(args.sentence, model_dir=args.model_dir)
    grouped = group_labeled_spans(tokens, labels, tokenizer)

    for tok, lab in zip(tokens, labels):
        print(f"{tok:10s} -> {lab}")

    
    print("\n[COL spans]")
    for text in grouped["COL"]:
        print(text)
    print("\n[VALUE spans]")
    for text in grouped["VALUE"]:
        print(text)


if __name__ == "__main__":
    main()