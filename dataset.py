from __future__ import annotations
from typing import Dict, List, Tuple

LABELS = ["O", "B-COL", "I-COL", "B-VALUE", "I-VALUE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def _char_spans_from_items(
    utterance: str, items: List[Dict], kind: str
) -> List[Tuple[int, int, str]]: # [(start_pos, end_pos, "COL(종류)"), (), ..]
    """
    원본 데이터셋에는 문자 span단위로 존재함 -> 토크나이저 단위 span으로 매칭시키기 위한 단위 변환 함수
    e.g.
    input: 
        utterance = "2010년에 폐업했던 영업장 면적과 소재지 전화번호를 알려줘", 
        items = [
            {"token": "영업장", "start": 12, "column_index": 0},
            {"token": "면적", "start": 16, "column_index": 1},
        kind = "COL"
    output: 
        [
            (12, 15, "COL"),  # "영업장" (길이 3)
            (16, 18, "COL"),  # "면적" (길이 2)
    ]
    """
    spans: List[Tuple[int, int, str]] = []
    for item in items:
        tok = item["token"]
        start = item["start"]

        end = start + len(tok)
        if utterance[start:end] != tok:
            candidates = []
            idx = utterance.find(tok)
            while idx != -1:
                candidates.append(idx)
                idx = utterance.find(tok, idx + 1)
            if not candidates:
                continue
            start = min(candidates, key=lambda x: abs(x - start))
            end = start + len(tok)

        if kind == "COL":
            spans.append((start, end, "COL"))
        elif kind == "VALUE":
            spans.append((start, end, "VALUE"))
        else:
            raise ValueError(f"Unknown kind: {kind}")
    return spans


def _assign_bio_to_tokens(
    offset_mapping: List[Tuple[int, int]],
    col_spans: List[Tuple[int, int]],
    value_spans: List[Tuple[int, int]],
) -> List[int]:
    """
    BIO 라벨로 변환하는 함수
    input:
        offset_mapping = [(0, 2), (2, 4), (4, 6), (6, 8)]
        col_spans = [(2, 6)]      # 문자 2~6 구간이 COL
        value_spans = [(6, 8)]    # 문자 6~8 구간이 VALUE
    output:
        [0, 1, 2, 3]
        # LABELS = ["O", "B-COL", "I-COL", "B-VALUE", "I-VALUE"]
        # LABEL2ID = {"O":0, "B-COL":1, "I-COL":2, "B-VALUE":3, "I-VALUE":4}
    """
    labels = [LABEL2ID["O"]] * len(offset_mapping)
    for i, (ts, te) in enumerate(offset_mapping):
        if ts == te:
            labels[i] = -100

    def apply_span(spans: List[Tuple[int, int]], b_label: str, i_label: str):
        for (s, e) in spans:
            started = False
            for i, (ts, te) in enumerate(offset_mapping):
                if labels[i] == -100:
                    continue
                if te <= s or ts >= e:
                    continue
                if not started:
                    labels[i] = LABEL2ID[b_label]
                    started = True
                else:
                    labels[i] = LABEL2ID[i_label]

    apply_span(col_spans, "B-COL", "I-COL")
    apply_span(value_spans, "B-VALUE", "I-VALUE")
    return labels

class Dataset:
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        utterance = ex["utterance"]

        cols = ex.get("cols", []) or []
        values = ex.get("values", []) or []

        col_spans_items = _char_spans_from_items(utterance, cols, kind = "COL")
        value_spans_items = _char_spans_from_items(utterance, values, kind = "VALUE")

        col_spans = [(s, e) for (s, e, _) in col_spans_items]
        value_spans = [(s, e) for (s, e, _) in value_spans_items]

        enc = self.tokenizer(
            utterance,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        offsets = enc["offset_mapping"]
        token_labels = _assign_bio_to_tokens(offsets, col_spans, value_spans)

        # Ignore offsets in model input; Trainer collator can't handle it by default
        enc.pop("offset_mapping", None)
        enc["labels"] = token_labels
        return enc


def data_split(data: List[Dict], seed: int = 42, train_ratio: float = 0.9):
    import numpy as np  # type: ignore

    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    cut = int(len(data) * train_ratio)
    train_idx, valid_idx = idx[:cut], idx[cut:]
    train = [data[i] for i in train_idx]
    valid = [data[i] for i in valid_idx]
    return train, valid