# 프로젝트 개요
사용자가 입력한 자연어 문장을 DB 검색에 적합한 쿼리 문으로 변환하는 과제입니다.

<br>

### 도전 과제
SQL 쿼리의 Syntax와 semantic 정확도 둘 다 잘 잡아야하기 때문에 성능을 높이기 어려운 Task입니다. 이를 유념하고 작업을 수행해야합니다.

<br>

### 아키텍쳐

```mermaid
flowchart LR
  A[사용자 자연어 입력] --> B[Schema Linking]
  B --> C[Prompt 구성]
  C --> D[pko-T5 (Text-to-SQL)]
  D --> E[SQL Query 출력]
```

<br>

### 디렉터리 구조

```bash
root/
├── __pycache__/
├── .git/
├── data/
│   ├── Training/
│   └── Validation/
├── GitHub/
├── outputs/
├── .gitignore
├── config.yaml
├── dataset.py
├── infer.py
├── inference_schema.json
├── main.py
├── note.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py
```

<br>

### 데이터 출처: 
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71351

---

# fff

데이터 및 체크포인트 등은 ignore처리해서 올려주시기 바랍니다.
