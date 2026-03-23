# glue — NLP 벤치마크 점수 데이터

> GLUE 벤치마크에서 다양한 NLP 모델의 태스크별 성능 점수. 모델 비교 시각화에 활용.


---


## 기본 정보

- **행 수**: 816개
- **열 수**: 4개
- **주요 용도**: 모델 비교 시각화, barplot, 피벗 테이블
- **시험 활용**: ADP (데이터 분석·시각화)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `Model` | object | NLP 모델 이름 (BERT, GPT 등) |
| `Task` | object | NLP 태스크 이름 (SST, MNLI 등) |
| `Score` | float64 | 성능 점수 |
| `Type` | object | 모델 유형 (Transfer, ELMo 등) |

## 샘플 데이터 (상위 3행)

| Model | Task | Score | Type |
|---|---|---|---|
| BiLSTM | CoLA | 6.4 | ELMo |
| BiLSTM+ELMo | CoLA | 36.0 | ELMo |
| BERT | CoLA | 60.5 | Transfer |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('glue')

# 태스크별 최고 점수
print(df.groupby('Task')['Score'].max())

# 모델 유형별 평균 점수
print(df.groupby('Type')['Score'].mean())
```
