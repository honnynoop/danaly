# attention — 주의력 실험 데이터

> 주의력 조건에 따른 문제 풀이 점수 실험 데이터. 반복 측정 분산분석(RM-ANOVA) 실습에 활용.


---


## 기본 정보

- **행 수**: 60개
- **열 수**: 5개
- **주요 용도**: 반복 측정 ANOVA, 그룹 간 비교, 막대 그래프 시각화
- **시험 활용**: ADP (통계 검정, 분산분석)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `Unnamed: 0` | int64 | 행 인덱스 |
| `subject` | int64 | 피험자 번호 (1~30) |
| `attention` | object | 주의력 조건 (divided / focused) |
| `solutions` | int64 | 문제 수 (1~3) |
| `score` | int64 | 점수 |

## 샘플 데이터 (상위 3행)

| Unnamed: 0 | subject | attention | solutions |
|---|---|---|---|
| 1 | divided | 1 | 2 |
| 2 | divided | 2 | 3 |
| 3 | divided | 3 | 7 |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('attention')

# 주의력 조건별 평균 점수
print(df.groupby('attention')['score'].mean())

# 문제 수별 점수 분포
print(df.groupby('solutions')['score'].describe())
```
