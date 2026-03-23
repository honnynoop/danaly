# exercise — 운동 실험 데이터

> 운동 종류와 식이 요법에 따른 심박수 변화 데이터. 반복 측정 시각화에 활용.


---


## 기본 정보

- **행 수**: 90개
- **열 수**: 6개
- **주요 용도**: 반복 측정 시각화, lineplot, ANOVA
- **시험 활용**: ADP (시각화·검정)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `Unnamed: 0` | int64 | 행 인덱스 |
| `id` | int64 | 피험자 ID |
| `diet` | object | 식이 유형 (no fat / low fat) |
| `pulse` | int64 | 심박수 (bpm) |
| `time` | object | 측정 시간 (1 min / 15 min / 30 min) |
| `kind` | object | 운동 종류 (rest / walking / running) |

## 샘플 데이터 (상위 3행)

| Unnamed: 0 | id | diet | pulse | time |
|---|---|---|---|---|
| 1 | no fat | 85 | 1 min | rest |
| 2 | no fat | 85 | 1 min | rest |
| 3 | no fat | 88 | 1 min | walking |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('exercise')

# 운동 종류별·시간별 평균 심박수
pivot = df.groupby(['time','kind'])['pulse'].mean().unstack()
print(pivot)
```
