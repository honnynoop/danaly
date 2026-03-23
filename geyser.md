# geyser — 올드 페이스풀 간헐천 분출 데이터

> 옐로스톤 국립공원 올드 페이스풀 간헐천의 분출 시간 및 대기 시간 데이터. 이변량 분포 시각화에 적합.


---


## 기본 정보

- **행 수**: 272개
- **열 수**: 2개
- **주요 용도**: 이변량 분포 시각화, KDE 플롯, 군집 분석 (2군집 분포)
- **시험 활용**: 빅분기 Type1 (기초 통계), ADP (분포 분석)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `duration` | float64 | 분출 지속 시간 (분) |
| `waiting` | int64 | 다음 분출까지 대기 시간 (분) |

## 샘플 데이터 (상위 3행)

| duration | waiting |
|---|---|
| 3.600 | 79 |
| 1.800 | 54 |
| 3.333 | 74 |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('geyser')

print(df.describe())

# 분출 시간 2군집 확인
short = df[df['duration'] < 3]
long  = df[df['duration'] >= 3]
print("짧은 분출 평균 대기:", short['waiting'].mean())
print("긴 분출 평균 대기:",  long['waiting'].mean())
```
