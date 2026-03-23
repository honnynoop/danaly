# taxis — 뉴욕시 택시 데이터

> 뉴욕시 택시 승하차 기록 데이터. datetime 처리, 구역별 집계, 요금 예측 실습에 적합.


---


## 기본 정보

- **행 수**: 6,433개
- **열 수**: 14개
- **주요 용도**: 회귀 분석 (total 예측), datetime 처리, 구역별 집계, 범주형 분석
- **시험 활용**: 빅분기 Type3 (회귀), ADP (전처리·피처 엔지니어링)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `pickup` | object | 승차 일시 (datetime) |
| `dropoff` | object | 하차 일시 (datetime) |
| `passengers` | float64 | 탑승 인원 |
| `distance` | float64 | 이동 거리 (마일) |
| `fare` | float64 | 기본 요금 (달러) |
| `tip` | float64 | 팁 금액 |
| `tolls` | float64 | 통행료 |
| `total` | float64 | 총 결제 금액 |
| `color` | object | 택시 색상 (yellow / green) |
| `payment` | object | 결제 방법 (credit card / cash 등) |
| `pickup_zone` | object | 승차 구역명 |
| `dropoff_zone` | object | 하차 구역명 |
| `pickup_borough` | object | 승차 자치구 |
| `dropoff_borough` | object | 하차 자치구 |

## 샘플 데이터 (상위 3행)

| pickup | dropoff | passengers | distance | fare | tip | tolls | total | color | payment |
|---|---|---|---|---|---|---|---|---|---|
| 2019-03-23 20:21 | 2019-03-23 20:27 | 1.0 | 1.10 | 7.0 | 2.15 | 0.0 | 12.95 | yellow | credit card |
| 2019-03-04 16:11 | 2019-03-04 16:19 | 1.0 | 0.79 | 5.0 | 0.00 | 0.0 | 9.30 | yellow | cash |

## 코드 예시

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset('taxis')
df['pickup'] = pd.to_datetime(df['pickup'])
df['hour'] = df['pickup'].dt.hour

# 시간대별 평균 요금
print(df.groupby('hour')['fare'].mean())

# 결제 방법별 팁 비율
df['tip_rate'] = df['tip'] / df['fare']
print(df.groupby('payment')['tip_rate'].mean())
```
