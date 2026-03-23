# dowjones — 다우존스 지수 (시계열)

> 다우존스 산업평균지수의 월별 종가 데이터. 금융 시계열 분석 실습에 활용.


---


## 기본 정보

- **행 수**: 121개
- **열 수**: 2개
- **주요 용도**: 시계열 분석, 이동 평균, 수익률 계산, 추세 분석
- **시험 활용**: ADP (시계열 분석)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `Date` | object | 날짜 (월별, YYYY-MM-DD) |
| `Price` | float64 | 다우존스 지수 종가 |

## 샘플 데이터 (상위 3행)

| Date | Price |
|---|---|
| 1914-01-01 | 53.0 |
| 1914-02-01 | 49.95 |
| 2023-01-01 | 33715.0 |

## 코드 예시

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset('dowjones')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 월간 수익률
df['return'] = df['Price'].pct_change()

# 이동 평균
df['MA12'] = df['Price'].rolling(12).mean()
print(df.tail(10))
```
