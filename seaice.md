# seaice — 북극 해빙 면적 (시계열)

> 1979년부터 측정된 북극 해빙 면적 데이터. 기후변화 시각화 및 시계열 분석에 활용.


---


## 기본 정보

- **행 수**: 14,219개
- **열 수**: 2개
- **주요 용도**: 시계열 분석, 추세선, 계절성 분해, datetime 처리
- **시험 활용**: ADP (시계열 분석)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `Date` | object | 측정 날짜 (YYYY-MM-DD) |
| `Extent` | float64 | 해빙 면적 (백만 km²) |

## 샘플 데이터 (상위 3행)

| Date | Extent |
|---|---|
| 1979-01-01 | 10.231 |
| 1979-01-03 | 10.420 |
| 2023-12-30 | 12.091 |

## 코드 예시

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset('seaice')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 연도별 평균 해빙 면적 (감소 추세 확인)
print(df.groupby('Year')['Extent'].mean().tail(10))
```
