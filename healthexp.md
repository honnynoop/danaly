# healthexp — 기대수명 & 의료비 데이터

> 주요 국가들의 연도별 의료 지출과 기대수명 데이터. 회귀 분석 및 국가 간 비교 시각화에 활용.


---


## 기본 정보

- **행 수**: 274개
- **열 수**: 4개
- **주요 용도**: 시계열 분석, 국가 간 비교, 선형 회귀 (지출 vs 수명)
- **시험 활용**: ADP (시계열·회귀·비교 분석)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `Year` | int64 | 연도 |
| `Country` | object | 국가명 |
| `Spending_USD` | float64 | 1인당 의료 지출 (달러) |
| `Life_Expectancy` | float64 | 기대 수명 (년) |

## 샘플 데이터 (상위 3행)

| Year | Country | Spending_USD | Life_Expectancy |
|---|---|---|---|
| 1970 | USA | 374.0 | 70.9 |
| 1970 | Great Britain | 170.0 | 71.9 |
| 2020 | USA | 11859.0 | 77.3 |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('healthexp')

# 최신 연도 데이터
latest = df[df['Year'] == df['Year'].max()]
print(latest.sort_values('Spending_USD', ascending=False))

# 미국 시계열
usa = df[df['Country'] == 'USA']
print(usa[['Year','Spending_USD','Life_Expectancy']])
```
