# diamonds — 다이아몬드 가격 데이터

> 약 54,000개 다이아몬드의 품질과 가격 정보. 회귀 분석 및 범주형 변수 처리 실습에 활용.


---


## 기본 정보

- **행 수**: 53,940개
- **열 수**: 10개
- **주요 용도**: 회귀 분석 (price 예측), 순서형 범주 인코딩, 대용량 데이터 처리
- **시험 활용**: 빅분기 Type3 (회귀), ADP (회귀·범주 인코딩)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `carat` | float64 | 무게 (캐럿) |
| `cut` | category | 컷 품질 (Fair < Good < Very Good < Premium < Ideal) |
| `color` | category | 색상 등급 (D=최상 ~ J=최하) |
| `clarity` | category | 투명도 (I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF) |
| `depth` | float64 | 깊이 비율 (%) |
| `table` | float64 | 테이블 너비 비율 (%) |
| `price` | int64 | 가격 (달러) ← 타겟 |
| `x` | float64 | 길이 (mm) |
| `y` | float64 | 너비 (mm) |
| `z` | float64 | 높이 (mm) |

## 샘플 데이터 (상위 3행)

| carat | cut | color | clarity | depth | table | price | x | y | z |
|---|---|---|---|---|---|---|---|---|---|
| 0.23 | Ideal | E | SI2 | 61.5 | 55 | 326 | 3.95 | 3.98 | 2.43 |
| 0.21 | Premium | E | SI1 | 59.8 | 61 | 326 | 3.89 | 3.84 | 2.31 |
| 0.23 | Good | E | VS1 | 56.9 | 65 | 327 | 4.05 | 4.07 | 2.31 |

## 코드 예시

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset('diamonds')

# cut 순서 인코딩
cut_order = ['Fair','Good','Very Good','Premium','Ideal']
df['cut_enc'] = df['cut'].map({v:i for i,v in enumerate(cut_order)})

# 캐럿별 평균 가격
print(df.groupby('cut')['price'].mean().sort_values())
```
