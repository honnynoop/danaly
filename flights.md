# flights — 항공편 승객 수 (시계열)

> 1949~1960년 월별 항공편 승객 수. 시계열 분석 및 히트맵 시각화 실습 데이터.


---


## 기본 정보

- **행 수**: 144개
- **열 수**: 3개
- **주요 용도**: 시계열 분석, pivot_table + 히트맵, 계절성 분석
- **시험 활용**: 빅분기 Type1 (데이터 변환), ADP (시계열·시각화)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `year` | int64 | 연도 (1949~1960) |
| `month` | category | 월 이름 (Jan~Dec) |
| `passengers` | int64 | 월별 승객 수 (천 명) |

## 샘플 데이터 (상위 3행)

| year | month | passengers |
|---|---|---|
| 1949 | Jan | 112 |
| 1949 | Feb | 118 |
| 1949 | Mar | 132 |

## 코드 예시

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset('flights')

# 피벗 테이블 (히트맵용)
pivot = df.pivot_table(index='month', columns='year', values='passengers')

# 연도별 총 승객 수
print(df.groupby('year')['passengers'].sum())
```
