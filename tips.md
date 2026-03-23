# tips — 식당 팁 데이터

> 미국 식당에서 수집된 팁 관련 데이터셋. 회귀 분석, 범주형 변수 시각화 실습에 많이 활용됨.


---


## 기본 정보

- **행 수**: 244개
- **열 수**: 7개
- **주요 용도**: 회귀 분석 (tip ~ total_bill), 범주형 시각화, 그룹별 분석
- **시험 활용**: 빅분기 Type1 (기초 분석), ADP (회귀·시각화)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `total_bill` | float64 | 식사 총 금액 (달러) |
| `tip` | float64 | 팁 금액 (달러) |
| `sex` | category | 성별 (Male / Female) |
| `smoker` | category | 흡연 여부 (Yes / No) |
| `day` | category | 요일 (Thur / Fri / Sat / Sun) |
| `time` | category | 식사 시간대 (Lunch / Dinner) |
| `size` | int64 | 테이블 인원 수 |

## 샘플 데이터 (상위 3행)

| total_bill | tip | sex | smoker | day | time | size |
|---|---|---|---|---|---|---|
| 16.99 | 1.01 | Female | No | Sun | Dinner | 2 |
| 10.34 | 1.66 | Male | No | Sun | Dinner | 3 |
| 21.01 | 3.50 | Male | No | Sun | Dinner | 3 |

## 코드 예시

```python
import seaborn as sns
df = sns.load_dataset('tips')

# 기본 확인
print(df.head())
print(df.info())
print(df.describe())

# 팁 비율 컬럼 추가
df['tip_rate'] = df['tip'] / df['total_bill']

# 성별 평균 팁
print(df.groupby('sex')['tip'].mean())
```
