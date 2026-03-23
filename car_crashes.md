# car_crashes — 미국 주별 교통사고 데이터

> 미국 50개 주의 교통사고 관련 통계 데이터. 지리적 시각화 및 상관 분석에 활용.


---


## 기본 정보

- **행 수**: 51개
- **열 수**: 8개
- **주요 용도**: 상관 분석, 히트맵, 지역별 비교, 회귀 분석
- **시험 활용**: ADP (상관분석·회귀)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `total` | float64 | 10억 마일당 사고 건수 |
| `speeding` | float64 | 과속 관련 사고 비율 |
| `alcohol` | float64 | 음주 관련 사고 비율 |
| `not_distracted` | float64 | 비 집중 관련 사고 비율 |
| `no_previous` | float64 | 사고 이력 없는 운전자 비율 |
| `ins_premium` | float64 | 자동차 보험료 (달러) |
| `ins_losses` | float64 | 보험 손실 비용 (달러) |
| `abbrev` | object | 주 약자 (AL, AK, ...) |

## 샘플 데이터 (상위 3행)

| total | speeding | alcohol | not_distracted | no_previous | ins_premium | ins_losses | abbrev |
|---|---|---|---|---|---|---|---|
| 18.8 | 7.332 | 5.640 | 18.048 | 15.040 | 784.55 | 145.08 | AL |
| 18.1 | 7.421 | 4.525 | 16.290 | 17.014 | 1053.48 | 133.93 | AK |
| 18.6 | 6.510 | 5.208 | 15.624 | 17.856 | 899.47 | 110.35 | AZ |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('car_crashes')

# 상관계수 행렬
print(df.drop('abbrev', axis=1).corr())

# 보험료 상위 5개 주
print(df.nlargest(5, 'ins_premium')[['abbrev','ins_premium']])
```
