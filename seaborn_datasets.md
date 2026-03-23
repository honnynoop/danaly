# Seaborn(`sns`) 내장 데이터셋 종류

> `seaborn.load_dataset(name)` 으로 불러올 수 있는 데이터셋 목록

---

## 전체 목록 확인 방법

```python
import seaborn as sns

# 사용 가능한 모든 데이터셋 이름 출력
print(sns.get_dataset_names())
```

---

## 주요 내장 데이터셋 목록

| 데이터셋 이름 | 설명 | 주요 변수 |
|---|---|---|
| `tips` | 식당 팁 데이터 | total_bill, tip, sex, smoker, day, time, size |
| `titanic` | 타이타닉 생존자 데이터 | survived, pclass, sex, age, fare, embarked 등 |
| `iris` | 붓꽃 분류 데이터 | sepal_length, sepal_width, petal_length, petal_width, species |
| `penguins` | 펭귄 종 분류 데이터 | species, island, bill_length_mm, flipper_length_mm, body_mass_g, sex |
| `diamonds` | 다이아몬드 가격 데이터 | carat, cut, color, clarity, depth, table, price, x, y, z |
| `flights` | 항공편 승객 수 (시계열) | year, month, passengers |
| `mpg` | 자동차 연비 데이터 | mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, name |
| `planets` | 외계행성 탐사 데이터 | method, number, orbital_period, mass, distance, year |
| `fmri` | 뇌 fMRI 신호 데이터 | subject, timepoint, event, region, signal |
| `attention` | 주의력 실험 데이터 | subject, attention, solutions, score |
| `exercise` | 운동 실험 데이터 | id, diet, pulse, time, kind |
| `car_crashes` | 미국 주별 교통사고 데이터 | total, speeding, alcohol, not_distracted, no_previous, ins_premium, ins_losses, abbrev |
| `anscombe` | 앤스컴 사중주 데이터 | dataset, x, y |
| `brain_networks` | 뇌 네트워크 상관관계 데이터 | 다수의 네트워크 컬럼 |
| `dowjones` | 다우존스 지수 (시계열) | Date, Price |
| `gammas` | 감마 분포 신호 데이터 | timepoint, ROI, BOLD signal |
| `geyser` | 올드 페이스풀 간헐천 분출 | duration, waiting |
| `glue` | NLP 벤치마크 점수 데이터 | Model, Task, Score, Type |
| `healthexp` | 기대수명 & 의료비 데이터 | Year, Country, Spending_USD, Life_Expectancy |
| `seaice` | 북극 해빙 면적 (시계열) | Date, Extent |
| `taxis` | NYC 택시 데이터 | pickup, dropoff, passengers, distance, fare, tip, tolls, total, color, payment, pickup_zone 등 |

---

## 용도별 추천 데이터셋

| 용도 | 추천 데이터셋 |
|---|---|
| **분류 문제** | `iris`, `titanic`, `penguins` |
| **회귀 문제** | `tips`, `diamonds`, `mpg` |
| **시계열** | `flights`, `dowjones`, `seaice` |
| **시각화 연습** | `anscombe`, `tips`, `fmri` |
| **클러스터링** | `iris`, `penguins`, `brain_networks` |

---

## 데이터셋 불러오기 예시

```python
import seaborn as sns

# 예시 1 - tips
tips = sns.load_dataset('tips')
print(tips.head())

# 예시 2 - titanic
titanic = sns.load_dataset('titanic')
print(titanic.info())

# 예시 3 - iris
iris = sns.load_dataset('iris')
print(iris.describe())
```

---

> 💡 **빅분기 / ADP 실기 시험**에서는 주로 `iris`, `titanic`, `tips`, `diamonds` 계열이 자주 활용됩니다.
