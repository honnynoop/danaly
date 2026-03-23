# titanic — 타이타닉 생존자 데이터

> 1912년 타이타닉 호 침몰 사고의 탑승객 데이터셋. 분류 모델 학습과 EDA 실습의 대표 데이터셋.


---


## 기본 정보

- **행 수**: 891개
- **열 수**: 15개
- **주요 용도**: 이진 분류, 결측값 처리, 피처 엔지니어링, EDA
- **시험 활용**: 빅분기 Type2/3 (분류 모델), ADP (머신러닝 전처리)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `survived` | int64 | 생존 여부 (0=사망, 1=생존) ← 타겟 |
| `pclass` | int64 | 객실 등급 (1/2/3) |
| `sex` | object | 성별 (male / female) |
| `age` | float64 | 나이 (결측값 존재) |
| `sibsp` | int64 | 함께 탑승한 형제/배우자 수 |
| `parch` | int64 | 함께 탑승한 부모/자녀 수 |
| `fare` | float64 | 운임 요금 |
| `embarked` | object | 탑승 항구 (C=Cherbourg, Q=Queenstown, S=Southampton) |
| `class` | category | 객실 등급 텍스트 (First/Second/Third) |
| `who` | object | 탑승객 분류 (man/woman/child) |
| `adult_male` | bool | 성인 남성 여부 |
| `deck` | category | 갑판 (A~G, 결측 많음) |
| `embark_town` | object | 탑승 도시명 |
| `alive` | object | 생존 텍스트 (yes/no) |
| `alone` | bool | 혼자 탑승 여부 |

## 샘플 데이터 (상위 3행)

| survived | pclass | sex | age | sibsp | parch | fare | embarked |
|---|---|---|---|---|---|---|---|
| 0 | 3 | male | 22.0 | 1 | 0 | 7.25 | S |
| 1 | 1 | female | 38.0 | 1 | 0 | 71.28 | C |
| 1 | 3 | female | 26.0 | 0 | 0 | 7.92 | S |

## 코드 예시

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')

# 결측값 확인
print(df.isnull().sum())

# 나이 결측 처리
df['age'].fillna(df['age'].median(), inplace=True)

# 성별·등급별 생존율
print(df.groupby(['sex', 'pclass'])['survived'].mean())
```
