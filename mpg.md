# mpg — 자동차 연비 데이터

> 1970~1982년 자동차 모델의 연비 및 성능 데이터. 회귀 분석과 다변량 EDA에 활용.


---


## 기본 정보

- **행 수**: 398개
- **열 수**: 9개
- **주요 용도**: 회귀 분석 (mpg 예측), 다중공선성 확인, 피처 중요도
- **시험 활용**: 빅분기 Type3 (회귀), ADP (회귀·전처리)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `mpg` | float64 | 연비 (miles per gallon) ← 타겟, 결측값 존재 |
| `cylinders` | int64 | 실린더 수 (3/4/5/6/8) |
| `displacement` | float64 | 배기량 (cubic inches) |
| `horsepower` | float64 | 마력, 결측값 존재 |
| `weight` | int64 | 차량 중량 (lbs) |
| `acceleration` | float64 | 가속 성능 (0→60mph 시간, 초) |
| `model_year` | int64 | 연식 (70~82, 1900년대) |
| `origin` | int64 | 생산 국가 (1=미국, 2=유럽, 3=일본) |
| `name` | object | 차량 모델명 |

## 샘플 데이터 (상위 3행)

| mpg | cylinders | displacement | horsepower | weight | acceleration | model_year | origin | name |
|---|---|---|---|---|---|---|---|---|
| 18.0 | 8 | 307.0 | 130.0 | 3504 | 12.0 | 70 | 1 | chevrolet chevelle malibu |
| 15.0 | 8 | 350.0 | 165.0 | 3693 | 11.5 | 70 | 1 | buick skylark 320 |
| 32.0 | 4 | 83.0 | 61.0 | 2003 | 19.0 | 74 | 3 | datsun 710 |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('mpg')

# 결측값 제거
df.dropna(inplace=True)

# 원산지별 평균 연비
print(df.groupby('origin')['mpg'].mean())

# 상관관계
print(df[['mpg','displacement','horsepower','weight']].corr())
```
