# planets — 외계행성 탐사 데이터

> NASA 외계행성 아카이브 기반의 발견된 외계행성 데이터. 시계열 + 다변량 EDA에 활용.


---


## 기본 정보

- **행 수**: 1,035개
- **열 수**: 6개
- **주요 용도**: 결측값 처리, 로그 변환 (skewed data), 그룹별 집계
- **시험 활용**: ADP (EDA, 이상값·결측값 처리)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `method` | object | 탐지 방법 (Radial Velocity, Transit 등) |
| `number` | int64 | 발견된 행성 수 |
| `orbital_period` | float64 | 공전 주기 (일), 결측값 존재 |
| `mass` | float64 | 질량 (목성 질량 단위), 결측값 존재 |
| `distance` | float64 | 거리 (파섹), 결측값 존재 |
| `year` | int64 | 발견 연도 |

## 샘플 데이터 (상위 3행)

| method | number | orbital_period | mass | distance | year |
|---|---|---|---|---|---|
| Radial Velocity | 1 | 269.3 | 7.1 | 77.4 | 2006 |
| Radial Velocity | 1 | 874.8 | 2.21 | 56.95 | 2008 |
| Transit | 1 | 2.47 | NaN | 130.0 | 2010 |

## 코드 예시

```python
import seaborn as sns
import numpy as np

df = sns.load_dataset('planets')

# 탐지 방법별 발견 행성 수
print(df.groupby('method')['number'].sum().sort_values(ascending=False))

# 로그 변환 (왜도 보정)
df['log_mass'] = np.log1p(df['mass'])
```
