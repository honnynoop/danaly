# iris — 붓꽃(아이리스) 분류 데이터

> Fisher(1936)가 발표한 3종 붓꽃의 꽃잎·꽃받침 크기 데이터. 분류 알고리즘 기초 실습의 대표 데이터셋.


---


## 기본 정보

- **행 수**: 150개
- **열 수**: 5개
- **주요 용도**: 다중 분류, 클러스터링(K-Means), PCA, 산점도 행렬(pairplot)
- **시험 활용**: 빅분기 Type2/3 (분류·클러스터링), ADP (PCA·시각화)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `sepal_length` | float64 | 꽃받침 길이 (cm) |
| `sepal_width` | float64 | 꽃받침 너비 (cm) |
| `petal_length` | float64 | 꽃잎 길이 (cm) |
| `petal_width` | float64 | 꽃잎 너비 (cm) |
| `species` | object | 붓꽃 종류 ← 타겟 (setosa / versicolor / virginica) |

## 샘플 데이터 (상위 3행)

| sepal_length | sepal_width | petal_length | petal_width | species |
|---|---|---|---|---|
| 5.1 | 3.5 | 1.4 | 0.2 | setosa |
| 4.9 | 3.0 | 1.4 | 0.2 | setosa |
| 7.0 | 3.2 | 4.7 | 1.4 | versicolor |

## 코드 예시

```python
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = sns.load_dataset('iris')

# 종별 기초 통계
print(df.groupby('species').mean())

# 타겟 인코딩
le = LabelEncoder()
df['target'] = le.fit_transform(df['species'])
```
