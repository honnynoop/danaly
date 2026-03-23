# brain_networks — 뇌 네트워크 상관관계 데이터

> fMRI 기반 뇌 네트워크 간 기능적 연결성(상관관계) 행렬 데이터. 클러스터맵 시각화에 최적.


---


## 기본 정보

- **행 수**: 87개
- **열 수**: 87개
- **주요 용도**: 상관관계 히트맵, clustermap, 군집 분석, 네트워크 분석
- **시험 활용**: ADP (상관분석·클러스터링)

> ℹ️ ⚠️ 헤더가 2단계(MultiIndex)로 구성되어 있어 불러올 때 `header=[0,1]` 옵션 필요:

```python
df = sns.load_dataset('brain_networks', index_col=0, header=[0,1])
```


## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `network_N_region` | float64 | N번 네트워크의 region 간 상관계수 (87×87 행렬 구조) |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('brain_networks', index_col=0, header=[0,1])

# 상관행렬
corr = df.corr()
print(corr.shape)  # (87, 87)
```
