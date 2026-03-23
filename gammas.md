# gammas — 감마 신호 데이터

> 뇌 감마 대역 BOLD 신호 실험 데이터. 시계열 + 다중 ROI 비교 시각화에 활용.


---


## 기본 정보

- **행 수**: 1,700개
- **열 수**: 3개
- **주요 용도**: 다중 시계열 시각화, lineplot, 그룹별 신호 비교
- **시험 활용**: ADP (시계열·시각화)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `timepoint` | float64 | 시간 포인트 |
| `ROI` | object | 관심 영역 (뇌 부위 코드) |
| `BOLD signal` | float64 | 감마 대역 BOLD 신호 |

## 샘플 데이터 (상위 3행)

| timepoint | ROI | BOLD signal |
|---|---|---|
| 0.0 | IPS | 0.0 |
| 0.1 | IPS | 0.002 |
| 0.0 | AG | 0.0 |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('gammas')

# ROI별 평균 신호
print(df.groupby('ROI')['BOLD signal'].mean())

# 시간별 전체 평균 신호
print(df.groupby('timepoint')['BOLD signal'].mean().head(10))
```
