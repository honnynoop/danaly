# fmri — 뇌 fMRI 신호 데이터

> fMRI 실험에서 측정된 뇌 BOLD 신호 데이터. 시계열·반복 측정 시각화에 활용.


---


## 기본 정보

- **행 수**: 1,064개
- **열 수**: 5개
- **주요 용도**: 선형 그래프, 반복 측정 시각화, groupby + 평균 시계열
- **시험 활용**: ADP (시계열·그룹 분석)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `subject` | object | 피험자 ID (s0~s13) |
| `timepoint` | int64 | 시간 포인트 (0~17) |
| `event` | object | 자극 유형 (stim / cue) |
| `region` | object | 뇌 영역 (frontal / parietal) |
| `signal` | float64 | BOLD 신호 값 |

## 샘플 데이터 (상위 3행)

| subject | timepoint | event | region | signal |
|---|---|---|---|---|
| s13 | 18 | stim | parietal | -0.0179 |
| s5 | 14 | stim | parietal | -0.0354 |
| s12 | 18 | stim | parietal | -0.0577 |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('fmri')

# 자극별·시간별 평균 신호
pivot = df.groupby(['timepoint','event'])['signal'].mean().unstack()
print(pivot)
```
