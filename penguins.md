# penguins — 펭귄 종 분류 데이터

> 남극 팔머 군도의 3종 펭귄 형태 측정 데이터. iris의 현대적 대안으로 자주 사용됨.


---


## 기본 정보

- **행 수**: 344개
- **열 수**: 7개
- **주요 용도**: 다중 분류, 결측값 처리, 클러스터링, EDA
- **시험 활용**: 빅분기 Type2/3 (분류), ADP (전처리·시각화)

## 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `species` | object | 펭귄 종류 ← 타겟 (Adelie / Chinstrap / Gentoo) |
| `island` | object | 서식 섬 (Torgersen / Biscoe / Dream) |
| `bill_length_mm` | float64 | 부리 길이 (mm), 결측값 존재 |
| `bill_depth_mm` | float64 | 부리 깊이 (mm) |
| `flipper_length_mm` | float64 | 지느러미 길이 (mm) |
| `body_mass_g` | float64 | 체중 (g) |
| `sex` | object | 성별 (Male / Female), 결측값 존재 |

## 샘플 데이터 (상위 3행)

| species | island | bill_length_mm | bill_depth_mm | flipper_length_mm | body_mass_g | sex |
|---|---|---|---|---|---|---|
| Adelie | Torgersen | 39.1 | 18.7 | 181.0 | 3750.0 | Male |
| Adelie | Torgersen | 39.5 | 17.4 | 186.0 | 3800.0 | Female |
| Gentoo | Biscoe | 46.1 | 13.2 | 211.0 | 4500.0 | Female |

## 코드 예시

```python
import seaborn as sns

df = sns.load_dataset('penguins')

# 결측값 제거
df.dropna(inplace=True)

# 종별 평균 체중
print(df.groupby('species')['body_mass_g'].mean())

# 성별 인코딩
df['sex_enc'] = (df['sex'] == 'Male').astype(int)
```
