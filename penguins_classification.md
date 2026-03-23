# 🐧 Penguins 다중 클래스 분류 (Multi-class Classification)

> **팔머 펭귄 데이터셋(Palmer Penguins)** 을 이용한 지도학습 분류 분석  
> 데이터 출처: Palmer Station, Antarctica — Horst et al. (2020)

---

## 1. 문제 정의 (Problem Statement)

### 우리가 풀려는 것

> **질문:** 펭귄의 신체 측정값(부리 크기, 날개 길이, 몸무게 등)으로  
> **어떤 종(species)인지 자동으로 분류**할 수 있는가?

| 구분 | 내용 |
|------|------|
| **문제 유형** | 지도학습 (Supervised Learning) — 다중 클래스 분류 (Multi-class Classification) |
| **타겟 변수** | `species` — Adelie / Chinstrap / Gentoo (3종) |
| **입력 변수** | 부리 길이·깊이, 날개 길이, 몸무게, 성별, 서식 섬 (6개) |
| **평가 지표** | Accuracy, Precision, Recall, F1-score, Confusion Matrix |

### 실제 활용 맥락

- 야외 조사에서 펭귄을 직접 보지 않고 **센서 측정값만으로 종 판별**
- 생태학적 모니터링 자동화 파이프라인의 기초 모델
- 머신러닝 입문 예제로 **전처리 → 모델 → 평가**의 전 과정 학습

---

## 2. 데이터셋 탐색 (EDA)

### 2-1. 기본 정보

```
전체 행 수: 344
피처 수:   8 (species, island, bill_length_mm, bill_depth_mm,
               flipper_length_mm, body_mass_g, sex, year)
결측치:    수치형 2행 / sex 11행
분석 사용: 결측치 제거 후 333행
```

### 2-2. 컬럼 설명

| 컬럼명 | 타입 | 설명 | 단위 |
|--------|------|------|------|
| `species` | 범주 | **타겟: 펭귄 종** | Adelie / Chinstrap / Gentoo |
| `island` | 범주 | 서식지 섬 | Biscoe / Dream / Torgersen |
| `bill_length_mm` | 수치 | 부리 길이 (culmen length) | mm |
| `bill_depth_mm` | 수치 | 부리 깊이 (culmen depth) | mm |
| `flipper_length_mm` | 수치 | 날개(지느러미) 길이 | mm |
| `body_mass_g` | 수치 | 몸무게 | g |
| `sex` | 범주 | 성별 | male / female |
| `year` | 수치 | 조사 연도 (미사용) | 2007–2009 |

### 2-3. 타겟 분포

```
Adelie    : 152마리 (44.2%)  ← 다수 클래스
Gentoo    : 124마리 (36.0%)
Chinstrap :  68마리 (19.8%)  ← 소수 클래스
```

> ⚠️ **클래스 불균형**: Adelie가 Chinstrap의 2.2배. `stratify` 옵션으로 비율 유지 필요.

### 2-4. 수치형 피처 기초 통계

| 피처 | 평균 | 표준편차 | 최솟값 | 최댓값 |
|------|------|----------|--------|--------|
| bill_length_mm | 43.92 | 5.46 | 32.1 | 59.6 |
| bill_depth_mm | 17.15 | 1.97 | 13.1 | 21.5 |
| flipper_length_mm | 200.92 | 14.06 | 172 | 231 |
| body_mass_g | 4201.75 | 801.95 | 2700 | 6300 |

### 2-5. 종별 특성 요약

| 특성 | Adelie | Chinstrap | Gentoo |
|------|--------|-----------|--------|
| 서식 섬 | Biscoe·Dream·**Torgersen** | **Dream** | **Biscoe** |
| 부리 길이 | 중간 (~38.8mm) | **가장 김** (~48.8mm) | 중간 (~47.5mm) |
| 부리 깊이 | **가장 깊음** (~18.3mm) | 중간 (~18.4mm) | **가장 얕음** (~15.0mm) |
| 날개 길이 | 짧음 (~190mm) | 중간 (~196mm) | **가장 김** (~217mm) |
| 몸무게 | 작음 (~3701g) | 중간 (~3733g) | **가장 큼** (~5076g) |

---

## 3. 전처리 파이프라인 (Preprocessing)

```python
from palmerpenguins import load_penguins
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ① 데이터 로드
df = load_penguins()                  # 344행

# ② 결측치 제거 (listwise deletion)
df_clean = df.dropna()                # 333행

# ③ 범주형 인코딩 (Label Encoding)
le_sex    = LabelEncoder()
le_island = LabelEncoder()
df_clean['sex_enc']    = le_sex.fit_transform(df_clean['sex'])
df_clean['island_enc'] = le_island.fit_transform(df_clean['island'])
#   sex:    female=0, male=1
#   island: Biscoe=0, Dream=1, Torgersen=2

# ④ 피처 선택 (year 제외 — 학술 목적 변수)
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'sex_enc', 'island_enc']

X = df_clean[features]
y = LabelEncoder().fit_transform(df_clean['species'])
#   Adelie=0, Chinstrap=1, Gentoo=2

# ⑤ 학습/테스트 분리 (8:2, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Train: 266행  |  Test: 67행

# ⑥ 스케일링 (거리 기반 모델에만 적용)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

> **중요 포인트**  
> - 스케일링은 `fit`을 학습 데이터에만 적용하고, 테스트에는 `transform`만 적용 (데이터 누수 방지)  
> - `stratify=y` : 클래스 비율 유지  
> - Logistic Regression·SVM → 스케일링 필요 / Random Forest·GBM → 불필요

---

## 4. 모델링 (Modeling)

### 4-1. 사용 모델 4종

| 모델 | 특징 | 스케일링 |
|------|------|----------|
| **Logistic Regression** | 선형 결정 경계, 확률 출력, 해석 용이 | 필요 |
| **Random Forest** | 앙상블(배깅), 비선형, 과적합 강함 | 불필요 |
| **SVM (RBF kernel)** | 고차원 경계, 서포트 벡터 기반 | 필요 |
| **Gradient Boosting** | 순차 앙상블(부스팅), 높은 정확도 | 불필요 |

### 4-2. 전체 학습 코드

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'SVM (RBF)':           (SVC(kernel='rbf', random_state=42), True),
    'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, scaled) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='accuracy')
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    print(f"{name}: CV={cv_scores.mean():.4f}(±{cv_scores.std():.4f}), Test={accuracy_score(y_test,y_pred):.4f}")
```

---

## 5. 결과 (Results)

### 5-1. 모델 성능 비교

| 모델 | CV 평균 정확도 | CV 표준편차 | **테스트 정확도** |
|------|:---:|:---:|:---:|
| Logistic Regression | 99.25% | ±0.92% | **100.00%** |
| Random Forest | 98.87% | ±1.51% | **100.00%** |
| SVM (RBF) | 99.25% | ±0.92% | **100.00%** |
| Gradient Boosting | 98.50% | ±1.41% | 97.01% |

> 🏆 **Logistic Regression, Random Forest, SVM** 모두 테스트셋 완벽 분류 (67/67)

### 5-2. Confusion Matrix (Random Forest)

```
예측 →        Adelie  Chinstrap  Gentoo
실제 Adelie      29        0        0    ← 29/29 완벽
실제 Chinstrap    0       14        0    ← 14/14 완벽
실제 Gentoo       0        0       24    ← 24/24 완벽
```

**해석:**  
- 대각선 = 올바른 분류 / 비대각선 = 오분류  
- 모든 오분류 = 0 → **완벽한 분류**
- 특히 Chinstrap(소수 클래스)도 1개도 틀리지 않음

### 5-3. Classification Report (Random Forest)

| 클래스 | Precision | Recall | F1-score | Support |
|--------|:---------:|:------:|:--------:|:-------:|
| **Adelie** | 1.00 | 1.00 | 1.00 | 29 |
| **Chinstrap** | 1.00 | 1.00 | 1.00 | 14 |
| **Gentoo** | 1.00 | 1.00 | 1.00 | 24 |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **67** |

| 지표 | 공식 | 의미 |
|------|------|------|
| **Precision** | TP/(TP+FP) | 예측을 A종으로 했을 때 실제로 A종인 비율 |
| **Recall** | TP/(TP+FN) | 실제 A종 중 A종으로 올바르게 예측한 비율 |
| **F1-score** | 2×(P×R)/(P+R) | Precision과 Recall의 조화 평균 |

---

## 6. 피처 중요도 분석 (Feature Importance)

### Random Forest 기반 중요도

| 순위 | 피처 | 중요도 | 비율 |
|------|------|:------:|:----:|
| 🥇 1 | `bill_length_mm` (부리 길이) | 0.3913 | **39.1%** |
| 🥈 2 | `flipper_length_mm` (날개 길이) | 0.2193 | **21.9%** |
| 🥉 3 | `island_enc` (서식 섬) | 0.1393 | **13.9%** |
| 4 | `bill_depth_mm` (부리 깊이) | 0.1295 | 13.0% |
| 5 | `body_mass_g` (몸무게) | 0.1132 | 11.3% |
| 6 | `sex_enc` (성별) | 0.0073 | 0.7% |

### 해석

1. **부리 길이(39.1%)** 가 가장 중요  
   - Adelie(~38.8mm) vs Chinstrap(~48.8mm)을 가장 잘 구분
   
2. **날개 길이(21.9%)** 가 2위  
   - Gentoo가 ~217mm로 압도적으로 길어서 쉽게 분리됨

3. **서식 섬(13.9%)** 도 중요  
   - Torgersen 섬에는 Adelie만 서식 → 서식지 자체가 강력한 분류 단서

4. **성별(0.7%)** 은 미미  
   - 같은 종 내에서 성별 차이가 있지만, 종 구분에는 거의 기여 안 함

---

## 7. 로지스틱 회귀 계수 해석 (Coefficient Analysis)

| 클래스 | 가장 중요한 피처 (절대값 기준 Top 3) | 의미 |
|--------|--------------------------------------|------|
| **Adelie** | bill_length_mm(-2.63), bill_depth_mm(+0.85) | 부리가 짧고 깊을수록 Adelie |
| **Chinstrap** | bill_length_mm(+2.19), body_mass_g(-0.96) | 부리가 길고 가벼울수록 Chinstrap |
| **Gentoo** | bill_depth_mm(-1.33), flipper_length_mm(+1.05), body_mass_g(+1.00) | 날개 길고 무거울수록, 부리 얕을수록 Gentoo |

---

## 8. 종합 해석 및 인사이트

### 왜 이렇게 높은 정확도가 나왔는가?

1. **종마다 신체 특성이 뚜렷이 다름**  
   - Gentoo는 날개·몸무게로, Chinstrap은 부리 길이로 자연스럽게 분리됨

2. **서식 섬과 종의 연관성이 강함**  
   - Torgersen 섬: Adelie만 서식 → 섬만 알아도 바로 Adelie

3. **피처 수(6)에 비해 데이터 품질이 좋음**  
   - 다차원 공간에서 클래스가 선형적으로도 분리 가능 → LR도 100%

### 주의할 점 (현실 적용 시)

| 주의사항 | 내용 |
|----------|------|
| **과적합 가능성** | 67개 테스트셋은 작음 → 더 큰 독립 데이터로 검증 필요 |
| **클래스 불균형** | Chinstrap이 적어 실제 현장에서는 재현율 낮을 수 있음 |
| **결측치 처리** | 단순 제거가 아닌 KNN/평균 대체 전략 검토 필요 |
| **년도 변수** | `year`는 이 분석에서 제외했지만, 장기적 개체군 변화 연구에 유용 |

---

## 9. 전체 실행 코드 (Complete Code)

```python
# ============================================================
# 🐧 Palmer Penguins 다중 클래스 분류 - 완전 코드
# ============================================================

from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. 데이터 로드 & EDA
# ─────────────────────────────────────────────
df = load_penguins()
print(f"Shape: {df.shape}")
print(df['species'].value_counts())

# ─────────────────────────────────────────────
# 2. 전처리
# ─────────────────────────────────────────────
df_clean = df.dropna().copy()

le_sex    = LabelEncoder()
le_island = LabelEncoder()
df_clean['sex_enc']    = le_sex.fit_transform(df_clean['sex'])
df_clean['island_enc'] = le_island.fit_transform(df_clean['island'])

features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'sex_enc', 'island_enc']

X = df_clean[features]
le_y = LabelEncoder()
y = le_y.fit_transform(df_clean['species'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. 모델 학습 & 평가
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'SVM (RBF)':           (SVC(kernel='rbf', random_state=42), True),
    'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, scaled) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
    cv_sc = cross_val_score(model, Xtr, y_train, cv=cv, scoring='accuracy')
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    print(f"{name}: CV={cv_sc.mean():.4f}(±{cv_sc.std():.4f}), Test={accuracy_score(y_test, y_pred):.4f}")

# ─────────────────────────────────────────────
# 4. 최종 평가 (Random Forest)
# ─────────────────────────────────────────────
rf  = models['Random Forest'][0]
y_pred_rf = rf.predict(X_test)

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred_rf)
print(pd.DataFrame(cm, index=le_y.classes_, columns=le_y.classes_))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_rf, target_names=le_y.classes_))

# ─────────────────────────────────────────────
# 5. 피처 중요도 시각화
# ─────────────────────────────────────────────
fi_df = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(8, 4))
plt.barh(fi_df['feature'], fi_df['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Random Forest - Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 6. Confusion Matrix 시각화
# ─────────────────────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_y.classes_, yticklabels=le_y.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Random Forest)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
```

---

## 10. 요약

```
📌 문제: 펭귄의 신체 측정값 6개로 종(3종) 분류
📌 데이터: 333행 × 6 피처 (결측치 제거 후)
📌 최고 성능: Logistic Regression / Random Forest / SVM → 테스트 100%
📌 핵심 피처: 부리 길이(39.1%) > 날개 길이(21.9%) > 서식 섬(13.9%)
📌 교훈:
   ✅ 피처가 클래스를 자연스럽게 분리할 때, 단순 모델도 완벽에 가까운 성능
   ✅ 범주형 변수(섬)도 분류에 중요한 단서가 될 수 있음
   ✅ CV 표준편차로 모델 안정성까지 함께 확인해야 함
```
