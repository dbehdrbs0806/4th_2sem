# =========================================================
# Titanic: EDA급 전처리 + 피처공학 + 분류/회귀 파이프라인
# =========================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
import statsmodels.api as sm

# ---------------------------------------------------------
# 0) 데이터 로드
# ---------------------------------------------------------
PATH = "/mnt/data/titanic.csv"  # 업로드한 파일 경로
df_raw = pd.read_csv(PATH)

# 컬럼 표준화(대/소문자 차이 호환)
df_raw.columns = [c.strip().lower() for c in df_raw.columns]
df = df_raw.copy()

# Survived 컬럼명 통일
if "survived" not in df.columns and "survived" in [c.lower() for c in df_raw.columns]:
    pass  # 이미 survived
elif "survived" not in df.columns and "Survived" in df_raw.columns:
    df["survived"] = df_raw["Survived"]

# ---------------------------------------------------------
# 1) 결측치 개요 확인 + 기본 처리
#    - 숫자: 중앙값 대치
#    - 범주: 최빈값 대치
#    - Age: (sex, pclass) 그룹 중앙값으로 정교 대치 (groupby/transform)
#    - Cabin: 결측 많음 → 'Unknown'으로 채우고, deck(첫 문자) 파생
# ---------------------------------------------------------
# 결측치 개수 확인
null_counts = df.isnull().sum()

# 숫자/범주 분리(데이터셋마다 약간 다를 수 있어 안전하게 추출)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# cabin 우선 문자열 처리
if "cabin" in df.columns:
    df["cabin"] = df["cabin"].fillna("Unknown").astype(str)
    # deck: 첫 문자 추출(숫자/기타는 'U'로)
    df["deck"] = df["cabin"].str[0].where(df["cabin"].str[0].str.isalpha(), "U")
    cat_cols = list(set(cat_cols + ["deck"]))

# Age 결측: sex+pclass 그룹 중앙값으로
for c in ["sex", "pclass"]:
    if c not in df.columns:
        # 캐글 원본 기준 컬럼이 있다고 가정하지만, 방어 코드
        pass

if "age" in df.columns:
    # 그룹 중앙값 계산 (groupby + transform)
    age_group_median = df.groupby(["sex", "pclass"])["age"].transform("median")
    df["age"] = df["age"].fillna(age_group_median)
    # 여전히 남으면 전체 중앙값
    df["age"] = df["age"].fillna(df["age"].median())

# 나머지 숫자 결측 중앙값 대치
for c in num_cols:
    if c == "survived":
        continue
    df[c] = df[c].fillna(df[c].median())

# 범주형 결측 최빈값 대치
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode(dropna=True).iloc[0]) if df[c].isnull().any() else df[c]

# ---------------------------------------------------------
# 2) 데이터 변환/그룹핑/인덱싱/부울 컬럼 생성 예시
#    - 그룹핑 통계: pclass, sex별 평균 생존률, 평균 요금 등
#    - family_size, is_alone 파생
#    - high_fare(상위 75% 이상) 부울
#    - child(<=12세), young_adult(13~29), senior(>=60) 부울
#    - title(Name에서 호칭 추출) → 희소 클래스 묶기
# ---------------------------------------------------------
# (1) 그룹핑 예시
grp = df.groupby(["pclass", "sex"]).agg(
    survived_rate=("survived", "mean") if "survived" in df.columns else ("fare", "mean"),
    fare_mean=("fare", "mean") if "fare" in df.columns else ("age", "mean"),
    count=("sex", "count")
).reset_index()

# (2) 파생
if set(["sibsp", "parch"]).issubset(df.columns):
    df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
else:
    df["family_size"] = 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

# (3) 요금 기반 부울
if "fare" in df.columns:
    thr = df["fare"].quantile(0.75)
    df["high_fare"] = (df["fare"] >= thr).astype(int)

# (4) 나이 구간 부울
if "age" in df.columns:
    df["child"] = (df["age"] <= 12).astype(int)
    df["young_adult"] = ((df["age"] > 12) & (df["age"] < 30)).astype(int)
    df["senior"] = (df["age"] >= 60).astype(int)

# (5) 타이틀 추출 (이름이 있으면)
if "name" in df.columns:
    df["title"] = (
        df["name"].str.extract(r",\s*([^\.]+)\.", expand=False)
        .str.strip()
        .replace({
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Royalty", "Countess": "Royalty", "Dona": "Royalty",
            "Sir": "Royalty", "Jonkheer": "Royalty", "Don": "Royalty",
            "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer", "Rev": "Officer"
        })
    )
    cat_cols = list(set(cat_cols + ["title"]))

# ---------------------------------------------------------
# 3) 열 선택 & 불필요 열 드랍(drop)
# ---------------------------------------------------------
drop_cols = []
drop_cols += ["name", "ticket"] if "ticket" in df.columns else ["name"] if "name" in df.columns else []
drop_cols += ["cabin"] if "cabin" in df.columns else []
# id 비슷한 컬럼 존재시 제거
for c in ["passengerid", "passenger_id", "id"]:
    if c in df.columns:
        drop_cols.append(c)

use_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# ---------------------------------------------------------
# 4) 원-핫 인코딩 (drop_first=True로 다중공선성 최소화)
# ---------------------------------------------------------
# 실제 범주 컬럼 재확인(숫자지만 범주로 쓸 수 있는 pclass도 포함)
cat_candidates = set(use_df.select_dtypes(exclude=[np.number]).columns.tolist())
if "pclass" in use_df.columns:
    cat_candidates.add("pclass")
if "sex" in use_df.columns:
    cat_candidates.add("sex")
if "embarked" in use_df.columns:
    cat_candidates.add("embarked")
if "deck" in use_df.columns:
    cat_candidates.add("deck")
if "title" in use_df.columns:
    cat_candidates.add("title")

cat_cols_final = sorted(list(cat_candidates))
df_oh = pd.get_dummies(use_df, columns=cat_cols_final, drop_first=True)

# ---------------------------------------------------------
# 5-A) [분류] Survived 예측 (로지스틱)
#      - X, y 분리 → train/test → 스케일링(수치만) → 학습/성능
# ---------------------------------------------------------
if "survived" in df_oh.columns:
    y_cls = df_oh["survived"].astype(int)
    X_cls = df_oh.drop(columns=["survived"])

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # 스케일링: 수치형 컬럼만
    num_cols_cls = X_train_c.select_dtypes(include=[np.number]).columns.tolist()
    scaler_c = StandardScaler()
    X_train_c[num_cols_cls] = scaler_c.fit_transform(X_train_c[num_cols_cls])
    X_test_c[num_cols_cls]  = scaler_c.transform(X_test_c[num_cols_cls])

    # 모델 선택: LogisticRegression(L2)
    logit = LogisticRegression(max_iter=2000, n_jobs=None)
    logit.fit(X_train_c, y_train_c)

    # 예측/성능
    probs = logit.predict_proba(X_test_c)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_c, y_pred).ravel()
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)      # Sensitivity/TPR
    specificity = tn / (tn + fp + 1e-12)    # TNR
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    auc = roc_auc_score(y_test_c, probs)

    print("\n[Classification @thr=0.5]")
    print(f"Accuracy   : {accuracy:.3f}")
    print(f"Precision  : {precision:.3f}")
    print(f"Recall     : {recall:.3f} (Sensitivity/TPR)")
    print(f"Specificity: {specificity:.3f} (TNR)")
    print(f"F1         : {f1:.3f}")
    print(f"AUC        : {auc:.3f}")
    print("\n[Classification Report]")
    print(classification_report(y_test_c, y_pred, digits=3))

# ---------------------------------------------------------
# 5-B) [회귀] log(Fare+1) 예측 (OLS/Ridge/Lasso/EN 비교)
#      - X, y 분리 → train/test → 스케일링(수치만) → 학습/성능
# ---------------------------------------------------------
if "fare" in df_oh.columns:
    y_reg = np.log1p(df_oh["fare"].astype(float))  # 연속형 타깃
    X_reg = df_oh.drop(columns=["fare"])

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # 스케일링: 수치형만
    num_cols_reg = X_train_r.select_dtypes(include=[np.number]).columns.tolist()
    scaler_r = StandardScaler()
    X_train_r[num_cols_reg] = scaler_r.fit_transform(X_train_r[num_cols_reg])
    X_test_r[num_cols_reg]  = scaler_r.transform(X_test_r[num_cols_reg])

    # ----- OLS (statsmodels 요약표)
    X_train_sm = sm.add_constant(X_train_r)
    ols_sm = sm.OLS(y_train_r, X_train_sm).fit()
    print("\n[Statsmodels OLS Summary @ log(Fare+1)]")
    print(ols_sm.summary())

    # ----- Sklearn 선형/규제 모델 비교
    models = {
        "Linear": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Lasso(alpha=0.01)": Lasso(alpha=0.01, random_state=42, max_iter=10000),
        "ElasticNet(a=0.01,l1_ratio=0.5)": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
    }

    print("\n[Regression Metrics @ log(Fare+1)]")
    for name, mdl in models.items():
        mdl.fit(X_train_r, y_train_r)
        pred = mdl.predict(X_test_r)
        rmse = mean_squared_error(y_test_r, pred, squared=False)
        r2   = r2_score(y_test_r, pred)
        print(f"{name:26s}  RMSE: {rmse:.4f}   R²: {r2:.4f}")

# ---------------------------------------------------------
# 6) 임계값 변화 예시(분류) - 필요 시 주석 해제
# ---------------------------------------------------------
# if "survived" in df_oh.columns:
#     for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
#         y_pred_t = (probs >= thr).astype(int)
#         tn, fp, fn, tp = confusion_matrix(y_test_c, y_pred_t).ravel()
#         spec = tn / (tn + fp + 1e-12)
#         rec  = tp / (tp + fn + 1e-12)
#         from sklearn.metrics import precision_score
#         prec = precision_score(y_test_c, y_pred_t, zero_division=0)
#         fpr  = fp / (fp + tn + 1e-12)
#         tpr  = rec
#         print(f"[thr={thr}] Precision={prec:.3f}, Recall={rec:.3f}, Specificity={spec:.3f}, tpr={tpr:.3f}, fpr={fpr:.3f}")
