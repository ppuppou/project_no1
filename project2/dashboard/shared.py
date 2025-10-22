import pandas as pd
from pathlib import Path
import joblib
import numpy as np
try:
    # 권장 경로
    from hdbscan.prediction import approximate_predict
except Exception:
    import hdbscan
    approximate_predict = getattr(hdbscan, "approximate_predict")

try:
    from scipy.stats import chi2
    _HAS_CHI2 = True
except Exception:
    chi2 = None
    _HAS_CHI2 = False

# ================================
# 🔴 이상치 판별 임계값(요구사항)
# ================================
ANOMALY_PROBA_THRESHOLD = 0.9  # anom_proba >= 0.9 이면 이상치로 판정

# ================================
# 📁 데이터 로딩 (인코딩 안전)
# ================================
app_dir = Path(__file__).parent

def load_csv_robustly(file_path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("read", b"", 0, 1, f"Cannot open {file_path.name} with utf-8/cp949")

try:
    train_df = load_csv_robustly(app_dir / "data/train_df.csv")
    test_df = load_csv_robustly(app_dir / "data/test.csv")
    test_label_df = load_csv_robustly(app_dir / "data/test_label.csv")
    print("✅ 데이터 로드 완료")
except Exception as e:
    print(f"⚠️ 데이터 로드 실패: {e}")
    train_df, test_df, test_label_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ================================
# 📁 불량(이진) 분류 모델 로딩
# ================================
try:
    model_dict = joblib.load(app_dir / "data/final_model.pkl")
    defect_model = model_dict["model"]
    feature_cols = model_dict["feature_names"]
    defect_threshold = 0.8346291545170944
    print("✅ 불량탐지 모델 로드 완료")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    defect_model, feature_cols, defect_threshold = None, [], 0.5

# ================================
# ✨ 데이터 전처리(필요 필드만)
# ================================
if "Unnamed: 0" in train_df.columns:
    train_df = train_df.drop(columns=["Unnamed: 0"])

if "registration_time" in test_df.columns:
    try:
        test_df["registration_time"] = pd.to_datetime(test_df["registration_time"])
        test_df["hour"] = test_df["registration_time"].dt.hour
    except Exception as e:
        print(f"⚠️ 'registration_time' 처리 경고: {e}")

# 실시간 스트리밍 원천
streaming_df = test_df.copy()

# ================================
# ⏱️ RealTimeStreamer
# ================================
class RealTimeStreamer:
    def __init__(self, data: pd.DataFrame):
        self.full_data = data.reset_index(drop=True).copy()
        self.current_index = 0

    def get_next_batch(self, batch_size: int = 1):
        if self.current_index >= len(self.full_data):
            return None
        end = min(self.current_index + batch_size, len(self.full_data))
        batch = self.full_data.iloc[self.current_index:end].copy()
        self.current_index = end
        return batch

    def get_current_data(self):
        if self.current_index == 0:
            return pd.DataFrame()
        return self.full_data.iloc[: self.current_index].copy()

    def reset_stream(self):
        self.current_index = 0

    def progress(self) -> float:
        return 0.0 if len(self.full_data) == 0 else (self.current_index / len(self.full_data)) * 100

# ================================
# 📁 HDBSCAN Router 로드
# ================================
try:
    # 구조: {'segment_col','preprocessor','global', 'segments'...}
    hdb_router = joblib.load(app_dir / "data/hdbscan_router.pkl")
    hdb_seg_col = hdb_router.get("segment_col", "mold_code")
    hdb_prep = hdb_router.get("preprocessor", None)
    hdb_global = hdb_router.get("global", {})
    hdb_segments = hdb_router.get("segments", {})
    print("✅ HDBSCAN Router 로드 완료")
except Exception as e:
    print(f"⚠️ HDBSCAN Router 로드 실패: {e}")
    hdb_router, hdb_seg_col, hdb_prep, hdb_global, hdb_segments = None, None, None, None, {}

ANOM_BUNDLE         = hdb_router
ANOM_PREPROCESSOR   = hdb_prep
ANOM_FEATURE_ORDER  = (hdb_global or {}).get("feature_order", [])

# 방어: feature_order가 비어있으면 (거의 없겠지만) 학습에 쓰인 원본 칼럼으로 대체
# -> 가장 안전한 건 train_df 컬럼과 교집합을 쓰는 것
if not ANOM_FEATURE_ORDER:
    try:
        # train_df 가 shared.py 안에서 이미 로드되어 있다면:
        ANOM_FEATURE_ORDER = [c for c in train_df.columns]
    except Exception:
        pass  # 최소한 빈 리스트가 아니게만 보장하면 됨

def anomaly_transform(df_raw: pd.DataFrame) -> np.ndarray:
    """
    모델과 '동일' 전처리(ColumnTransformer)를 적용해서
    raw DataFrame -> 전처리 후 numpy array 로 변환.
    - 결측치는 KNNImputer/Most Frequent 등, 학습 때와 같은 규칙으로 처리
    - 범주는 OrdinalEncoder(unknown_value=-1) 규칙으로 처리
    - 칼럼 순서는 hdb_global['feature_order']를 따른다
    """
    if ANOM_PREPROCESSOR is None:
        raise RuntimeError("ANOM_PREPROCESSOR가 로드되지 않았습니다.")

    if not ANOM_FEATURE_ORDER:
        raise RuntimeError("ANOM_FEATURE_ORDER(=feature_order)가 비어있습니다.")

    # 누락 칼럼 자동 생성(=NaN) + 순서 맞추기
    X = df_raw.reindex(columns=ANOM_FEATURE_ORDER)

    # ColumnTransformer가 알아서 숫자/범주 파이프라인을 적용하고 결측도 처리
    Z = ANOM_PREPROCESSOR.transform(X)
    return Z

def get_preproc_feature_names_out() -> list:
    """
    전처리 결과의 칼럼명(스케일/인코딩 후 피처명)을 반환.
    verbose_feature_names_out=False 로 저장되었다면 원래 이름과 1:1 매핑됨.
    """
    try:
        return list(ANOM_PREPROCESSOR.get_feature_names_out(ANOM_FEATURE_ORDER))
    except Exception:
        # 버전에 따라 없을 수 있으니 안전판
        Z0 = anomaly_transform(pd.DataFrame([ {c: np.nan for c in ANOM_FEATURE_ORDER} ]))
        return [f"feat_{i}" for i in range(Z0.shape[1])]

# (선택) 세그먼트 컬럼을 앱에서 쓰고 싶으면 같이 export
ANOM_SEGMENT_COL = hdb_seg_col
HDB_SEGMENTS     = hdb_segments


# ================================================================
# 🔴 [수정됨] 📊 참조 분포(점수/거리) 준비 (미리 계산된 파일 로드)
# ================================================================
#
# 기존의 무거운 계산 로직(train_df.copy(), transform, predict 등)을
# 모두 제거하고, 'create_reference_stats.py'로 미리 생성한
# 'hdbscan_reference_stats.pkl' 파일을 로드합니다.
#

def _get_feature_order(preprocessor, bundle) -> list:
    """전처리기에서 피처 순서 추출 (이 함수는 predict_anomaly에서 필요하므로 남겨둡니다)"""
    try:
        if "feature_order" in bundle:
            return list(bundle["feature_order"])
        num_cols, cat_cols = [], []
        for name, _, cols in getattr(preprocessor, "transformers_", []):
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
        return num_cols + cat_cols
    except Exception:
        return []

try:
    # 1단계에서 생성한 .pkl 파일을 로드합니다.
    stats_path = app_dir / "data/hdbscan_reference_stats.pkl"
    stats_bundle = joblib.load(stats_path)
    
    _ANOM_SCORE_REF = stats_bundle.get("ANOM_SCORE_REF")
    _RD2_REF = stats_bundle.get("RD2_REF")
    _N_NUM_COLS = stats_bundle.get("N_NUM_COLS", 0)
    
    if _RD2_REF is not None:
        print(f"✅ RD2 참조 로드 완료 (n={len(_RD2_REF)})")
    if _ANOM_SCORE_REF is not None:
        print(f"✅ Score 참조 로드 완료 (n={len(_ANOM_SCORE_REF)})")

except Exception as e:
    print(f"⚠️ 'hdbscan_reference_stats.pkl' 로드 실패: {e}")
    print("   -> 🔴 중요: 배포 전, 로컬에서 'create_reference_stats.py' 를 먼저 실행해야 합니다.")
    _ANOM_SCORE_REF, _RD2_REF, _N_NUM_COLS = None, None, 0
    
# ================================================================
# 🔴 [수정 완료]
# ================================================================

# ================================
# 🔎 실시간 이상치 탐지 (임계값 0.9 사용)
# ================================
def predict_anomaly(df: pd.DataFrame) -> pd.DataFrame | None:
    """HDBSCAN Router 기반 이상치 탐지.
    반환: id, anomaly_status(0/1), prob(strength), anom_proba
    """
    if hdb_router is None or hdb_prep is None or df.empty:
        return None

    df_in = df.copy().reset_index(drop=False)  # 원 인덱스 보존
    idx_col = df_in.columns[0]

    # 세그먼트 키
    if hdb_seg_col and (hdb_seg_col in df_in.columns):
        seg_series = df_in[hdb_seg_col].astype("object")
        seg_series = seg_series.mask(pd.isna(seg_series), "NA")
        seg_keys = seg_series.apply(lambda x: f"{hdb_seg_col}={x}")
    else:
        seg_keys = pd.Series(["__global__"] * len(df_in))

    out_anom = np.zeros(len(df_in), dtype=int)
    out_prob = np.zeros(len(df_in), dtype=float)        # membership strength
    out_anom_proba = np.zeros(len(df_in), dtype=float)  # 우리가 그릴 값

    for key, idx in seg_keys.groupby(seg_keys).groups.items():
        bundle = hdb_segments.get(key, None) or hdb_global
        feat_order = _get_feature_order(hdb_prep, bundle)
        if not feat_order:
            continue

        Xg = df_in.loc[idx].copy()
        for c in feat_order:
            if c not in Xg.columns:
                Xg[c] = 0
        Xg = Xg[feat_order]

        try:
            Xp = hdb_prep.transform(Xg)
            pca = bundle.get("pca", None)
            Z = pca.transform(Xp) if pca is not None else Xp

            labels, strengths = approximate_predict(bundle["hdb"], Z)
            strengths = np.asarray(strengths).astype(float).reshape(-1)
            anom_scores = np.where(np.asarray(labels) == -1, 1.0, 1.0 - strengths)

            # anom_proba 계산: 참조 점수분포 기준의 누적백분위(없으면 min-max)
            if _ANOM_SCORE_REF is not None and len(_ANOM_SCORE_REF) > 0:
                pos = np.searchsorted(_ANOM_SCORE_REF, anom_scores, side="right")
                anom_proba = pos / len(_ANOM_SCORE_REF)
            else:
                s_min, s_ptp = np.min(anom_scores), np.ptp(anom_scores)
                anom_proba = (anom_scores - s_min) / (s_ptp + 1e-9) if s_ptp > 0 else np.zeros_like(anom_scores)

        except Exception as e:
            # Fallback: 거리 기반
            print(f"ℹ️ Fallback(seg={key}): {e}")
            strengths = np.zeros(len(idx), dtype=float)
            if hasattr(hdb_prep, "transformers_") and any(t[0] == "num" for t in hdb_prep.transformers_):
                rd2 = np.sum(np.square(Xp[:, :_N_NUM_COLS]), axis=1) if _N_NUM_COLS > 0 else np.zeros(len(idx))
                if _RD2_REF is not None and len(_RD2_REF) > 0:
                    pos = np.searchsorted(_RD2_REF, rd2, side="right")
                    anom_proba = pos / len(_RD2_REF)
                elif _HAS_CHI2 and _N_NUM_COLS > 0:
                    anom_proba = chi2.cdf(rd2, df=_N_NUM_COLS)
                else:
                    anom_proba = (rd2 - rd2.min()) / (rd2.ptp() + 1e-9) if rd2.ptp() > 0 else np.zeros_like(rd2)
            else:
                anom_proba = np.zeros(len(idx))

        # 🔴 임계값 적용 (여기가 핵심)
        preds = (anom_proba >= ANOMALY_PROBA_THRESHOLD).astype(int)

        out_anom[list(idx)] = preds
        out_prob[list(idx)] = strengths
        out_anom_proba[list(idx)] = anom_proba

    return pd.DataFrame({
        "id": df_in[idx_col].values,
        "anomaly_status": out_anom,
        "prob": out_prob,           # membership strength (정상일수록 큼)
        "anom_proba": out_anom_proba
    })
