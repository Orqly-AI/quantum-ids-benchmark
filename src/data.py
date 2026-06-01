"""Multi-dataset loading & preprocessing for quantum/classical intrusion detection.

Single entry point for the quantum pipeline::

    from data import load
    Xtr, ytr, Xte, yte = load("nslkdd", n_features=8, reduction="pca", binary=True)

Datasets: NSL-KDD, UNSW-NB15, CICIDS2017. Each is cleaned (NaN/inf handling,
leaky-column removal, categorical encoding), scaled, and reduced to `n_features`
dimensions so the feature count maps onto the qubit count of the quantum models.

Dimensionality reduction (`reduction`):
  * "pca"         - linear PCA (fit on train only).
  * "autoencoder" - a small classical MLP autoencoder bottleneck (PyTorch),
                    fit on train only; falls back to PCA if torch is missing.
  * "none"        - keep the (encoded/scaled) feature matrix as-is.

IMPORTANT (variance note): one-hot expansion produces many sparse binary columns.
StandardScaling those before PCA gives every sparse dummy equal variance, so PCA
wastes its budget (8 PCs explained only ~27% var on NSL-KDD). Scaling to [0, 1]
with MinMax *before* PCA preserves the dominant continuous structure and lifts 8-PC
explained variance to ~84% on NSL-KDD. The unified load() therefore MinMax-scales
to [0,1] pre-reduction. (The legacy load_nslkdd keeps its original StandardScaler
path for backward compat; prefer load("nslkdd", ...) for the better variance.)

Backward-compat / classical-baseline helpers are preserved:
  * load_nslkdd(...)       - original reduced NSL-KDD loader (unchanged behaviour).
  * load_nslkdd_full(...)  - full-dim NSL-KDD for classical baselines.
  * _preprocess_raw(...)   - shared raw NSL-KDD one-hot/align helper.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 41 NSL-KDD features + label + difficulty (standard column order)
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS, header=None)
    df = df.drop(columns=["difficulty"])
    return df


def _preprocess_raw(binary: bool):
    """Load raw NSL-KDD, build binary labels, one-hot encode + align train/test.

    Returns (Xtr_full, y_train, Xte_full, y_test) where the X matrices are the
    122-dim aligned one-hot feature space (no scaling, no reduction yet).
    Uses the official KDDTrain+/KDDTest+ split — this split is the entire point
    of NSL-KDD: the test set contains novel attack families absent from train,
    so honouring it is what makes the benchmark measure generalisation rather
    than memorisation.
    """
    train = _load_raw(os.path.join(DATA_DIR, "KDDTrain+.txt"))
    test = _load_raw(os.path.join(DATA_DIR, "KDDTest+.txt"))

    def make_y(df):
        if binary:
            return (df["label"] != "normal").astype(int).values
        return df["label"].values

    y_train, y_test = make_y(train), make_y(test)
    train, test = train.drop(columns=["label"]), test.drop(columns=["label"])

    # One-hot encode categoricals, align train/test columns (test categories
    # unseen in train -> dropped; train categories absent in test -> 0-filled).
    train = pd.get_dummies(train, columns=CATEGORICAL)
    test = pd.get_dummies(test, columns=CATEGORICAL)
    train, test = train.align(test, join="left", axis=1, fill_value=0)
    test = test[train.columns]  # identical order
    Xtr = train.values.astype(np.float32)
    Xte = test.values.astype(np.float32)
    return Xtr, y_train.astype(np.int64), Xte, y_test.astype(np.int64)


def load_nslkdd(n_features: int = 8, binary: bool = True, scale: str = "minmax",
                seed: int = 42):
    """Return (X_train, y_train, X_test, y_test) reduced to `n_features` dims.

    binary=True -> normal (0) vs attack (1). Categorical cols one-hot encoded with
    the train/test columns aligned. PCA fit on train only (no leakage).
    `scale`: 'minmax' (good for angle encoding, range [0, pi]) or 'standard'.

    NOTE: kept for backward compat. Uses StandardScaler before PCA (the original
    behaviour, ~27% explained var at 8 dims). For better variance retention use
    load("nslkdd", ...), which MinMax-scales pre-PCA (~84% at 8 dims).
    """
    Xtr, y_train, Xte, y_test = _preprocess_raw(binary)

    # Scale BEFORE PCA (PCA is variance-sensitive); fit on train only
    std = StandardScaler().fit(Xtr)
    Xtr, Xte = std.transform(Xtr), std.transform(Xte)

    # Dimensionality reduction to qubit count
    pca = PCA(n_components=n_features, random_state=seed).fit(Xtr)
    Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    evr = float(pca.explained_variance_ratio_.sum())

    # Final scaling for the models / quantum encoding
    if scale == "minmax":
        mm = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr)  # angle encoding range
    else:
        mm = StandardScaler().fit(Xtr)
    Xtr, Xte = mm.transform(Xtr), mm.transform(Xte)

    print(f"[data] NSL-KDD | features={n_features} | PCA explained var={evr:.3f} "
          f"| train={Xtr.shape} test={Xte.shape} | attack%% train={y_train.mean():.3f}")
    return (Xtr.astype(np.float32), y_train,
            Xte.astype(np.float32), y_test)


def load_nslkdd_full(binary: bool = True, scale: str = "standard", seed: int = 42):
    """Full-dimensional NSL-KDD (122 aligned one-hot dims), scaled, no PCA.

    This is the view the *classical* baselines (RF / XGBoost / SVM / MLP / CNN)
    should be evaluated on: reducing to a handful of PCA components purely to
    match the qubit budget would artificially handicap the classical models and
    would (rightly) be flagged by a reviewer as an unfair comparison. The
    quantum models use the reduced `load_nslkdd` view; the classical baselines
    are reported on BOTH views so the paper can state (a) best-achievable
    classical performance and (b) a like-for-like same-feature-budget comparison.

    `scale`: 'standard' (z-score, default for tree/SVM/NN) or 'minmax' ([0,1]).
    Scaler is fit on train only (no leakage).
    """
    Xtr, y_train, Xte, y_test = _preprocess_raw(binary)
    if scale == "minmax":
        sc = MinMaxScaler().fit(Xtr)
    else:
        sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    print(f"[data] NSL-KDD FULL | dims={Xtr.shape[1]} | train={Xtr.shape} "
          f"test={Xte.shape} | attack%% train={y_train.mean():.3f} "
          f"test={y_test.mean():.3f}")
    return (Xtr.astype(np.float32), y_train,
            Xte.astype(np.float32), y_test)


# --------------------------------------------------------------------------- #
# Per-dataset preprocessing -> aligned numeric DataFrames + integer labels    #
# Each returns (Xtr_df, ytr, Xte_df, yte, class_names) BEFORE scaling/reduction#
# --------------------------------------------------------------------------- #
def _prep_nslkdd(binary: bool, seed: int):
    """NSL-KDD via the canonical train/test split (reuses _preprocess_raw)."""
    train = _load_raw(os.path.join(DATA_DIR, "KDDTrain+.txt"))
    test = _load_raw(os.path.join(DATA_DIR, "KDDTest+.txt"))

    ytr_raw = train["label"].values
    yte_raw = test["label"].values
    train = train.drop(columns=["label"])
    test = test.drop(columns=["label"])

    train = pd.get_dummies(train, columns=CATEGORICAL)
    test = pd.get_dummies(test, columns=CATEGORICAL)
    train, test = train.align(test, join="left", axis=1, fill_value=0)
    test = test[train.columns]

    ytr, yte, class_names = _encode_labels(ytr_raw, yte_raw, binary,
                                           normal_label="normal")
    return train, ytr, test, yte, class_names


# --- UNSW-NB15 -------------------------------------------------------------- #
UNSW_CATEGORICAL = ["proto", "service", "state"]
# 'id' is a row index -> leakage if kept. Labels derived from attack_cat/label.
UNSW_DROP = ["id"]


def _prep_unsw(binary: bool, seed: int):
    tr_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
    te_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")
    if not (os.path.exists(tr_path) and os.path.exists(te_path)):
        raise FileNotFoundError(
            "UNSW-NB15 CSVs not found. Run: "
            "wsl -d Ubuntu-22.04 -- bash -lc 'bash ~/download_unsw.sh' "
            "(script at src/download_unsw.sh)")
    # utf-8-sig strips the BOM some redistributions prepend to the 'id' header,
    # otherwise the first column is '﻿id' and the UNSW_DROP match silently fails.
    train = pd.read_csv(tr_path, encoding="utf-8-sig")
    test = pd.read_csv(te_path, encoding="utf-8-sig")
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]

    # 'attack_cat' is multiclass ('Normal' for benign); 'label' is 0/1.
    ytr_raw = train["attack_cat"].astype(str).str.strip().values
    yte_raw = test["attack_cat"].astype(str).str.strip().values

    drop_cols = [c for c in UNSW_DROP + ["attack_cat", "label"] if c in train.columns]
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)

    train = _clean_numeric(train, UNSW_CATEGORICAL)
    test = _clean_numeric(test, UNSW_CATEGORICAL)

    cats = [c for c in UNSW_CATEGORICAL if c in train.columns]
    train = pd.get_dummies(train, columns=cats)
    test = pd.get_dummies(test, columns=cats)
    train, test = train.align(test, join="left", axis=1, fill_value=0)
    test = test[train.columns]

    ytr, yte, class_names = _encode_labels(ytr_raw, yte_raw, binary,
                                           normal_label="Normal")
    return train, ytr, test, yte, class_names


# --- CICIDS2017 ------------------------------------------------------------- #
# MachineLearningCVE CSVs: no canonical split (we stratify), leading spaces in
# headers, Inf/NaN in flow-rate cols. Drop flow-identifier/leaky cols if present.
CICIDS_LEAKY = [
    "flow id", "flowid", "source ip", "src ip", "destination ip", "dst ip",
    "source port", "src port", "destination port", "dst port",
    "timestamp", "fwd header length.1",  # known duplicate column
]


def _prep_cicids(binary: bool, seed: int):
    path = os.path.join(DATA_DIR, "cicids2017.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "CICIDS2017 combined CSV not found. Run: "
            "wsl -d Ubuntu-22.04 -- bash -lc 'bash ~/download_cicids.sh' "
            "(script at src/download_cicids.sh) which downloads & concatenates the "
            "MachineLearningCVE CSVs into data/cicids2017.csv")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    label_col = "label"
    if label_col not in df.columns:
        cand = [c for c in df.columns if "label" in c]
        if not cand:
            raise ValueError("CICIDS2017: no label column found")
        label_col = cand[0]

    y_raw = df[label_col].astype(str).str.strip().values
    df = df.drop(columns=[label_col])

    drop = [c for c in df.columns if c in CICIDS_LEAKY]
    if drop:
        df = df.drop(columns=drop)

    df = _clean_numeric(df, categorical=[])  # ML-CVE features are all numeric

    # Benign traffic is labelled 'BENIGN'.
    y_raw = np.array(["BENIGN" if str(v).upper().startswith("BENIGN") else v
                      for v in y_raw])

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=seed,
                                      stratify=y_raw)
    train = df.iloc[tr_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)
    ytr_raw, yte_raw = y_raw[tr_idx], y_raw[te_idx]

    ytr, yte, class_names = _encode_labels(ytr_raw, yte_raw, binary,
                                           normal_label="BENIGN")
    return train, ytr, test, yte, class_names


# --- NF-ToN-IoT-v2 (NetFlow IoT) -------------------------------------------- #
# NetFlow-format ToN_IoT (Sarhan et al.); a standard IoT-IDS benchmark used by
# several QML-IDS papers we compare against. Label cols: 'Label' (0/1) and
# 'Attack' (multiclass, 'Benign' for normal). The 4-tuple of flow identifiers is
# strongly leaky (host/port memorisation) and MUST be dropped.
TONIOT_LEAKY = [
    "ipv4_src_addr", "ipv4_dst_addr", "l4_src_port", "l4_dst_port",
    "flow_id", "flow_start_milliseconds", "flow_end_milliseconds",
]
# Cap rows read (the full CSV is huge); stratified down-sample keeps class mix.
TONIOT_MAX_ROWS = int(os.environ.get("TONIOT_MAX_ROWS", "200000"))


def _prep_toniot(binary: bool, seed: int):
    # Accept either a single combined file or the HF train.csv we download.
    candidates = ["nf_toniot.csv", "NF-ToN-IoT-v2.csv", "toniot.csv"]
    path = next((os.path.join(DATA_DIR, c) for c in candidates
                 if os.path.exists(os.path.join(DATA_DIR, c))), None)
    if path is None:
        raise FileNotFoundError(
            "NF-ToN-IoT-v2 CSV not found. Run: "
            "wsl -d Ubuntu-22.04 -- bash -lc 'bash ~/download_toniot.sh' "
            "(script at src/download_toniot.sh) -> data/nf_toniot.csv")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Labels: prefer 'attack' (multiclass); 'label' is the 0/1 indicator.
    if "attack" in df.columns:
        y_raw = df["attack"].astype(str).str.strip()
        normal = "Benign"
        # normalise case so 'benign'/'BENIGN'/'Benign' all map to normal
        y_raw = y_raw.apply(lambda v: "Benign" if v.lower() == "benign" else v).values
    elif "label" in df.columns:
        y_raw = df["label"].astype(int).map({0: "Benign", 1: "attack"}).values
        normal = "Benign"
    else:
        raise ValueError("NF-ToN-IoT: no 'attack'/'label' column found")

    drop = [c for c in df.columns if c in TONIOT_LEAKY or c in ("attack", "label")]
    df = df.drop(columns=drop)
    df = _clean_numeric(df, categorical=[])  # NetFlow features are numeric

    # Optional stratified down-sample to keep memory/time sane on the laptop.
    n = len(df)
    if TONIOT_MAX_ROWS and n > TONIOT_MAX_ROWS:
        keep, _ = train_test_split(np.arange(n), train_size=TONIOT_MAX_ROWS,
                                   random_state=seed, stratify=y_raw)
        df = df.iloc[keep].reset_index(drop=True)
        y_raw = y_raw[keep]

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=seed,
                                      stratify=y_raw)
    train = df.iloc[tr_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)
    ytr_raw, yte_raw = y_raw[tr_idx], y_raw[te_idx]

    ytr, yte, class_names = _encode_labels(ytr_raw, yte_raw, binary,
                                           normal_label=normal)
    return train, ytr, test, yte, class_names


# --------------------------------------------------------------------------- #
# Shared cleaning / encoding helpers                                          #
# --------------------------------------------------------------------------- #
def _clean_numeric(df: pd.DataFrame, categorical) -> pd.DataFrame:
    """Coerce non-categorical columns to numeric, replace +/-inf with NaN, then
    fill NaN with 0. Train/test-symmetric (no fitted state)."""
    cat = set(categorical)
    for c in df.columns:
        if c in cat:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = [c for c in df.columns if c not in cat]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(0.0)
    return df


def _encode_labels(ytr_raw, yte_raw, binary: bool, normal_label: str):
    """Return (ytr, yte, class_names).

    binary=True  -> 0 = normal (`normal_label`), 1 = any attack.
    binary=False -> integer-encoded multiclass; encoder fit on the UNION of
                    train+test label strings so unseen test classes don't crash.
    """
    if binary:
        ytr = (np.asarray(ytr_raw) != normal_label).astype(np.int64)
        yte = (np.asarray(yte_raw) != normal_label).astype(np.int64)
        return ytr, yte, ["normal", "attack"]

    le = LabelEncoder()
    le.fit(np.concatenate([np.asarray(ytr_raw), np.asarray(yte_raw)]))
    ytr = le.transform(np.asarray(ytr_raw)).astype(np.int64)
    yte = le.transform(np.asarray(yte_raw)).astype(np.int64)
    return ytr, yte, list(le.classes_)


# --------------------------------------------------------------------------- #
# Dimensionality reduction                                                    #
# --------------------------------------------------------------------------- #
def _reduce(Xtr, Xte, n_features, reduction, seed):
    """Reduce to `n_features` dims. Returns (Xtr, Xte, info_dict).

    `Xtr`/`Xte` are assumed already MinMax-scaled to [0, 1] (variance-preserving
    for the heavily one-hot feature spaces here -- see module docstring).
    """
    reduction = (reduction or "none").lower()
    d = Xtr.shape[1]
    if reduction == "none":
        # 'none' means keep ALL encoded dims; n_features is ignored (the full
        # classical-baseline view). Passing n_features=0 must NOT slice to empty.
        return Xtr, Xte, {"reduction": "none", "explained_variance": 1.0}
    if n_features >= d:
        # Asked for at least as many components as we have -> nothing to reduce.
        return Xtr, Xte, {"reduction": "none", "explained_variance": 1.0}

    if reduction == "pca":
        pca = PCA(n_components=n_features, random_state=seed).fit(Xtr)
        info = {"reduction": "pca",
                "explained_variance": float(pca.explained_variance_ratio_.sum())}
        return pca.transform(Xtr), pca.transform(Xte), info

    if reduction == "autoencoder":
        return _autoencoder_reduce(Xtr, Xte, n_features, seed)

    raise ValueError(f"unknown reduction '{reduction}'")


# Cap the AE training set: fitting on millions of rows is needlessly slow and can
# OOM the 4GB GPU. We fit the encoder on a random subset, then ENCODE all rows.
AE_FIT_MAX_ROWS = int(os.environ.get("AE_FIT_MAX_ROWS", "50000"))


def _autoencoder_reduce(Xtr, Xte, n_features, seed, epochs=60, bs=256, lr=1e-3):
    """Train a small classical MLP autoencoder on train, return bottleneck codes.

    Encoder: d -> 64 -> 32 -> n_features (Sigmoid -> codes in [0,1]); symmetric
    decoder. Reconstruction MSE. The encoder is FIT on a random subset of train
    (<= AE_FIT_MAX_ROWS) for speed, then applied to all train/test rows. Falls
    back to PCA if torch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover - environment fallback
        print(f"[data] torch unavailable ({e}); autoencoder -> PCA fallback")
        return _reduce(Xtr, Xte, n_features, "pca", seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Xtr.shape[1]
    h1, h2 = 64, 32

    # Random subset for fitting (deterministic via seed); encode everything later.
    Xfit = Xtr
    if AE_FIT_MAX_ROWS and len(Xtr) > AE_FIT_MAX_ROWS:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(Xtr), size=AE_FIT_MAX_ROWS, replace=False)
        Xfit = Xtr[sel]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, n_features), nn.Sigmoid(),  # codes in [0,1]
            )
            self.dec = nn.Sequential(
                nn.Linear(n_features, h2), nn.ReLU(),
                nn.Linear(h2, h1), nn.ReLU(),
                nn.Linear(h1, d), nn.Sigmoid(),  # inputs are MinMax [0,1]
            )

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    model = AE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xf = torch.tensor(Xfit, dtype=torch.float32, device=dev)
    n = len(Xf)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            recon, _ = model(Xf[idx])
            loss = lossf(recon, Xf[idx])
            loss.backward()
            opt.step()
    model.eval()

    def _encode(X):  # batch the encode so huge test sets don't OOM the GPU
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor(X[i:i + 8192], dtype=torch.float32, device=dev)
                out.append(model.enc(xb).cpu().numpy())
        return np.concatenate(out, axis=0)

    with torch.no_grad():
        recon_fit, _ = model(Xf)
        recon_mse = float(lossf(recon_fit, Xf).item())
    ztr = _encode(Xtr)
    zte = _encode(Xte)
    info = {"reduction": "autoencoder", "recon_mse": recon_mse,
            "explained_variance": float("nan")}
    return ztr.astype(np.float32), zte.astype(np.float32), info


# --------------------------------------------------------------------------- #
# Unified public API                                                          #
# --------------------------------------------------------------------------- #
_PREP = {
    "nslkdd": _prep_nslkdd,
    "nsl-kdd": _prep_nslkdd,
    "unsw": _prep_unsw,
    "unsw-nb15": _prep_unsw,
    "cicids": _prep_cicids,
    "cicids2017": _prep_cicids,
    "toniot": _prep_toniot,
    "ton_iot": _prep_toniot,
    "nf-toniot": _prep_toniot,
    "iot": _prep_toniot,
}


def _process(dataset_name, n_features, reduction, binary, scale, seed):
    """Internal: returns (Xtr, ytr, Xte, yte, meta). Shared by load/load_meta."""
    key = dataset_name.strip().lower()
    if key not in _PREP:
        raise ValueError(f"unknown dataset '{dataset_name}'. "
                         f"options: {sorted(set(_PREP))}")
    Xtr_df, ytr, Xte_df, yte, class_names = _PREP[key](binary, seed)

    raw_dim = int(Xtr_df.shape[1])
    # Clip in float64 to the float32 representable range BEFORE casting: some
    # NetFlow byte/throughput columns exceed float32 max, so a naive .astype
    # (float32) turns them into +/-inf and breaks PCA/MinMax downstream.
    f32max = np.finfo(np.float32).max
    Xtr = np.clip(Xtr_df.values.astype(np.float64), -f32max, f32max).astype(np.float32)
    Xte = np.clip(Xte_df.values.astype(np.float64), -f32max, f32max).astype(np.float32)
    # Belt-and-braces: any residual non-finite -> 0.
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=f32max, neginf=-f32max)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=f32max, neginf=-f32max)

    # MinMax to [0,1] BEFORE reduction (variance-preserving for one-hot spaces),
    # fit on train only.
    pre = MinMaxScaler().fit(Xtr)
    Xtr, Xte = pre.transform(Xtr), pre.transform(Xte)

    Xtr, Xte, rinfo = _reduce(Xtr, Xte, n_features, reduction, seed)

    # Final scaling for the model / quantum encoding, fit on train only.
    if scale == "minmax":
        mm = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr)  # angle encoding range
    elif scale == "standard":
        mm = StandardScaler().fit(Xtr)
    else:
        raise ValueError(f"unknown scale '{scale}'")
    Xtr, Xte = mm.transform(Xtr), mm.transform(Xte)

    meta = {
        "dataset": key,
        "n_features": int(Xtr.shape[1]),
        "reduction": rinfo.get("reduction"),
        "explained_variance": rinfo.get("explained_variance"),
        "recon_mse": rinfo.get("recon_mse"),
        "binary": binary,
        "n_classes": int(len(class_names)),
        "class_names": class_names,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "raw_dim_after_encoding": raw_dim,
        "attack_rate_train": float(np.mean(ytr != 0)) if binary else float("nan"),
    }
    return (Xtr.astype(np.float32), ytr.astype(np.int64),
            Xte.astype(np.float32), yte.astype(np.int64), meta)


def load(dataset_name: str, n_features: int = 8, reduction: str = "pca",
         binary: bool = True, scale: str = "minmax", seed: int = 42):
    """Load & preprocess an IDS dataset to a fixed feature width.

    Parameters
    ----------
    dataset_name : "nslkdd" | "unsw" | "cicids" (aliases accepted)
    n_features   : output dimensionality (== qubit count for the quantum models)
    reduction    : "pca" | "autoencoder" | "none"
    binary       : True -> normal(0)/attack(1); False -> multiclass int labels
    scale        : "minmax" (final range [0, pi], angle encoding) | "standard"
    seed         : RNG seed (PCA / AE init / CICIDS split)

    Returns
    -------
    (X_train, y_train, X_test, y_test)
        X_* : np.float32, shape (N, n_features); y_* : np.int64, shape (N,)
    """
    Xtr, ytr, Xte, yte, meta = _process(dataset_name, n_features, reduction,
                                        binary, scale, seed)
    ev = meta["explained_variance"]
    ev_s = f"{ev:.3f}" if isinstance(ev, float) and ev == ev else "n/a"
    print(f"[data] {meta['dataset']} | feat={meta['n_features']} "
          f"| reduction={meta['reduction']} | explained_var={ev_s} "
          f"| train={Xtr.shape} test={Xte.shape} | classes={meta['n_classes']}")
    return Xtr, ytr, Xte, yte


def load_meta(dataset_name: str, n_features: int = 8, reduction: str = "pca",
              binary: bool = True, scale: str = "minmax", seed: int = 42):
    """Same as load() but also returns a metadata dict as a 5th element:
    (X_train, y_train, X_test, y_test, meta)."""
    return _process(dataset_name, n_features, reduction, binary, scale, seed)


def load_full(dataset_name: str, binary: bool = True, scale: str = "standard",
              seed: int = 42, with_meta: bool = False):
    """Full-dimensional view (no reduction) for the classical baselines.

    Generalises load_nslkdd_full to all datasets: returns every encoded/cleaned
    feature so RF/XGBoost/SVM/MLP/CNN are reported at full strength (reducing to
    ~n_qubits PCA dims would unfairly handicap them). Default scale='standard'
    (z-score) is the usual choice for tree/SVM/NN baselines; 'minmax' also works.

    Returns (Xtr, ytr, Xte, yte) or, if with_meta, (..., meta). n_features is
    ignored — ALL dims are kept (reduction='none').
    """
    Xtr, ytr, Xte, yte, meta = _process(dataset_name, n_features=0,
                                        reduction="none", binary=binary,
                                        scale=scale, seed=seed)
    print(f"[data] {meta['dataset']} FULL | dims={meta['n_features']} "
          f"| train={Xtr.shape} test={Xte.shape} | classes={meta['n_classes']}")
    return (Xtr, ytr, Xte, yte, meta) if with_meta else (Xtr, ytr, Xte, yte)


if __name__ == "__main__":
    # Quick self-check on whatever datasets are available locally.
    for name in ["nslkdd", "unsw", "cicids"]:
        try:
            Xtr, ytr, Xte, yte, m = load_meta(name, n_features=8, reduction="pca")
            print(f"OK {name}: train={Xtr.shape} test={Xte.shape} "
                  f"evr={m['explained_variance']}")
        except FileNotFoundError as e:
            print(f"SKIP {name}: {e}")
