import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict

CAT_COLS = ["neighbourhood_group", "room_type"]
NUM_COLS = ["minimum_nights", "number_of_reviews", "availability_365", "amenity_score"]
TARGET_COL = "price_class"

@dataclass
class PreprocessArtifacts:
    onehot_columns: List[str]
    num_means: np.ndarray
    num_stds: np.ndarray
    num_medians: Dict[str, float]
    cat_fill_value: str = "Unknown"

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out

def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_pred = np.argmax(y_prob, axis=1)
    return float(np.mean(y_pred == y_true))

def stratified_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts_train, parts_val = [], []

    for cls, grp in df.groupby(TARGET_COL):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)

        cut = int(len(idx) * (1 - val_frac))
        parts_train.append(df.loc[idx[:cut]])
        parts_val.append(df.loc[idx[cut:]])

    train_df = pd.concat(parts_train).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(parts_val).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df


def fit_preprocess(train_df: pd.DataFrame) -> PreprocessArtifacts:
    df = train_df.copy()

    # numeric median fill
    num_medians = {c: float(df[c].median()) for c in NUM_COLS}
    for c in NUM_COLS:
        df[c] = df[c].fillna(num_medians[c])

    # categorical fill
    for c in CAT_COLS:
        df[c] = df[c].fillna("Unknown").astype(str)

    # one-hot columns fixed from training
    dummies = pd.get_dummies(df[CAT_COLS], prefix=CAT_COLS, drop_first=False)
    onehot_columns = list(dummies.columns)

    # standardization stats from training
    num_values = df[NUM_COLS].to_numpy(dtype=np.float64)
    means = num_values.mean(axis=0)
    stds = num_values.std(axis=0) + 1e-8

    return PreprocessArtifacts(
        onehot_columns=onehot_columns,
        num_means=means,
        num_stds=stds,
        num_medians=num_medians
    )

def transform(df_in: pd.DataFrame, art: PreprocessArtifacts) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = df_in.copy()

    for c in NUM_COLS:
        df[c] = df[c].fillna(art.num_medians[c])

    for c in CAT_COLS:
        df[c] = df[c].fillna(art.cat_fill_value).astype(str)

    dummies = pd.get_dummies(df[CAT_COLS], prefix=CAT_COLS, drop_first=False)
    for col in art.onehot_columns:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[art.onehot_columns]

    num = df[NUM_COLS].to_numpy(dtype=np.float64)
    num = (num - art.num_means) / art.num_stds

    X = np.concatenate([num, dummies.to_numpy(dtype=np.float64)], axis=1)

    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].to_numpy(dtype=int)

    feature_names = NUM_COLS + art.onehot_columns
    return X, y, feature_names
