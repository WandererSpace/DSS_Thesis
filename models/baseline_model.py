from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

def _guess_feature_types(df: pd.DataFrame, reserved: list[str]):
    Xcols = [c for c in df.columns if c not in reserved]
    # 规则：对象/分类→cat；数值但唯一值<=30或名字像编码（_dv/_cc）→cat；其余→num
    cat, num = [], []
    for c in Xcols:
        s = df[c]
        if s.dtype.name in ("object","category"):
            cat.append(c); continue
        if pd.api.types.is_bool_dtype(s):
            cat.append(c); continue
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 30:
            cat.append(c); continue
        if re.search(r"(_dv|_cc)$", c):
            cat.append(c); continue
        num.append(c)
    return num, cat

def make_baseline_pipeline(task: str, df: pd.DataFrame, reserved: list[str], model_name="logreg") -> Pipeline:
    num_features, cat_features = _guess_feature_types(df, reserved)

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])

    if task == "classification":
        if model_name == "logreg":
            est = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
        elif model_name == "rf":
            est = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced_subsample")
        else:
            est = HistGradientBoostingClassifier()
    else:
        if model_name == "rf":
            est = RandomForestRegressor(n_estimators=400, n_jobs=-1)
        elif model_name == "hgbt":
            est = HistGradientBoostingRegressor()
        else:
            est = LinearRegression()

    pipe = Pipeline([("pre", pre), ("model", est)])
    return pipe
