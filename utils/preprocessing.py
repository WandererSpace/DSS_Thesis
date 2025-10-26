from __future__ import annotations
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
import pyreadstat
from .config import load_config, abspath, ensure_dirs

ALIASES = {  # 常见字段别名兼容
    "n_age_dv": "n_dvage",
}

def _resolve_keep_cols(keep_cols, available):
    resolved = []
    for c in keep_cols:
        if c in available:
            resolved.append(c)
        elif c in ALIASES and ALIASES[c] in available:
            resolved.append(ALIASES[c])
    # 去重保序
    seen=set(); out=[]
    for c in resolved:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def load_candidate_vars(cfg, root: Path) -> list[str]:
    manifest = abspath(root, cfg["ukhls"]["candidate_vars_csv"])
    df = pd.read_csv(manifest)
    cols = df["varname"].astype(str).tolist()
    # 常用元数据列兜底加入
    for must in ["pidp","n_hidp","n_pno", cfg["processing"]["sample_flag_col"]]:
        if must not in cols:
            cols.append(must)
    return cols

def read_indresp_selected(cfg=None, root: Path | None=None) -> pd.DataFrame:
    cfg, root = load_config(root)
    dta_path = abspath(root, cfg["ukhls"]["indresp_dta"])

    # 加载候选字段清单
    manifest = abspath(root, cfg["ukhls"]["candidate_vars_csv"])
    df_manifest = pd.read_csv(manifest)
    keep_cols = df_manifest["varname"].astype(str).unique().tolist()

    # 使用 pandas 读取
    df = pd.read_stata(dta_path, columns=keep_cols)
    return df

def normalize_missing(df: pd.DataFrame, neg_codes: list[int]) -> pd.DataFrame:
    NEG = set(neg_codes)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64")
            df[c] = df[c].where(~df[c].isin(NEG), np.nan)
    return df

def filter_self_completion(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    # 只保留完成自填问卷的成人样本（确保 GHQ/SF12 可用）
    if flag_col in df.columns:
        return df[df[flag_col] == 1].copy()
    return df

def engineer_employment_history_features(df: pd.DataFrame) -> pd.DataFrame:
    # 片段数量（可能不存在，缺则填0）
    for col in ["n_nmpsp_dv","n_nnmpsp_dv","n_nunmpsp_dv"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 结束原因（裁员/解雇/合同期满…/COVID）
    end_cols  = [c for c in df.columns if re.match(r"n_reasendoth\d+_code", c)]
    next_cols = [c for c in df.columns if re.match(r"n_nextelse\d+", c)]

    def _count_codes(row, cols, codes):
        cnt = 0
        for c in cols:
            v = row.get(c)
            if pd.notna(v):
                try:
                    if int(v) in codes:
                        cnt += 1
                except Exception:
                    pass
        return cnt

    if end_cols:
        df["end_involuntary_cnt"] = df.apply(lambda r: _count_codes(r, end_cols, {3,4,5,26,27,28}), axis=1)
        df["end_covid_cnt"]       = df.apply(lambda r: _count_codes(r, end_cols, {23}), axis=1)
    else:
        df["end_involuntary_cnt"] = 0
        df["end_covid_cnt"] = 0

    if next_cols:
        df["next_unemp_cnt"] = df.apply(lambda r: _count_codes(r, next_cols, {1}), axis=1)  # 1=Unemployed/seeking work
    else:
        df["next_unemp_cnt"] = 0

    return df

def build_targets(df: pd.DataFrame, cfg) -> pd.DataFrame:
    # GHQ caseness（>=阈值记为1）
    ghq2 = cfg["modeling"]["target_classification"]
    thr  = cfg["modeling"]["caseness_threshold"]
    if ghq2 in df.columns:
        df["target_cls"] = (df[ghq2] >= thr).astype("Int64")

    # SF-12 MCS 作为回归目标
    mcs = cfg["modeling"]["target_regression"]
    if mcs in df.columns:
        df["target_reg"] = df[mcs].astype("float")
    return df

def run_basic_processing_and_save(cfg=None, root: Path | None=None):
    cfg, root = load_config(root)
    paths = cfg["paths"]; proc = cfg["processing"]

    out_sel  = abspath(root, proc["outputs"]["indresp_selected"])
    out_feat = abspath(root, proc["outputs"]["features"])
    out_ds   = abspath(root, proc["outputs"]["dataset_for_model"])
    ensure_dirs(out_sel.parent, out_feat.parent, out_ds.parent)

    df = read_indresp_selected(cfg, root)
    df = normalize_missing(df, proc["negative_missing_codes"])
    df = filter_self_completion(df, proc["sample_flag_col"])
    df = engineer_employment_history_features(df)
    df = build_targets(df, cfg)

    # 保存
    df.to_parquet(out_sel, index=False)
    # 可按需在此处做进一步特征处理后另存为 out_feat/out_ds
    df.to_parquet(out_feat, index=False)
    df.to_parquet(out_ds, index=False)
    return df, out_ds
