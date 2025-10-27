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
    """Robustly read the subset of variables that actually exist in the .dta file."""
    cfg, root = load_config(root)
    dta_path = abspath(root, cfg["ukhls"]["indresp_dta"])

    # 1. 读取候选变量清单
    manifest = abspath(root, cfg["ukhls"]["candidate_vars_csv"])
    df_manifest = pd.read_csv(manifest)
    keep_cols = df_manifest["varname"].astype(str).unique().tolist()

    # 2. 安全获取 .dta 中的真实列名（兼容旧 pandas）
    try:
        reader = pd.read_stata(dta_path, iterator=True)
        preview = reader.read(1)
        available_cols = list(preview.columns)
        reader.close()
    except Exception:
        # 如果 iterator 失败，就直接全读一遍（最慢但最安全）
        preview = pd.read_stata(dta_path)
        available_cols = list(preview.columns)

    # 3. 只保留确实存在的列
    usecols = [c for c in keep_cols if c in available_cols]

    # 4. 真正读取需要的字段
    df = pd.read_stata(dta_path, columns=usecols)
    return df

def normalize_missing(df: pd.DataFrame, neg_codes: list[int]) -> pd.DataFrame:
    NEG = set(neg_codes)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64")
            df[c] = df[c].where(~df[c].isin(NEG), np.nan)
    return df

def engineer_employment_history_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import re

    # 强制数值化：就业片段计数列
    for col in ["n_nmpsp_dv", "n_nnmpsp_dv", "n_nunmpsp_dv"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

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
        df["next_unemp_cnt"] = df.apply(lambda r: _count_codes(r, next_cols, {1}), axis=1)
    else:
        df["next_unemp_cnt"] = 0

    return df

def build_targets(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    构建分类目标变量 target_cls，基于 GHQ caseness 指标：n_scghq2_dv。
    使用阈值（默认 4+）划分心理困扰状态。

    注意：不再使用 n_scflag_dv 进行样本筛选，直接以目标字段值是否有效为标准。
    """

    target_col = cfg["processing"]["classification_target"]
    threshold = cfg["processing"]["classification_threshold"]

    # 保留有效 GHQ 值（0~12），去除非法编码 -9/-8 等
    df = df[df[target_col].between(0, 12)]

    # 创建二分类目标列
    df.loc[:, "target_cls"] = (df[target_col] >= threshold).astype("Int64")

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
    # ⬇️ 添加调试行
    print("target_cls existence:", "target_cls" in df.columns)

    df = engineer_employment_history_features(df)
    df = build_targets(df, cfg)
    # 🚫 消除 category 类型（否则 to_parquet/pyarrow 会出错）
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)

    # 保存
    df.to_parquet(out_sel, index=False)
    # 可按需在此处做进一步特征处理后另存为 out_feat/out_ds
    df.to_parquet(out_feat, index=False)
    df.to_parquet(out_ds, index=False)
    print("Final processed dataset shape:", df.shape)
    print("target_cls value counts:", df["target_cls"].value_counts(dropna=False) if "target_cls" in df.columns else "No target_cls")
    return df, out_ds
    