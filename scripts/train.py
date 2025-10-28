from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from models.baseline_model import make_baseline_pipeline
from utils.config import load_config, abspath, ensure_dirs
from utils.preprocessing import run_basic_processing_and_save
from utils.metrics import classification_metrics, regression_metrics, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification","regression"], default="classification")
    ap.add_argument("--model", choices=["logreg","rf","hgbt"], default=None)
    ap.add_argument("--rebuild", action="store_true", help="从原始dta重建processed数据")
    args = ap.parse_args()

    cfg, root = load_config()
    proc_out = cfg["processing"]["outputs"]["dataset_for_model"]
    ds_path = abspath(root, proc_out)

    if args.rebuild or not ds_path.exists():
        _, _ = run_basic_processing_and_save(cfg, root)

    df = pd.read_parquet(ds_path)

    task = args.task
    model_name = args.model or cfg["modeling"]["model_name"]
    weight_col = cfg["modeling"].get("weight_col")
    group_col  = cfg["modeling"].get("group_col")

    if task == "classification":
        y = df["target_cls"].astype("Int64").dropna()
    else:
        y = df["target_reg"].astype("float").dropna()

    # 同步 X，去除缺失y的行
    df = df.loc[y.index]

    reserved = [group_col, weight_col, "target_cls", "target_reg", "pidp", "n_hidp", "n_pno"]
    X = df.drop(columns=[c for c in reserved if c in df.columns])

    pipe = make_baseline_pipeline(task, X, reserved=[])

    # 分组CV（以户为单位防泄露）
    gkf = GroupKFold(n_splits=cfg["modeling"]["n_splits_cv"])
    groups = df[group_col] if group_col in df.columns else None
    sample_weight = df[weight_col] if (weight_col and weight_col in df.columns) else None

    preds, probs, trues = [], [], []
    for tr, te in gkf.split(X, y, groups):
        pipe.fit(X.iloc[tr], y.iloc[tr], **({"model__sample_weight": sample_weight.iloc[tr]} if sample_weight is not None and task=="classification" else {}))
        p = pipe.predict(X.iloc[te])
        preds.append(p); trues.append(y.iloc[te])
        if task == "classification":
            try:
                pr = pipe.predict_proba(X.iloc[te])[:,1]
            except Exception:
                pr = None
            probs.append(pr)

    y_true = pd.concat([pd.Series(t) for t in trues], axis=0).values
    y_pred = np.concatenate(preds)

    if task == "classification":
        y_prob = None if any(p is None for p in probs) else np.concatenate(probs)
        m = classification_metrics(y_true, y_pred, y_prob, sample_weight=None)
    else:
        m = regression_metrics(y_true, y_pred, sample_weight=None)

    # 以全量拟合后保存模型
    pipe.fit(X, y)
    model_file = abspath(root, cfg["modeling"]["outputs"]["model_file"])
    ensure_dirs(model_file.parent)
    joblib.dump(pipe, model_file)

    metrics_file = abspath(root, cfg["modeling"]["outputs"]["metrics_file"])
    save_json(m, metrics_file)

    # 保存预测
    pred_path = abspath(root, cfg["modeling"]["outputs"]["predictions_file"])
    out_df = df[[group_col]].copy() if group_col in df.columns else pd.DataFrame(index=df.index)
    out_df["y_true"] = y
    out_df["y_pred"] = pipe.predict(X)
    if task == "classification":
        try:
            out_df["y_prob"] = pipe.predict_proba(X)[:,1]
        except Exception:
            pass
    ensure_dirs(pred_path.parent)
    out_df.to_parquet(pred_path, index=False)

    print("Saved model to:", model_file)
    print("Saved metrics to:", metrics_file)
    print("Saved predictions to:", pred_path)
    print("CV metrics:", json.dumps(m, indent=2))

if __name__ == "__main__":
    main()