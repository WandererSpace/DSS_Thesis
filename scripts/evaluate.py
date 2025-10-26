from pathlib import Path
import joblib, json, pandas as pd
from utils.config import load_config, abspath
from utils.metrics import classification_metrics, regression_metrics, save_json

def main():
    cfg, root = load_config()
    model = joblib.load(abspath(root, cfg["modeling"]["outputs"]["model_file"]))
    ds_path = abspath(root, cfg["processing"]["outputs"]["dataset_for_model"])
    df = pd.read_parquet(ds_path)

    if "target_cls" in df:
        y = df["target_cls"].dropna()
        df = df.loc[y.index]
        y_pred = model.predict(df)
        try:
            y_prob = model.predict_proba(df)[:,1]
        except Exception:
            y_prob = None
        m = classification_metrics(y, y_pred, y_prob)
    else:
        y = df["target_reg"].dropna()
        df = df.loc[y.index]
        y_pred = model.predict(df)
        m = regression_metrics(y, y_pred)

    metrics_file = abspath(root, cfg["modeling"]["outputs"]["metrics_file"])
    save_json(m, metrics_file)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
