from pathlib import Path
import argparse, joblib, pandas as pd
from utils.config import load_config, abspath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, help="parquet/csv to predict; 默认使用 processed 数据集")
    ap.add_argument("--output", required=False, help="输出预测路径；默认写到 config 中的 predictions_file")
    args = ap.parse_args()

    cfg, root = load_config()
    model = joblib.load(abspath(root, cfg["modeling"]["outputs"]["model_file"]))

    if args.input:
        in_path = Path(args.input)
        df = pd.read_parquet(in_path) if in_path.suffix==".parquet" else pd.read_csv(in_path)
    else:
        df = pd.read_parquet(abspath(root, cfg["processing"]["outputs"]["dataset_for_model"]))

    pred = model.predict(df)
    out = df.copy()
    out["prediction"] = pred

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = abspath(root, cfg["modeling"]["outputs"]["predictions_file"])
    out.to_parquet(out_path, index=False)
    print("Predictions saved to:", out_path)

if __name__ == "__main__":
    main()
