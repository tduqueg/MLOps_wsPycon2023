import argparse
from io import StringIO

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import wandb

# ───────────── CLI ─────────────
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="Execution ID")
args = parser.parse_args()
exec_id = args.IdExecution or "local-test"
print(f"IdExecution: {exec_id}")

# ───────── LOAD & SPLIT ────────
def make_splits(train_size=0.8, val_size=0.1):
    cali = fetch_california_housing(as_frame=True)
    df = cali.frame          

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=train_size, random_state=42
    )
    val_rel = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=val_rel, random_state=42
    )

    return {
        "training":   pd.concat([X_train, y_train], axis=1),
        "validation": pd.concat([X_val,  y_val],  axis=1),
        "test":       pd.concat([X_test, y_test], axis=1),
    }

# ─────────── W&B LOG ───────────
def load_and_log():
    with wandb.init(
        project="ExperienciasEnAnalitica",
        name=f"Load California Data ExecId-{exec_id}",
        job_type="load-data",
    ) as run:
        splits = make_splits()
        art = wandb.Artifact(
            "california-raw",
            type="dataset",
            description="California Housing split train/val/test",
            metadata={k: len(v) for k, v in splits.items()},
        )

        for name, df in splits.items():
            buf = StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            with art.new_file(f"{name}.csv", mode="w") as f:
                f.write(buf.read())

        run.log_artifact(art)
        print("🟢 Dataset California Housing subido.")

if __name__ == "__main__":
    load_and_log()
