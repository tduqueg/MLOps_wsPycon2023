import argparse, os, joblib, wandb
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler

# ──────────── CLI ────────────
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="Execution ID")
exec_id = parser.parse_args().IdExecution or "local-test"
print(f"IdExecution: {exec_id}")

# ─────────── HELPERS ─────────
def load_split(dir_path, split):
    return pd.read_csv(os.path.join(dir_path, f"{split}.csv"))

def preprocess_and_log():
    with wandb.init(
        project="ExperienciasEnAnalitica",
        name=f"Preprocess Data ExecId-{exec_id}",
        job_type="preprocess-data",
    ) as run:
        
        raw_art = run.use_artifact("california-raw:latest")
        raw_dir = raw_art.download()

        train_df = load_split(raw_dir, "training")
        val_df   = load_split(raw_dir, "validation")
        test_df  = load_split(raw_dir, "test")

        
        scaler = StandardScaler()
        X_train = train_df.drop(columns=["MedHouseVal"])
        scaler.fit(X_train)

        def transform(df):
            X = df.drop(columns=["MedHouseVal"])
            y = df["MedHouseVal"]
            X_scaled = pd.DataFrame(
                scaler.transform(X), columns=X.columns, index=X.index
            )
            return pd.concat([X_scaled, y], axis=1)

        train_p = transform(train_df)
        val_p   = transform(val_df)
        test_p  = transform(test_df)

        
        proc_art = wandb.Artifact(
            "california-preprocess",
            type="dataset",
            description="California Housing escalado con StandardScaler",
            metadata={"scaler": "StandardScaler()"},
        )

        for name, df in zip(
            ["training", "validation", "test"], [train_p, val_p, test_p]
        ):
            buf = StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            with proc_art.new_file(f"{name}.csv", mode="w") as f:
                f.write(buf.read())

        
        joblib.dump(scaler, "scaler.pkl")
        proc_art.add_file("scaler.pkl")

        run.log_artifact(proc_art)
        print("🟢 Artefacto `california-preprocess` registrado.")

if __name__ == "__main__":
    preprocess_and_log()