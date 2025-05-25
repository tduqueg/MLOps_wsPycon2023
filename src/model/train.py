import argparse, os, joblib, pandas as pd, wandb
from sklearn.metrics import mean_squared_error, r2_score
#run
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="Execution ID")
exec_id = parser.parse_args().IdExecution or "local-test"
print(f"IdExecution: {exec_id}")

# ---------- HELPERS ----------
def load_split(dir_path, split):
    df = pd.read_csv(os.path.join(dir_path, f"{split}.csv"))
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return X, y

# ---------- TRAIN + LOG ----------
def train_and_log():
    with wandb.init(
        project="ExperienciasEnAnalitica",
        name=f"Train Model ExecId-{exec_id}",
        job_type="train-model",
    ) as run:
    
        data_art = run.use_artifact("california-preprocess:latest")
        data_dir = data_art.download()
        X_train, y_train = load_split(data_dir, "training")
        X_val,   y_val   = load_split(data_dir, "validation")

        
        model_art = run.use_artifact("gbr-regressor:latest")
        model_dir = model_art.download()
        model = joblib.load(os.path.join(model_dir, "initialized_model_gbr.pkl"))

    
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        r2  = r2_score(y_val, preds)

        wandb.log({"validation/mse": mse, "validation/r2": r2})
        print(f"🔶  Val MSE: {mse:.3f} | R²: {r2:.3f}")

        
        joblib.dump(model, "trained_model_gbr.pkl")
        trained_art = wandb.Artifact(
            "trained-model",
            type="model",
            description="GBR trained on California Housing (scaled)",
            metadata={"mse": mse, "r2": r2},
        )
        trained_art.add_file("trained_model_gbr.pkl")
        run.log_artifact(trained_art)
        print("🟢 Modelo entrenado subido a W&B.")

if __name__ == "__main__":
    train_and_log()
