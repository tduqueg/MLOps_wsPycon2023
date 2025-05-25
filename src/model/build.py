import argparse
import os
import joblib
import wandb

from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor

# ────────────────────────── CLI ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="Execution ID")
exec_id = parser.parse_args().IdExecution or "local-test"
print(f"IdExecution: {exec_id}")

# ───────────── CONFIG DEL MODELO ─────────────
model_cfg = {
    "model": "HistGradientBoostingRegressor",
    "params": {
        "max_depth": None,          
        "max_iter": 500,            
        "learning_rate": 0.05,
        "l2_regularization": 1.0,
        "subsample": 0.8,           
        "early_stopping": True,
        "scoring": "loss",
        "random_state": 42,
    },
}

model = HistGradientBoostingRegressor(**model_cfg["params"])

# ───────────── SERIALIZAR LOCAL ─────────────
os.makedirs("./model", exist_ok=True)
local_path = "./model/initialized_model_gbr_hist.pkl"
joblib.dump(model, local_path)

# ───────────── SUBIR A W&B ─────────────
with wandb.init(
    project="ExperienciasEnAnalitica",
    name=f"Initialize Model ExecId-{exec_id}",
    job_type="initialize-model",
    config=model_cfg,
) as run:
    art = wandb.Artifact(
        "gbr-hist",
        type="model",
        description="HistGradientBoostingRegressor base (tuned)",
        metadata=model_cfg,
    )
    art.add_file(local_path)
    run.log_artifact(art)
    print("🟢 Modelo base HistGradientBoosting subido a W&B.")
