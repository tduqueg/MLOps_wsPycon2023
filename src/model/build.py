
import argparse, os, joblib, wandb
from sklearn.ensemble import GradientBoostingRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="Execution ID")
exec_id = parser.parse_args().IdExecution or "local-test"
print(f"IdExecution: {exec_id}")

model_cfg = {
    "model": "GradientBoostingRegressor(random_state=42)"
}

model = GradientBoostingRegressor(random_state=42)

os.makedirs("./model", exist_ok=True)
local_path = "./model/initialized_model_gbr.pkl"
joblib.dump(model, local_path)

with wandb.init(
    project="ExperienciasEnAnalitica",
    name=f"Initialize Model ExecId-{exec_id}",
    job_type="initialize-model",
    config=model_cfg,
) as run:
    art = wandb.Artifact(
        "gbr-regressor",
        type="model",
        description="GradientBoostingRegressor (base, sin scaler)",
        metadata=model_cfg,
    )
    art.add_file(local_path)
    run.log_artifact(art)
    print("🟢 Modelo base GBR subido a W&B.")
