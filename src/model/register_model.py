import json
import os
import sys
import logging
import subprocess
import mlflow
from mlflow.exceptions import MlflowException

# ─── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_registration")

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)


def load_model_info(path: str) -> dict:
    """Load JSON and ensure it has run_id & model_path."""
    with open(path, "r") as f:
        info = json.load(f)
    if "run_id" not in info or "model_path" not in info:
        raise ValueError("experiment_info.json must contain run_id and model_path")
    logger.debug(f"Loaded model info: {info}")
    return info


def find_local_tracking_uri() -> str:
    """
    Look for a local MLflow tracking directory (mlruns/ or mlartifacts/).
    Return a properly formatted file:/// URI.
    """
    cands = ["mlruns", "mlartifacts"]
    for d in cands:
        absd = os.path.abspath(d)
        if os.path.isdir(absd):
            # e.g. C:\path\to\project\mlruns
            uri_path = absd.replace(os.sep, "/")  # forward-slash
            uri = f"file:///{uri_path}"
            logger.info(f"Using local MLflow tracking store at: {uri}")
            return uri

    # fallback to ~/.mlflow
    home_dir = os.path.abspath(os.path.expanduser("~/.mlflow"))
    os.makedirs(home_dir, exist_ok=True)
    uri_path = home_dir.replace(os.sep, "/")
    uri = f"file:///{uri_path}"
    logger.info(f"No project mlruns/. Found. Falling back to: {uri}")
    return uri


def direct_register_model(name: str, info: dict) -> bool:
    run_id = info["run_id"]
    raw_path = info["model_path"]

    # always reconstruct as runs:/<run_id>/<artifact_subpath>
    if raw_path.startswith("mlflow-artifacts:"):
        artifact_sub = raw_path.split("/artifacts/")[-1]
    else:
        artifact_sub = raw_path
    model_uri = f"runs:/{run_id}/{artifact_sub}"
    logger.info(f"Registering model '{name}' from URI '{model_uri}'")

    # point MLflow at our detected local store
    tracking_uri = find_local_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        mv = mlflow.register_model(model_uri=model_uri, name=name)
        logger.info(f"Registered '{name}' as version {mv.version}")
        client.transition_model_version_stage(
            name=name, version=mv.version, stage="Staging"
        )
        logger.info(f"Transitioned '{name}' version {mv.version} → Staging")
        return True

    except MlflowException as e:
        logger.error(f"MLflowException: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


def main():
    try:
        # 1) Load your experiment_info.json
        info = load_model_info("experiment_info.json")
        model_name = "yt_chrome_plugin_model"

        # 2) Log mlflow CLI version
        out = subprocess.run(
            ["mlflow", "--version"], capture_output=True, text=True
        ).stdout.strip()
        logger.info(f"MLflow CLI version: {out}")

        # 3) Register + stage
        if direct_register_model(model_name, info):
            print(f"✅ Successfully registered '{model_name}'")
            logger.info("Done.")
        else:
            raise RuntimeError("direct_register_model returned False")

    except Exception as e:
        logger.error(f"Script failure: {e}", exc_info=True)
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
