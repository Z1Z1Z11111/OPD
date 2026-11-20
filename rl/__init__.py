"""ALFWorld baseline selection and GRPO checkpoint locations."""
import json
from pathlib import Path

from alfworld_env import project_path


def run_root(config):
    return project_path(config.get("rl_output_dir", "outputs/grpo")) / Path(config["model"]).name


def evaluation_config(config, method=None):
    method = str(method or config.get("method", "react")).lower()
    if method == "react":
        return dict(config, method="react")
    if method != "rl":
        raise ValueError(f"Unknown method {method!r}; choose react or rl")
    if config.get("react_style", "raw") != "hiper":
        raise ValueError("GRPO checkpoints use react_style: hiper")
    checkpoint = config.get("rl_checkpoint")
    if checkpoint:
        checkpoint = project_path(checkpoint)
    else:
        manifest = run_root(config) / "latest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"No GRPO export at {manifest}. Run python -m rl.train or set rl_checkpoint.")
        saved = json.loads(manifest.read_text())
        if saved["base_model"] != str(project_path(config["model"]).resolve()):
            raise ValueError(f"GRPO base model mismatch in {manifest}")
        checkpoint = manifest.parent / saved["checkpoint"]
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing exported GRPO model config in {checkpoint}")
    return dict(config, method="rl", base_model=config["model"], model=str(checkpoint.resolve()))
