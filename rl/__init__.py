"""ALFWorld method selection (react vs grpo) and GRPO checkpoint resolution."""
import json
from pathlib import Path

from alfworld_env import project_path

METHODS = ("react", "grpo")


def run_root(config):
    return project_path(config.get("rl_output_dir", "outputs/grpo")) / Path(config["model"]).name


def evaluation_config(config, method=None):
    selected = str(method or config.get("method", "react")).lower()
    if selected == "react":
        return dict(config, method="react")
    if selected != "grpo":
        raise ValueError(f"Unknown method {selected!r}; choose 'react' or 'grpo'")
    checkpoint = config.get("checkpoint") or config.get("rl_checkpoint")
    if checkpoint:
        checkpoint = project_path(checkpoint)
    else:
        manifest = run_root(config) / "latest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"No GRPO export at {manifest}. Run ./train-rl (or python -m rl.train) or pass --checkpoint."
            )
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["base_model"] != str(project_path(config["model"]).resolve()):
            raise ValueError(f"GRPO base model mismatch in {manifest}")
        checkpoint = manifest.parent / saved["checkpoint"]
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing exported GRPO model config in {checkpoint}")
    return dict(config, method="grpo", base_model=config["model"], model=str(checkpoint.resolve()))
