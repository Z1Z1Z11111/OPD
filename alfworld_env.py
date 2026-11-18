"""ALFWorld adapter extracted from PoS; only class name and data path adapted."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = {
    "train": "train",
    "eval_in_distribution": "valid_seen",
    "eval_out_of_distribution": "valid_unseen",
}


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


class ALFWorldEnv:
    """Adapt the ALFWorld text environment to the shared agent interface."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""

        from alfworld.agents.environment import get_environment

        self.data = project_path(config.get("data_dir", "data/alfworld"))
        environment = get_environment("AlfredTWEnv")(
            self._environment_config(config),
            train_eval=config["split"],
        )

        games = sorted(environment.game_files)
        shard_count = int(config.get("shard_count", 1))
        shard_rank = int(config.get("shard_rank", 0))
        if shard_count < 1 or not 0 <= shard_rank < shard_count:
            raise ValueError("Invalid shard_count/shard_rank")
        self.case_indices = list(range(len(games)))[config["start_case"] : config["end_case"]][shard_rank::shard_count]
        self.case_indices = self.case_indices[config.get("lane_rank", 0)::config.get("lane_count", 1)]
        environment.game_files = [games[index] for index in self.case_indices]
        environment.num_games = len(environment.game_files)

        self.environment = environment.init_env(batch_size=1) if environment.game_files else None
        if self.environment is not None and config.get("seed") is not None:
            self.environment.seed(int(config["seed"]))
        self.total = len(environment.game_files)
        self.case_id = ""
        self.task = ""
        self.score = 0.0

    def __len__(self) -> int:
        """Len."""

        return self.total

    def reset(self) -> dict:
        """Reset."""

        observations, infos = self.environment.reset()
        game_file = Path(infos["extra.gamefile"][0])

        self.case_id = str(game_file.parent.relative_to(self.data))
        self.task = self._task_from_observation(observations[0])
        self.score = 0.0

        return self._state(observations[0], infos, done=False)

    def step(self, action: str) -> dict:
        """Step."""

        observations, scores, dones, infos = self.environment.step([action])
        self.score = float(scores[0])
        return self._state(observations[0], infos, done=bool(dones[0]))

    def close(self) -> None:
        """Close."""

        if self.environment is not None:
            self.environment.close()

    def _state(self, observation: str, infos: dict, done: bool) -> dict:
        """State."""

        return {
            "case_id": self.case_id,
            "task": self.task,
            "observation": observation,
            "actions": list(infos["admissible_commands"][0]),
            "done": done,
            "success": bool(infos["won"][0]),
            "score": self.score,
        }

    def _environment_config(self, config: dict) -> dict:
        """Environment config."""

        data = self.data
        return {
            "dataset": {
                "data_path": str(data / "json_2.1.1" / "train"),
                "eval_id_data_path": str(data / "json_2.1.1" / "valid_seen"),
                "eval_ood_data_path": str(data / "json_2.1.1" / "valid_unseen"),
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "logic": {
                "domain": str(data / "logic" / "alfred.pddl"),
                "grammar": str(data / "logic" / "alfred.twl2"),
            },
            "env": {
                "goal_desc_human_anns_prob": 0.0,
                "domain_randomization": False,
                "task_types": config["task_types"],
                "expert_type": "handcoded",
            },
            "general": {"training_method": "dagger"},
            "dagger": {"training": {"max_nb_steps_per_episode": config["max_steps"]}},
        }

    @staticmethod
    def _task_from_observation(observation: str) -> str:
        """Task from observation."""

        marker = "Your task is to:"
        return observation.split(marker, maxsplit=1)[1].strip()
