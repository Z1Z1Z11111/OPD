# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import os
from typing import List

from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
from agent_system.memory import SimpleMemory


def parse_gamefile(infos):
    return [info.get("extra.gamefile") for info in infos]


def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        infos[i]["extra.gamefile"] = gamefile[i] if "extra.gamefile" in infos[i] else None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {"text": full_text_obs, "image": image_obs, "anchor": text_obs}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        next_observations = {"text": full_text_obs, "image": image_obs, "anchor": text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find("Your task is to: ")
            if task_start != -1:
                self.tasks.append(obs[task_start + len("Your task is to: "):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )

        for i in range(len(text_obs)):
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != "help")
            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


def make_envs(config):
    """Create training and validation ALFWorld ReAct environments."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    if config.env.env_name != "alfworld/AlfredTWEnv":
        raise ValueError(f"Unsupported environment: {config.env.env_name}. Only alfworld/AlfredTWEnv is supported.")

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    alf_config_path = os.path.join(os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml")
    env_kwargs = {"eval_dataset": config.env.alfworld.eval_dataset}

    from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs

    train_envs = build_alfworld_envs(
        alf_config_path,
        config.env.seed,
        config.data.train_batch_size,
        group_n,
        is_train=True,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    val_envs = build_alfworld_envs(
        alf_config_path,
        config.env.seed + 1000,
        config.data.val_batch_size,
        1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    projection_f = partial(alfworld_projection)
    envs = AlfWorldEnvironmentManager(train_envs, projection_f, config)
    val_envs = AlfWorldEnvironmentManager(val_envs, projection_f, config)
    return envs, val_envs
