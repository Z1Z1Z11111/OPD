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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    import torch
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor holding one or more independent ALFWorld sub-environments.
    Packing multiple text sub-environments per actor avoids spawning hundreds of
    Ray/TextWorld processes and keeps host RAM bounded.
    """

    def __init__(self, config, seeds, base_env):
        self.envs = []
        for seed in seeds:
            env = base_env.init_env(batch_size=1)
            env.seed(seed)
            self.envs.append(env)

    def step(self, actions):
        """Execute one step in each sub-environment."""
        results = []
        for env, action in zip(self.envs, actions):
            if action == "__inactive__":
                results.append(
                    (
                        [""],
                        [0.0],
                        [True],
                        {
                            "admissible_commands": [[]],
                            "won": [False],
                            "observation_text": [""],
                            "extra.gamefile": [None],
                        },
                    )
                )
                continue
            obs, scores, dones, infos = env.step([action])
            infos['observation_text'] = obs
            results.append((obs, scores, dones, infos))
        return results

    def reset(self):
        """Reset all sub-environments."""
        results = []
        for env in self.envs:
            obs, infos = env.reset()
            infos['observation_text'] = obs
            results.append((obs, infos))
        return results

    def getobs(self):
        """Get current observation images for all sub-environments."""
        return [get_obs_image(env).cpu() for env in self.envs]

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes (pack text envs into 64 train / 32 val actors)
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        seeds = [seed + (i // self.group_n) for i in range(self.num_processes)]
        worker_cap = 64 if is_train else 32
        max_workers = self.num_processes if self.multi_modal else min(self.num_processes, worker_cap)
        base_size, rem = divmod(self.num_processes, max_workers)
        self.workers = []
        self.worker_slices = []
        start = 0
        for w in range(max_workers):
            end = start + base_size + (1 if w < rem else 0)
            self.worker_slices.append((start, end))
            self.workers.append(env_worker.remote(config, seeds[start:end], base_env))
            start = end

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = [
            worker.step.remote(actions[start:end])
            for worker, (start, end) in zip(self.workers, self.worker_slices)
        ]

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = [item for chunk in ray.get(futures) for item in chunk]
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = [worker.reset.remote() for worker in self.workers]

        # Collect results
        results = [item for chunk in ray.get(futures) for item in chunk]
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = [worker.getobs.remote() for worker in self.workers]
        return [img for chunk in ray.get(futures) for img in chunk]

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)