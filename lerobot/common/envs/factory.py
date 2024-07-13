#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib

import gymnasium as gym
from omegaconf import DictConfig


def make_env(cfg: DictConfig, n_envs: int | None = None) -> gym.vector.VectorEnv:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """
    if n_envs is not None and n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    package_name = f"gym_{cfg.env.name}"

    try:
        if package_name != "gym_ballgame":
            importlib.import_module(package_name)
        else:
            import lerobot.common.envs.ballgame as ballgame

            gym.register(
                id="gym_ballgame/Ballgame-v0",
                entry_point="lerobot.common.envs.ballgame:BallgameEnv",
                max_episode_steps=ballgame.BallgameEnv.DEFAULT_TIMEOUT,
                kwargs={
                    "maze": ballgame.BallgameEnv.DEFAULT_MAZE,
                    "key_locations": ballgame.BallgameEnv.DEFAULT_KEY_LOCATIONS,
                    "width": ballgame.BallgameEnv.DEFAULT_WIDTH,
                    "height": ballgame.BallgameEnv.DEFAULT_HEIGHT,
                    "cell_size": ballgame.BallgameEnv.DEFAULT_CELL_SIZE,
                    "ball_radius": ballgame.BallgameEnv.DEFAULT_BALL_RADIUS,
                    "force_magnitude": ballgame.BallgameEnv.DEFAULT_FORCE_MAGNITUDE,
                    "mass": ballgame.BallgameEnv.DEFAULT_MASS,
                    "damping": ballgame.BallgameEnv.DEFAULT_DAMPING,
                    "fps": ballgame.BallgameEnv.DEFAULT_FPS,
                    "timeout": 1000,
                    "reset_location_noise_scale": 0.1,
                },
            )

    except ModuleNotFoundError as e:
        print(
            f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.env.name}]'`"
        )
        raise e

    gym_handle = f"{package_name}/{cfg.env.task}"
    gym_kwgs = dict(cfg.env.get("gym", {}))

    if cfg.env.get("episode_length"):
        gym_kwgs["max_episode_steps"] = cfg.env.episode_length

    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv
    env = env_cls(
        [
            lambda: gym.make(gym_handle, disable_env_checker=True, **gym_kwgs)
            for _ in range(n_envs if n_envs is not None else cfg.eval.batch_size)
        ]
    )

    return env
