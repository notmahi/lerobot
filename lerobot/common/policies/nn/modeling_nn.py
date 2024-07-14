from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from huggingface_hub import PyTorchModelHubMixin

from lerobot.common.policies.nn.configuration_nn import NNPolicyConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import populate_queues


class NNPolicy(nn.Module, PyTorchModelHubMixin):
    """
    Defining the nearest-neighbor model
    """

    name = "nn"

    def __init__(
        self,
        config: NNPolicyConfig | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = NNPolicyConfig()
        self.config = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.num_images = len(self.expected_image_keys)

        if self.num_images:
            raise NotImplementedError("NNPolicy does not support image inputs yet.")

        self.database = NNDatabase(config)
        self.reset()

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        queues are populated during rollout of the policy, they contain the n latest observations and actions
        """
        self._queues = {
            "observation.images": deque(maxlen=self.config.n_obs_steps),
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.action_chunk_size),
        }

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        batch = self.normalize_inputs(batch)
        if self.num_images:
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.database(batch, rollout=True)[:, : self.config.action_chunk_size]

            # the dimension of returned action is (batch_size, action_chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            # since the data in the action queue's dimension is (action_chunk_size, batch_size, action_dim), we transpose the action and fill the queue
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        # VQ-BeT discretizes action using VQ-VAE before training BeT (please refer to section 3.2 in the VQ-BeT paper https://arxiv.org/pdf/2403.03181)
        # if Residual VQ is already trained, VQ-BeT trains its GPT and bin prediction head / offset prediction head parts.
        loss_dict = self.database(batch, rollout=False)

        return loss_dict


class NNDatabase(torch.nn.Module):
    def __init__(self, config: NNPolicyConfig):
        super().__init__()
        self.config = config
        self.observation_size = config.input_shapes["observation.state"]
        self.action_size = config.output_shapes["action"]
        self.register_buffer(
            "database",
            torch.zeros(
                (config.database_size, config.n_obs_steps * np.prod(config.input_shapes["observation.state"]))
            ),
        )
        self.register_buffer(
            "database_labels",
            torch.zeros(
                (config.database_size, config.action_chunk_size * np.prod(config.output_shapes["action"]))
            ),
        )
        self.register_buffer("database_occupied", torch.zeros(config.database_size, dtype=bool))
        self.register_buffer("database_pointer", torch.zeros(1, dtype=int))
        self.database_size = config.database_size
        self.distance_metric = config.distance_metric

    def forward(self, batch: dict[str, torch.Tensor], rollout: bool = False) -> torch.Tensor:
        """
        Args:
            batch: Batch of observations to query the database with.

        Returns:
            torch.Tensor: The action corresponding to the nearest observation in the database.
        """
        if not rollout:
            self.update_database(batch)
            return {"loss": torch.tensor(0.0, requires_grad=True, device=batch["observation.state"].device)}
        else:
            distance, actions = self.query_database(batch, self.config.topk)
            actions = NNDatabase.sample(distance, actions, self.config.action_sample_mode)
            # Reshape the actions before sending back.
            actions = actions.view(-1, self.config.action_chunk_size, *self.action_size)
            return actions

    def update_database(self, batch: dict[str, torch.Tensor]):
        """
        Update the database with the latest observations and actions.

        Args:
            batch: Batch of observations to update the database with.
        """
        # Insert a batch of experiences in the database.
        batch_size = batch["observation.state"].shape[0]
        observations = batch["observation.state"].view(batch_size, -1)
        actions = batch["action"].view(batch_size, -1)
        db_pointer = self.database_pointer.item()
        if self.database_size < db_pointer + batch_size:
            # Split it in two halves and enter them.
            self.database_occupied[db_pointer:] = True
            self.database_occupied[: (db_pointer + batch_size) % self.database_size] = True

            last_entries = self.database_size - db_pointer
            first_entries = batch_size - last_entries

            self.database[db_pointer:] = observations[:last_entries]
            self.database[:first_entries] = observations[last_entries:]
            self.database_labels[db_pointer:] = actions[:last_entries]
            self.database_labels[:first_entries] = actions[last_entries:]

        else:
            self.database_occupied[db_pointer : db_pointer + batch_size] = True
            self.database[db_pointer : db_pointer + batch_size] = observations
            self.database_labels[db_pointer : db_pointer + batch_size] = actions

        self.database_pointer = (self.database_pointer + batch_size) % self.database_size

    def query_database(self, batch: dict[str, torch.Tensor], topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the database with the latest observations and actions.

        Args:
            batch: Batch of observations to query the database with.

        Returns:
            torch.Tensor: The action corresponding to the nearest observation in the database.
        """
        batch_size = batch["observation.state"].shape[0]
        observations = batch["observation.state"].view(batch_size, -1)

        # Compute the distance between the observations and the database.
        distances = torch.cdist(observations, self.database[self.database_occupied], p=2)

        # Find the topk nearest neighbors.
        values, indices = torch.topk(distances, topk, largest=False)

        # now, choose the action from the topk nearest neighbors
        actions = self.database_labels[self.database_occupied][indices]

        return values, actions

    @staticmethod
    def sample(distances: torch.Tensor, actions: torch.Tensor, sample_mode: str) -> torch.Tensor:
        """
        Sample an action from the topk nearest neighbors.

        Args:
            distances: The distances of the topk nearest neighbors. (batch_size, topk)
            actions: The actions corresponding to the topk nearest neighbors. (batch_size, topk, action_dim)
            sample_mode: The mode to sample the action.

        Returns:
            torch.Tensor: The sampled action.
        """
        probs = torch.exp(-distances)
        if sample_mode == "sample":
            # Sample an action from the topk nearest neighbors based an exponential
            # distribution based on the distances.
            return actions[torch.multinomial(probs, 1).squeeze()]

        elif sample_mode == "weighted_avg":
            # Compute the weights for the weighted average.
            weights = probs
            weights /= weights.sum()

            # Compute the weighted average.
            return (weights.unsqueeze(-1) * actions).sum(dim=1)
        else:
            raise ValueError(f"Invalid sample mode {sample_mode}.")


class NNOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        # Initialize with empty parameter list
        defaults = {"lr": 0.0}
        super(NNOptimizer, self).__init__(nn.ParameterList([torch.tensor(0, dtype=torch.float32)]), defaults)

    def step(self, closure=None):
        # No-op
        if closure is not None:
            closure()

    def zero_grad(self, set_to_none: bool = False):
        # No-op
        pass
