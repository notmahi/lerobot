from dataclasses import dataclass, field


@dataclass
class NNPolicyConfig:
    """
    Configuration for nearest-neighbor policy.

    Defaults are configured for training with the maze-ball environment.
    """

    # Inputs / output structure.
    n_obs_steps: int = 1
    action_chunk_size: int = 2

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.state": [4],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [2],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.state": "mean_std",
        }
    )
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    distance_metric: str = "euclidean"
    database_size: int = 10_000
    topk: int = 5

    # Possible modes: weighted_avg, sample
    action_sample_mode: str = "sample"
