from __future__ import annotations

from typing import Tuple

import numpy as np

from sim_config import SimConfig


def is_peak_dinner(minute: int, cfg: SimConfig) -> bool:
    hour = (minute % cfg.minutes_per_day) / 60.0
    return 17 <= hour < 21


def choose_dispatch_variant(rng: np.random.Generator, cfg: SimConfig) -> str:
    if not cfg.enable_dispatch_experiment:
        return cfg.dispatch_policy_default
    return "nearest" if rng.random() < 0.5 else "min_eta"


def incentive_effects(
    base_accept_prob: float,
    cfg: SimConfig,
    treatment: bool,
    minute: int,
) -> Tuple[float, float, float]:
    if not (cfg.enable_dasher_incentive and treatment and is_peak_dinner(minute, cfg)):
        return base_accept_prob, 0.0, 1.0
    boosted_accept = min(0.98, base_accept_prob + cfg.incentive_accept_uplift)
    return boosted_accept, cfg.incentive_amount, cfg.incentive_availability_boost
