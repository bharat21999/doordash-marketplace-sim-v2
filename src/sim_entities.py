from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from sim_config import SimConfig


def dist_km(x1: float, y1: float, x2: float, y2: float, km_per_unit: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2) * km_per_unit)


def _uniform_xy(rng: np.random.Generator, n: int, city_size_units: float) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0, city_size_units, size=n)
    y = rng.uniform(0, city_size_units, size=n)
    return x, y


def make_consumers(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    x, y = _uniform_xy(rng, cfg.n_consumers, cfg.city_size_units)
    segments = rng.choice(
        ["price_sensitive", "convenience", "loyal"],
        size=cfg.n_consumers,
        p=[cfg.p_price_sensitive, cfg.p_convenience, cfg.p_loyal],
    )
    impatience_map = {
        "price_sensitive": 1.2,
        "convenience": 0.9,
        "loyal": 0.7,
    }
    repeat_propensity_map = {
        "price_sensitive": 0.18,
        "convenience": 0.28,
        "loyal": 0.42,
    }
    impatience = np.vectorize(impatience_map.get)(segments)
    baseline_repeat = np.vectorize(repeat_propensity_map.get)(segments)
    variants = rng.choice(["control", "treatment"], size=cfg.n_consumers)

    return pd.DataFrame(
        {
            "consumer_id": np.arange(cfg.n_consumers),
            "x": x,
            "y": y,
            "segment": segments,
            "impatience": impatience,
            "baseline_repeat_propensity": baseline_repeat,
            "satisfaction_score": np.full(cfg.n_consumers, 0.6),
            "variant": variants,
        }
    )


def make_merchants(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    x, y = _uniform_xy(rng, cfg.n_merchants, cfg.city_size_units)
    prep_mean = rng.normal(cfg.prep_mean_min, cfg.prep_std_min, size=cfg.n_merchants)
    prep_mean = np.clip(prep_mean, cfg.prep_min_min, cfg.prep_max_min)
    capacity = rng.integers(1, 4, size=cfg.n_merchants)
    quality = rng.uniform(0.7, 1.0, size=cfg.n_merchants)

    return pd.DataFrame(
        {
            "merchant_id": np.arange(cfg.n_merchants),
            "x": x,
            "y": y,
            "prep_mean": prep_mean,
            "capacity": capacity,
            "quality": quality,
            "available_at_min": np.zeros(cfg.n_merchants),
        }
    )


def make_dashers(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    x, y = _uniform_xy(rng, cfg.n_dashers, cfg.city_size_units)
    accept_prob = rng.normal(cfg.dasher_accept_mean, cfg.dasher_accept_std, size=cfg.n_dashers)
    accept_prob = np.clip(accept_prob, 0.4, 0.98)

    return pd.DataFrame(
        {
            "dasher_id": np.arange(cfg.n_dashers),
            "x": x,
            "y": y,
            "available_at_min": np.zeros(cfg.n_dashers),
            "base_accept_prob": accept_prob,
        }
    )
