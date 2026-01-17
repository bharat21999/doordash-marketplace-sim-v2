from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # Reproducibility
    seed: int = 42

    # Time and city
    minutes_per_day: int = 24 * 60
    days: int = 7
    city_size_units: float = 10.0
    km_per_unit: float = 0.5
    speed_km_per_min: float = 0.5  # ~30 km/h
    travel_time_noise_min: float = 2.0

    # Population sizes
    n_consumers: int = 5000
    n_merchants: int = 250
    n_dashers: int = 900

    # Demand (orders)
    base_arrival_rate_per_min: float = 2.6
    lunch_multiplier: float = 1.6
    dinner_multiplier: float = 1.9
    late_night_multiplier: float = 0.6

    # Demand (sessions)
    base_session_rate_per_min: float = 6.5
    checkout_base_price_sensitive: float = 0.18
    checkout_base_convenience: float = 0.28
    checkout_base_loyal: float = 0.34
    promo_checkout_uplift: float = 0.05
    fee_sensitivity: float = 0.04

    order_base_prob: float = 0.55
    eta_sensitivity: float = 0.018
    price_sensitivity: float = 0.015

    service_fee: float = 1.99

    # Consumers
    p_price_sensitive: float = 0.5
    p_convenience: float = 0.3
    p_loyal: float = 0.2

    # Merchants
    prep_mean_min: float = 18.0
    prep_std_min: float = 5.0
    prep_min_min: float = 6.0
    prep_max_min: float = 45.0

    # Dashers and matching
    dasher_accept_mean: float = 0.8
    dasher_accept_std: float = 0.1
    accept_time_slope: float = 0.03
    matching_radius_km: float = 5.0
    assign_delay_min: int = 3
    dispatch_policy_default: str = "nearest"

    # Cancellations
    base_cancel_prob: float = 0.03
    eta_cancel_slope: float = 0.01
    promo_cancel_reduction: float = 0.01

    # Economics
    take_rate: float = 0.22
    delivery_fee: float = 3.49
    base_dasher_pay: float = 4.25
    basket_lognorm_mean: float = 3.1
    basket_lognorm_sigma: float = 0.5
    basket_min: float = 8.0
    basket_max: float = 80.0

    # Experiments
    experiment_name: str = "promo_5_off"
    treatment_promo_amount: float = 5.0
    enable_dasher_incentive: bool = True
    incentive_amount: float = 1.5
    incentive_accept_uplift: float = 0.08
    incentive_availability_boost: float = 0.95
    enable_dispatch_experiment: bool = True

    # Retention
    late_threshold_min: float = 40.0
    late_tolerance_min: float = 6.0
    satisfaction_floor: float = 0.1
    satisfaction_ceiling: float = 1.0
