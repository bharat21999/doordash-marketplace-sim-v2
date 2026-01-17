from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from experiments import choose_dispatch_variant, incentive_effects, is_peak_dinner
from sim_config import SimConfig
from sim_entities import dist_km, make_consumers, make_dashers, make_merchants


def intensity_multiplier(minute_in_day: int, cfg: SimConfig) -> float:
    hour = minute_in_day / 60.0
    if 11 <= hour < 14:
        return cfg.lunch_multiplier
    if 17 <= hour < 21:
        return cfg.dinner_multiplier
    if hour < 6:
        return cfg.late_night_multiplier
    return 1.0


def choose_merchant(
    rng: np.random.Generator, consumer_row: pd.Series, merchants_df: pd.DataFrame, cfg: SimConfig
) -> pd.Series:
    dx = merchants_df["x"].to_numpy() - consumer_row["x"]
    dy = merchants_df["y"].to_numpy() - consumer_row["y"]
    dists_km = np.hypot(dx, dy) * cfg.km_per_unit
    scores = merchants_df["quality"].to_numpy() / (dists_km + 0.2)
    probs = scores / scores.sum()
    idx = rng.choice(merchants_df.index.to_numpy(), p=probs)
    return merchants_df.loc[idx]


def estimate_eta(
    rng: np.random.Generator, minute: int, consumer_row: pd.Series, merchant_row: pd.Series, cfg: SimConfig
) -> float:
    travel_to_consumer = dist_km(
        merchant_row["x"],
        merchant_row["y"],
        consumer_row["x"],
        consumer_row["y"],
        cfg.km_per_unit,
    ) / cfg.speed_km_per_min
    avg_pickup = 6.0
    prep_est = float(merchant_row["prep_mean"])
    eta_noise = float(rng.normal(0.0, cfg.travel_time_noise_min))
    return max(8.0, avg_pickup + prep_est + travel_to_consumer + eta_noise)


def match_dasher(
    minute: int,
    merchant_row: pd.Series,
    consumer_row: pd.Series,
    dashers_df: pd.DataFrame,
    cfg: SimConfig,
    policy: str,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    available = dashers_df[dashers_df["available_at_min"] <= minute]
    if available.empty:
        return None, None, None
    dx = available["x"].to_numpy() - merchant_row["x"]
    dy = available["y"].to_numpy() - merchant_row["y"]
    dists_km = np.hypot(dx, dy) * cfg.km_per_unit
    within = dists_km <= cfg.matching_radius_km
    if not np.any(within):
        return None, None, None

    available_idx = available.index.to_numpy()[within]
    dists_km = dists_km[within]
    travel_to_consumer = dist_km(
        merchant_row["x"],
        merchant_row["y"],
        consumer_row["x"],
        consumer_row["y"],
        cfg.km_per_unit,
    ) / cfg.speed_km_per_min

    if policy == "min_eta":
        travel_to_merchant = dists_km / cfg.speed_km_per_min
        arrival = minute + travel_to_merchant
        expected_wait = np.maximum(0.0, merchant_row["available_at_min"] - arrival)
        predicted_eta = travel_to_merchant + expected_wait + merchant_row["prep_mean"] + travel_to_consumer
        idx_local = int(np.argmin(predicted_eta))
        chosen_idx = int(available_idx[idx_local])
        return chosen_idx, float(travel_to_merchant[idx_local]), float(expected_wait[idx_local])

    idx_local = int(np.argmin(dists_km))
    chosen_idx = int(available_idx[idx_local])
    travel_to_merchant = float(dists_km[idx_local] / cfg.speed_km_per_min)
    return chosen_idx, travel_to_merchant, 0.0


def run_sim(cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    consumers = make_consumers(cfg, rng)
    merchants = make_merchants(cfg, rng)
    dashers = make_dashers(cfg, rng)

    orders = []
    events = []

    order_id = 0
    session_id = 0
    checkout_id = 0

    total_minutes = cfg.minutes_per_day * cfg.days

    for minute in range(total_minutes):
        minute_in_day = minute % cfg.minutes_per_day
        session_rate = cfg.base_session_rate_per_min * intensity_multiplier(minute_in_day, cfg)
        session_arrivals = rng.poisson(session_rate)

        baseline_repeat = consumers["baseline_repeat_propensity"].to_numpy()
        satisfaction = consumers["satisfaction_score"].to_numpy()
        weights = baseline_repeat * (0.6 + satisfaction)
        weights = weights / weights.sum()

        for _ in range(session_arrivals):
            consumer_idx = int(rng.choice(consumers.index.to_numpy(), p=weights))
            consumer = consumers.loc[consumer_idx]
            merchant = choose_merchant(rng, consumer, merchants, cfg)

            promo_amount = cfg.treatment_promo_amount if consumer["variant"] == "treatment" else 0.0
            fee_shown = cfg.delivery_fee + cfg.service_fee
            eta_estimate = estimate_eta(rng, minute, consumer, merchant, cfg)

            events.append(
                {
                    "session_id": session_id,
                    "event_type": "session_start",
                    "event_minute": minute,
                    "consumer_id": int(consumer["consumer_id"]),
                    "variant": consumer["variant"],
                    "promo_amount": promo_amount,
                    "fee_shown": fee_shown,
                    "eta_shown_estimate_min": eta_estimate,
                }
            )

            segment = consumer["segment"]
            base_checkout = {
                "price_sensitive": cfg.checkout_base_price_sensitive,
                "convenience": cfg.checkout_base_convenience,
                "loyal": cfg.checkout_base_loyal,
            }[segment]
            checkout_prob = base_checkout
            if promo_amount > 0:
                checkout_prob += cfg.promo_checkout_uplift
            checkout_prob -= cfg.fee_sensitivity * fee_shown
            checkout_prob = float(np.clip(checkout_prob, 0.02, 0.9))

            if rng.random() < checkout_prob:
                events.append(
                    {
                        "session_id": session_id,
                        "checkout_id": checkout_id,
                        "event_type": "checkout_start",
                        "event_minute": minute,
                        "consumer_id": int(consumer["consumer_id"]),
                        "variant": consumer["variant"],
                        "promo_amount": promo_amount,
                        "fee_shown": fee_shown,
                        "eta_shown_estimate_min": eta_estimate,
                    }
                )

                basket_estimate = float(
                    np.clip(
                        rng.lognormal(cfg.basket_lognorm_mean, cfg.basket_lognorm_sigma),
                        cfg.basket_min,
                        cfg.basket_max,
                    )
                )
                total_price = max(1.0, basket_estimate + fee_shown - promo_amount)
                order_prob = cfg.order_base_prob
                order_prob -= cfg.eta_sensitivity * (eta_estimate / 10.0)
                order_prob -= cfg.price_sensitivity * (total_price / 10.0)
                order_prob = float(np.clip(order_prob, 0.02, 0.9))

                if rng.random() < order_prob:
                    dispatch_variant = choose_dispatch_variant(rng, cfg)
                    incentive_variant = (
                        "treatment" if (cfg.enable_dasher_incentive and rng.random() < 0.5) else "control"
                    )

                    assign_delay = int(rng.integers(1, cfg.assign_delay_min + 1))
                    events.append(
                        {
                            "order_id": order_id,
                            "session_id": session_id,
                            "checkout_id": checkout_id,
                            "event_type": "order_created",
                            "event_minute": minute,
                            "consumer_id": int(consumer["consumer_id"]),
                            "variant": consumer["variant"],
                        }
                    )

                    dasher_idx, travel_to_merchant, expected_wait = match_dasher(
                        minute,
                        merchant,
                        consumer,
                        dashers,
                        cfg,
                        dispatch_variant,
                    )
                    assigned = dasher_idx is not None

                    travel_to_consumer = dist_km(
                        merchant["x"],
                        merchant["y"],
                        consumer["x"],
                        consumer["y"],
                        cfg.km_per_unit,
                    ) / cfg.speed_km_per_min

                    queue_delay = max(0.0, merchant["available_at_min"] - minute) / max(
                        1, merchant["capacity"]
                    )
                    prep_time = float(
                        np.clip(
                            rng.normal(merchant["prep_mean"], cfg.prep_std_min),
                            cfg.prep_min_min,
                            cfg.prep_max_min,
                        )
                    )
                    eta_noise = float(rng.normal(0.0, cfg.travel_time_noise_min))
                    eta_shown = (
                        max(5.0, travel_to_merchant or 0.0)
                        + queue_delay
                        + prep_time
                        + travel_to_consumer
                    )
                    eta_shown = max(5.0, eta_shown + eta_noise)

                    cancel_prob = (
                        cfg.base_cancel_prob
                        + cfg.eta_cancel_slope * (eta_shown / 10.0)
                        + 0.04 * float(consumer["impatience"])
                    )
                    if promo_amount > 0:
                        cancel_prob = max(0.0, cancel_prob - cfg.promo_cancel_reduction)
                    cancel_prob = float(np.clip(cancel_prob, 0.01, 0.8))

                    delivered = False
                    cancel_reason = None
                    actual_delivery_time = None
                    assignment_latency = None

                    if assigned:
                        base_accept = float(
                            dashers.loc[dasher_idx, "base_accept_prob"]
                            - cfg.accept_time_slope * travel_to_merchant
                        )
                        accept_prob, incentive_amount, availability_boost = incentive_effects(
                            base_accept,
                            cfg,
                            incentive_variant == "treatment",
                            minute,
                        )
                        accept_prob = float(np.clip(accept_prob, 0.05, 0.98))
                        accepted = rng.random() < accept_prob
                    else:
                        accepted = False
                        incentive_amount = 0.0
                        availability_boost = 1.0

                    if assigned and accepted:
                        assignment_latency = float(assign_delay)
                        events.append(
                            {
                                "order_id": order_id,
                                "event_type": "assigned",
                                "event_minute": minute + assign_delay,
                                "dispatch_policy": dispatch_variant,
                            }
                        )

                        canceled = rng.random() < cancel_prob
                        if canceled:
                            cancel_reason = "consumer_cancel"
                            events.append(
                                {
                                    "order_id": order_id,
                                    "event_type": "canceled",
                                    "event_minute": minute + assign_delay,
                                }
                            )
                            dashers.loc[dasher_idx, "available_at_min"] = minute + assign_delay
                        else:
                            arrival_at_merchant = minute + travel_to_merchant
                            start_prep = minute + queue_delay
                            ready_time = start_prep + prep_time
                            pickup_time = max(arrival_at_merchant, ready_time)
                            actual_delivery_time = pickup_time + travel_to_consumer
                            delivered = True

                            dashers.loc[dasher_idx, "available_at_min"] = minute + (
                                (actual_delivery_time - minute) * availability_boost
                            )
                            dashers.loc[dasher_idx, "x"] = consumer["x"]
                            dashers.loc[dasher_idx, "y"] = consumer["y"]
                            merchants.loc[merchant.name, "available_at_min"] = ready_time

                            events.append(
                                {
                                    "order_id": order_id,
                                    "event_type": "delivered",
                                    "event_minute": actual_delivery_time,
                                }
                            )
                    else:
                        cancel_reason = "no_dasher" if not assigned else "dasher_reject"
                        events.append(
                            {
                                "order_id": order_id,
                                "event_type": "canceled",
                                "event_minute": minute + assign_delay,
                            }
                        )

                    basket = float(
                        np.clip(
                            rng.lognormal(cfg.basket_lognorm_mean, cfg.basket_lognorm_sigma),
                            cfg.basket_min,
                            cfg.basket_max,
                        )
                    )
                    revenue = basket * cfg.take_rate + cfg.delivery_fee
                    promo_cost = promo_amount
                    dasher_cost = (cfg.base_dasher_pay + incentive_amount) if delivered else 0.0
                    profit = revenue - promo_cost - dasher_cost if delivered else -promo_cost

                    delivery_time = actual_delivery_time - minute if delivered else None

                    if delivered and delivery_time is not None:
                        late = (delivery_time > eta_shown + cfg.late_tolerance_min) or (
                            delivery_time > cfg.late_threshold_min
                        )
                        current_score = consumers.loc[consumer_idx, "satisfaction_score"]
                        if late:
                            new_score = max(cfg.satisfaction_floor, current_score - 0.08)
                        else:
                            new_score = min(cfg.satisfaction_ceiling, current_score + 0.03)
                        consumers.loc[consumer_idx, "satisfaction_score"] = new_score

                    orders.append(
                        {
                            "order_id": order_id,
                            "session_id": session_id,
                            "checkout_id": checkout_id,
                            "created_minute": minute,
                            "day": minute // cfg.minutes_per_day,
                            "consumer_id": int(consumer["consumer_id"]),
                            "segment": consumer["segment"],
                            "merchant_id": int(merchant["merchant_id"]),
                            "dasher_id": int(dashers.loc[dasher_idx, "dasher_id"]) if assigned else None,
                            "variant": consumer["variant"],
                            "promo_amount": promo_amount,
                            "incentive_variant": incentive_variant,
                            "dispatch_policy": dispatch_variant,
                            "basket_size": basket,
                            "eta_shown_min": eta_shown,
                            "delivery_time_min": delivery_time,
                            "assignment_latency_min": assignment_latency,
                            "delivered": delivered,
                            "canceled": not delivered,
                            "cancel_reason": cancel_reason,
                            "travel_to_merchant_min": travel_to_merchant,
                            "expected_wait_min": expected_wait,
                            "prep_time_min": prep_time,
                            "travel_to_consumer_min": travel_to_consumer,
                            "revenue": revenue,
                            "promo_cost": promo_cost,
                            "dasher_cost": dasher_cost,
                            "profit": profit,
                            "is_peak": is_peak_dinner(minute, cfg),
                        }
                    )
                    order_id += 1

                checkout_id += 1

            session_id += 1

    orders_df = pd.DataFrame(orders)
    events_df = pd.DataFrame(events)

    if not orders_df.empty:
        first_order_day = orders_df.groupby("consumer_id")["day"].min().rename("first_order_day")
        last_order_day = orders_df.groupby("consumer_id")["day"].max().rename("last_order_day")
        cohorts = pd.concat([first_order_day, last_order_day], axis=1)
        cohorts["repeat_within_7d"] = (cohorts["last_order_day"] - cohorts["first_order_day"]) >= 1
        orders_df = orders_df.merge(cohorts, on="consumer_id", how="left")

    return orders_df, events_df


def main() -> None:
    cfg = SimConfig()
    orders_df, events_df = run_sim(cfg)

    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    orders_df.to_parquet("data/orders.parquet", index=False)
    events_df.to_parquet("data/events.parquet", index=False)

    delivered = int(orders_df["delivered"].sum())
    canceled = int(orders_df["canceled"].sum())
    print(f"Total orders: {len(orders_df)}")
    print(f"Delivered: {delivered}")
    print(f"Canceled: {canceled}")


if __name__ == "__main__":
    main()
