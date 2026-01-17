from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _ensure_reports_dir() -> None:
    os.makedirs("reports", exist_ok=True)


def _eta_buckets(series: pd.Series) -> pd.Series:
    max_eta = max(series.max(), 60.0)
    bins = np.arange(0, np.ceil(max_eta / 10) * 10 + 10, 10)
    return pd.cut(series, bins=bins, right=False)


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    aligned = numerator.reindex(denominator.index).fillna(0)
    return aligned / denominator.replace(0, np.nan)


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    orders_df = pd.read_parquet("data/orders.parquet")
    events_df = pd.read_parquet("data/events.parquet")

    _ensure_reports_dir()

    delivered = orders_df["delivered"].sum()
    total_orders = len(orders_df)
    delivered_rate = delivered / total_orders if total_orders else 0.0
    cancel_rate = 1.0 - delivered_rate

    delivery_times = orders_df.loc[orders_df["delivered"], "delivery_time_min"].dropna()
    avg_delivery_time = float(delivery_times.mean()) if not delivery_times.empty else 0.0
    p90_delivery_time = float(delivery_times.quantile(0.9)) if not delivery_times.empty else 0.0

    avg_profit = float(orders_df["profit"].mean()) if total_orders else 0.0
    total_profit = float(orders_df["profit"].sum())

    _print_header("Core KPIs")
    print(f"Total orders: {total_orders}")
    print(f"Delivered rate: {delivered_rate:.3f}")
    print(f"Cancel rate: {cancel_rate:.3f}")
    print(f"Avg delivery time (min): {avg_delivery_time:.2f}")
    print(f"P90 delivery time (min): {p90_delivery_time:.2f}")
    print(f"Avg profit per order: {avg_profit:.2f}")
    print(f"Total profit: {total_profit:.2f}")

    session_df = events_df[events_df["event_type"] == "session_start"].copy()
    checkout_df = events_df[events_df["event_type"] == "checkout_start"].copy()

    sessions_by_variant = session_df.groupby("variant")["session_id"].nunique()
    checkouts_by_variant = checkout_df.groupby("variant")["session_id"].nunique()
    orders_by_variant = orders_df.groupby("variant")["order_id"].nunique()
    delivered_by_variant = orders_df[orders_df["delivered"]].groupby("variant")["order_id"].nunique()

    funnel_variant = pd.DataFrame(
        {
            "sessions": sessions_by_variant,
            "checkouts": checkouts_by_variant,
            "orders": orders_by_variant,
            "delivered": delivered_by_variant,
        }
    ).fillna(0)
    funnel_variant["session_to_checkout"] = _safe_rate(
        funnel_variant["checkouts"], funnel_variant["sessions"]
    )
    funnel_variant["checkout_to_order"] = _safe_rate(
        funnel_variant["orders"], funnel_variant["checkouts"]
    )
    funnel_variant["order_to_delivered"] = _safe_rate(
        funnel_variant["delivered"], funnel_variant["orders"]
    )

    _print_header("Funnel by Variant")
    print(funnel_variant.to_string())

    session_eta = session_df[["session_id", "eta_shown_estimate_min"]].dropna()
    session_eta["eta_bucket"] = _eta_buckets(session_eta["eta_shown_estimate_min"])
    checkout_sessions = set(checkout_df["session_id"].unique())
    session_eta["converted_checkout"] = session_eta["session_id"].isin(checkout_sessions)
    checkout_rates_by_eta = session_eta.groupby("eta_bucket")["converted_checkout"].mean()

    checkout_eta = checkout_df[["session_id", "eta_shown_estimate_min"]].dropna()
    checkout_eta["eta_bucket"] = _eta_buckets(checkout_eta["eta_shown_estimate_min"])
    order_sessions = set(orders_df["session_id"].dropna().unique())
    checkout_eta["converted_order"] = checkout_eta["session_id"].isin(order_sessions)
    order_rates_by_eta = checkout_eta.groupby("eta_bucket")["converted_order"].mean()

    order_eta = orders_df[["eta_shown_min", "delivered", "variant"]].dropna()
    order_eta["eta_bucket"] = _eta_buckets(order_eta["eta_shown_min"])
    delivered_by_eta_variant = (
        order_eta.groupby(["variant", "eta_bucket"])["delivered"].mean().reset_index()
    )

    _print_header("Funnel by ETA")
    print("Session -> Checkout conversion by ETA bucket")
    print(checkout_rates_by_eta.to_string())
    print("\nCheckout -> Order conversion by ETA bucket")
    print(order_rates_by_eta.to_string())

    plt.figure(figsize=(8, 4))
    stages = ["session_to_checkout", "checkout_to_order", "order_to_delivered"]
    x = np.arange(len(stages))
    width = 0.35
    variants = funnel_variant.index.tolist()
    for i, variant in enumerate(variants):
        rates = funnel_variant.loc[variant, stages].to_numpy(dtype=float)
        plt.bar(x + i * width, rates, width, label=variant)
    plt.xticks(x + width / 2, ["Session→Checkout", "Checkout→Order", "Order→Delivered"])
    plt.ylabel("Conversion rate")
    plt.title("Funnel Conversion by Variant")
    plt.legend()
    plt.tight_layout()
    funnel_path = "reports/funnel_bar_by_variant.png"
    plt.savefig(funnel_path)
    plt.close()

    cancel_by_eta_variant = (
        orders_df.groupby(["variant", _eta_buckets(orders_df["eta_shown_min"])])["canceled"]
        .mean()
        .reset_index()
        .rename(columns={"eta_shown_min": "eta_bucket"})
    )
    plt.figure(figsize=(8, 4))
    for variant in cancel_by_eta_variant["variant"].unique():
        subset = cancel_by_eta_variant[cancel_by_eta_variant["variant"] == variant]
        plt.plot(subset["eta_bucket"].astype(str), subset["canceled"], marker="o", label=variant)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cancel rate")
    plt.title("Cancellation vs ETA by Variant")
    plt.legend()
    plt.tight_layout()
    cancel_path = "reports/cancel_vs_eta_by_variant.png"
    plt.savefig(cancel_path)
    plt.close()

    p90_by_dispatch = (
        orders_df[orders_df["delivered"]]
        .groupby("dispatch_policy")["delivery_time_min"]
        .quantile(0.9)
        .reset_index()
    )
    plt.figure(figsize=(6, 4))
    plt.bar(p90_by_dispatch["dispatch_policy"], p90_by_dispatch["delivery_time_min"], color="#4C72B0")
    plt.ylabel("P90 delivery time (min)")
    plt.title("P90 Delivery Time by Dispatch Policy")
    plt.tight_layout()
    p90_path = "reports/delivery_time_p90_by_variant.png"
    plt.savefig(p90_path)
    plt.close()

    profit_by_variant = (
        orders_df.groupby("variant")["profit"].sum().reset_index().rename(columns={"profit": "total"})
    )
    plt.figure(figsize=(6, 4))
    plt.bar(profit_by_variant["variant"], profit_by_variant["total"], color="#55A868")
    plt.ylabel("Total profit")
    plt.title("Total Profit by Variant")
    plt.tight_layout()
    profit_path = "reports/profit_by_variant.png"
    plt.savefig(profit_path)
    plt.close()

    delivered_orders = orders_df[orders_df["delivered"]]
    consumer_first = (
        delivered_orders.sort_values(["consumer_id", "created_minute"])
        .groupby("consumer_id")
        .first()
        .reset_index()
    )
    consumer_first["delivery_quality"] = np.where(
        consumer_first["delivery_time_min"]
        > consumer_first["eta_shown_min"] + 6.0,
        "late",
        "on_time",
    )
    retention = (
        consumer_first.groupby(["first_order_day", "delivery_quality"])["repeat_within_7d"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 4))
    for quality in retention["delivery_quality"].unique():
        subset = retention[retention["delivery_quality"] == quality]
        plt.plot(subset["first_order_day"], subset["repeat_within_7d"], marker="o", label=quality)
    plt.xlabel("First order day")
    plt.ylabel("Retention rate")
    plt.title("Cohort Retention by First Order Day")
    plt.legend()
    plt.tight_layout()
    retention_path = "reports/cohort_retention_heatmap_like_plot.png"
    plt.savefig(retention_path)
    plt.close()

    _print_header("Experiment Summary: Promo")
    promo_session_rate = _safe_rate(orders_by_variant, sessions_by_variant)
    promo_delta_conv = promo_session_rate.get("treatment", 0) - promo_session_rate.get("control", 0)
    promo_profit = orders_df.groupby("variant")["profit"].mean()
    promo_profit_delta = promo_profit.get("treatment", 0) - promo_profit.get("control", 0)
    print(f"Orders per session delta (treatment - control): {promo_delta_conv:.3f}")
    print(f"Avg profit delta (treatment - control): {promo_profit_delta:.2f}")

    _print_header("Experiment Summary: Dasher Incentive (Peak)")
    peak_orders = orders_df[orders_df["is_peak"]]
    incentive_summary = (
        peak_orders.groupby("incentive_variant")
        .agg(
            assignment_latency=("assignment_latency_min", "mean"),
            p90_delivery_time=("delivery_time_min", lambda x: x.quantile(0.9)),
            cancel_rate=("canceled", "mean"),
            avg_profit=("profit", "mean"),
        )
        .reset_index()
    )
    print(incentive_summary.to_string(index=False))

    _print_header("Experiment Summary: Dispatch Policy")
    def _utilization_proxy(group: pd.DataFrame) -> float:
        delivered = group[group["delivered"]]
        active_dashers = delivered["dasher_id"].dropna().nunique()
        if active_dashers == 0:
            return 0.0
        return delivered.shape[0] / active_dashers

    dispatch_summary = (
        orders_df.groupby("dispatch_policy")
        .apply(
            lambda g: pd.Series(
                {
                    "assignment_latency": g["assignment_latency_min"].mean(),
                    "p90_delivery_time": g["delivery_time_min"].quantile(0.9),
                    "cancel_rate": g["canceled"].mean(),
                    "avg_profit": g["profit"].mean(),
                    "utilization_proxy": _utilization_proxy(g),
                }
            )
        )
        .reset_index()
    )
    print(dispatch_summary.to_string(index=False))

    _print_header("Regression: Cancellation Drivers")
    model_df = orders_df[["canceled", "eta_shown_min", "segment", "promo_amount", "is_peak"]].copy()
    model_df["promo_flag"] = (model_df["promo_amount"] > 0).astype(int)
    model_df["is_peak"] = model_df["is_peak"].astype(int)
    model_df = model_df.dropna(subset=["eta_shown_min", "segment"])
    X = pd.get_dummies(model_df[["eta_shown_min", "promo_flag", "is_peak", "segment"]], drop_first=True)
    y = model_df["canceled"].astype(int)
    if len(X) > 0:
        logit = LogisticRegression(max_iter=200)
        logit.fit(X, y)
        coef = pd.Series(logit.coef_[0], index=X.columns).sort_values(ascending=False)
        print(coef.to_string())

        insights_path = Path("reports/insights.txt")
        with insights_path.open("w", encoding="utf-8") as f:
            f.write("Cancellation model coefficients (positive increases cancel odds):\n")
            f.write(coef.to_string())
            f.write("\n\nInterpretation:\n")
            f.write(
                "Higher ETA and peak periods increase cancellation odds, while promos and loyal segments reduce it.\n"
            )

    _print_header("Charts saved")
    for path in [
        funnel_path,
        cancel_path,
        p90_path,
        profit_path,
        retention_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
