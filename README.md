# DoorDash-Style Marketplace Simulator

A reproducible, interview-ready simulator for a two-sided delivery marketplace (Consumers, Merchants, Dashers) built in pure Python + pandas. It generates session-to-order funnel data, order lifecycle events, and unit economics, then runs analytics to produce KPIs, experiment readouts, and retention insights.

## What this project is
A compact yet realistic marketplace analytics sandbox for modeling demand, dispatch, ETAs, cancellations, pricing, and incentives. It is designed to mimic the types of analyses done in Marketplace data science.

## Key business questions answered
- Where do we lose users in the funnel (session → checkout → order → delivered)?
- How do ETA and fees affect conversion and cancellation?
- Which dispatch policy minimizes delivery time without hurting profit?
- Do promos and dasher incentives improve outcomes or degrade unit economics?
- How does delivery quality influence short-term retention?

## How the simulation works (high level)
- Minute-by-minute sessions with time-of-day spikes (lunch/dinner).
- Session → checkout conversion depends on segment, promo, and fees.
- Checkout → order conversion depends on ETA estimate and total price.
- Orders are matched to dashers using A/B dispatch policies.
- Dasher incentive experiment during dinner peak affects acceptance and availability.
- ETA-driven cancellation and unit economics for each order.
- Retention proxy labels based on delivery quality and repeat behavior.

## Experiments
- **Promo A/B**: $0 vs $5 off (consumer-level).
- **Dasher incentive**: peak-hour treatment adds extra pay and boosts acceptance.
- **Dispatch policy A/B**: nearest-first vs minimize-ETA matching.

## Outputs
Generated in `data/` and `reports/`.

### Orders data (`data/orders.parquet`)
One row per order with outcomes and unit economics.

Columns include:
- Identifiers: `order_id`, `session_id`, `checkout_id`, `consumer_id`, `merchant_id`, `dasher_id`
- Timing: `created_minute`, `day`, `eta_shown_min`, `delivery_time_min`, `assignment_latency_min`
- Experiments: `variant`, `promo_amount`, `incentive_variant`, `dispatch_policy`, `is_peak`
- Outcomes: `delivered`, `canceled`, `cancel_reason`, `first_order_day`, `repeat_within_7d`
- Economics: `basket_size`, `revenue`, `promo_cost`, `dasher_cost`, `profit`

### Events data (`data/events.parquet`)
Event log with full funnel and lifecycle events.

Event types:
- `session_start`
- `checkout_start`
- `order_created`
- `assigned`
- `delivered`
- `canceled`

## Metrics reported
- Core KPIs: delivered rate, cancel rate, p90 delivery time, profit
- Funnel conversion (session → checkout → order → delivered)
- ETA impact on cancellation by variant
- Experiment summaries (promo, incentive, dispatch)
- Cohort retention by first order day and delivery quality
- Logistic regression on cancellation drivers

## Visualizations (saved to `reports/`)
- `funnel_bar_by_variant.png`
- `cancel_vs_eta_by_variant.png`
- `delivery_time_p90_by_variant.png`
- `profit_by_variant.png`
- `cohort_retention_heatmap_like_plot.png`

## How to run
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/sim_run.py
python src/analytics.py
```

## Config options
Key parameters in `src/sim_config.py`:
- `days`, `minutes_per_day`, `base_session_rate_per_min`
- `dispatch_policy_default`, `enable_dispatch_experiment`
- `enable_dasher_incentive`, `incentive_amount`
- `treatment_promo_amount`

## Key findings
- Promo increases conversion but can reduce per-order profit.
- Min-ETA dispatch reduces p90 delivery time.
- Late deliveries lower 7-day repeat rates.

## Limitations and next steps
- Simplified dispatch and merchant operations.
- No item-level catalog or inventory constraints.
- No explicit dasher supply elasticity or surge pricing.

Next steps:
- Add session → checkout → payment drop-off variants
- Model incentive elasticity and dasher churn
- Introduce batching and priority tiers in dispatch
- Extend to multi-week cohort retention
