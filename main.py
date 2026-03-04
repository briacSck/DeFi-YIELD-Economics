# main.py — DeFi-YIELD-Economics Pipeline Orchestrator
"""
Run order:
  1. Collect fresh APY data
  2. Build/update panel dataset
  3. Score protocols (risk)
  4. Run capital-at-risk analysis
  5. Run liquidity coverage analysis
  6. Forecast yields (ARIMA/LSTM/XGBoost)
  7. Run risk-adjusted optimization
  8. Run rebalancing backtest
  9. Export summary report
"""

import argparse
import logging
from pathlib import Path

from data.collect_apy_data import collect_apy_data
from data.build_timeseries import build_timeseries
from risk.protocol_scoring import score_protocols
from risk.capital_at_risk import calculate_portfolio_car
from risk.liquidity_coverage import calculate_funding_gap
from models.yield_forecasting import run_forecasting_suite
from risk.risk_adjusted_optimization import optimize_with_calm_constraints
from models.rebalancing_optimization import run_backtests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("YIELD-Pipeline")

def run_pipeline(
    deposit_sizes=[100, 500, 1000, 5000, 10000],
    collect_fresh: bool = True,
    skip_lstm: bool = False,      # LSTM is slow; skip for quick runs
):
    log.info("=== DeFi YIELD Economics Pipeline ===")

    # --- Stage 1: Data ---
    if collect_fresh:
        log.info("Stage 1a: Collecting APY data from DeFiLlama...")
        collect_apy_data()
    log.info("Stage 1b: Building panel dataset...")
    panel = build_timeseries()

    # --- Stage 2: Risk Scoring ---
    log.info("Stage 2a: Scoring protocols (CaR, credit, operational)...")
    protocol_scores = score_protocols(panel)

    log.info("Stage 2b: Capital-at-Risk analysis...")
    car_results = calculate_portfolio_car(protocol_scores)
    car_results.to_csv("risk/car_analysis_results.csv", index=False)

    log.info("Stage 2c: Liquidity Coverage Ratio analysis...")
    lcr_results = calculate_funding_gap(panel)
    lcr_results.to_csv("risk/liquidity_coverage_results.csv", index=False)

    # --- Stage 3: Forecasting ---
    log.info("Stage 3: Running forecasting suite (ARIMA, XGBoost, LSTM)...")
    forecasts = run_forecasting_suite(panel, skip_lstm=skip_lstm)
    forecasts.to_csv("outputs/forecast_results.csv", index=False)

    # --- Stage 4: Optimization ---
    log.info("Stage 4: Risk-adjusted CALM portfolio optimization...")
    optimal_alloc = optimize_with_calm_constraints(forecasts, car_results, lcr_results)
    optimal_alloc.to_csv("risk/optimal_allocation.csv", index=False)

    # --- Stage 5: Backtesting ---
    log.info("Stage 5: Rebalancing backtests across deposit sizes...")
    backtest_results = run_backtests(
        optimal_alloc, panel,
        deposit_sizes=deposit_sizes
    )
    backtest_results.to_csv("outputs/backtest_results.csv", index=False)

    log.info("=== Pipeline complete. Check /outputs and /risk for results. ===")
    return backtest_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeFi YIELD Economics Pipeline")
    parser.add_argument("--no-collect", action="store_true", help="Skip live data collection")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip slow LSTM training")
    args = parser.parse_args()

    run_pipeline(
        collect_fresh=not args.no_collect,
        skip_lstm=args.skip_lstm,
    )