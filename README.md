# DeFi Yield Optimization: Economic Framework

Economic analysis of optimal capital allocation across DeFi lending protocols.

## Background

This repository documents the economic framework behind YIELD, an AI-powered cross-chain 
yield optimizer that won showcase awards at:
- ?? EthGlobal San Francisco 2024
- ?? EthGlobal Cannes 2024

**Scope**: This repo focuses on the **economic model and decision logic** I designed, 
implemented here in Python for research/pedagogical purposes. The original hackathon 
implementation was a team effort across multiple languages.

## Core Economic Problem

**Objective**: Maximize net yield across N lending protocols on M chains

**Tradeoff**: Higher APY protocols may be on expensive chains (high gas costs)

**Decision**: When to rebalance? Benefit (yield delta) vs. Cost (gas + slippage)

## Model Components

1. **Yield calculation**: APY_net = APY_gross - (gas_cost / capital)
2. **Rebalancing threshold**: Only move capital if yield_delta > threshold
3. **Multi-agent coordination**: Monitor agents (track rates) + Executor agent (transactions)

## Repository Contents

- docs/architecture.md: Agent decision framework
- docs/yield_calculation.md: Mathematical formulation
- notebooks/01_simulation.ipynb: Python implementation with toy data
- papers/ethglobal_summary.pdf: Project overview and recognition

## Tech Stack

- Python (NumPy, pandas for simulations)
- Jupyter notebooks for analysis

## Status

Work in progress - documenting economic logic from hackathon project
