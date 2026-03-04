"""
Risk-Adjusted Portfolio Optimization
Integrates CALM risk scores into allocation decisions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

BASE_DIR = Path(__file__).parent.parent
RISK_DIR = BASE_DIR / 'risk'

class RiskAdjustedOptimizer:
    """Optimize portfolio allocation considering risk constraints"""
    
    def __init__(self, risk_budget=0.15):
        """
        Args:
            risk_budget: Maximum portfolio CaR as % of capital (default 15%)
        """
        self.risk_budget = risk_budget
        self.risk_scores = pd.read_csv(BASE_DIR / 'results' / 'protocol_risk_scores.csv')
        self.timeseries = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'yield_panel.csv')
    
    def optimize_allocation(self, target_protocols=10):
        """
        Maximize risk-adjusted return subject to:
        - Budget constraint: Σw_i = 1
        - Risk constraint: Σw_i * CaR_i ≤ risk_budget
        - Long-only: w_i ≥ 0
        """
        # Get top protocols by Sharpe-like metric
        df = self.risk_scores.nlargest(target_protocols, 'sharpe_proxy').copy()
        n = len(df)
        
        # Objective: Maximize Sharpe (minimize negative Sharpe)
        def objective(weights):
            portfolio_return = np.dot(weights, df['sharpe_proxy'].values)
            
            # Diversification penalty: penalize concentrated portfolios
            # HHI (Herfindahl-Hirschman Index): sum of squared weights
            concentration = np.sum(weights ** 2)
            diversification_penalty = concentration * 10  # Scale factor
    
            return -(portfolio_return - diversification_penalty)  # Maximize return - penalty

        
        # Constraint: Portfolio CaR ≤ risk_budget
        def risk_constraint(weights):
            portfolio_car = np.dot(weights, df['total_carr'].values / 100)
            return self.risk_budget - portfolio_car  # Must be >= 0
        
        # Constraint: Weights sum to 1
        def budget_constraint(weights):
            return np.sum(weights) - 1
        
        # Initial guess: Equal weights
        x0 = np.ones(n) / n
        
        # Bounds: 0 <= w_i <= 0.5 (max 50% in single protocol)
        # Enforce minimum  (preferable for commercial use)
        bounds = tuple((0.05, 0.25) for _ in range(n))  # Min 5%, Max 25% per position

        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': budget_constraint},
            {'type': 'ineq', 'fun': risk_constraint}
        ]
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            df['weight'] = result.x
            df['allocated_capital'] = df['weight'] * 100  # Assume $100 portfolio
            
            portfolio_return = np.dot(result.x, df['sharpe_proxy'].values)
            portfolio_car = np.dot(result.x, df['total_carr'].values / 100)
            
            print("\n=== RISK-ADJUSTED ALLOCATION ===")
            print(f"Portfolio Risk-Adjusted Return: {portfolio_return:.2f}")
            print(f"Portfolio CaR: {portfolio_car*100:.2f}% (Budget: {self.risk_budget*100}%)")
            print(f"\nTop 5 Allocations:")
            print(df.nlargest(5, 'weight')[['protocol_name', 'weight', 'risk_tier', 'total_carr']])
            
            # Save
            output_file = RISK_DIR / 'optimal_allocation.csv'
            df.to_csv(output_file, index=False)
            print(f"\nFull allocation saved: {output_file}")
            
            return df
        else:
            print("Optimization failed:", result.message)
            return None

def main():
    optimizer = RiskAdjustedOptimizer(risk_budget=0.15)
    optimizer.optimize_allocation(target_protocols=20)


# ============================================================================
# MAIN / PUBLIC API
# ============================================================================
# ============================================================================
# MAIN / PUBLIC API
# ============================================================================
def optimize_with_calm_constraints(
    forecasts,
    car_results,
    lcr_results,
    risk_aversion: float = 1.0,
):
    """
    Public wrapper used by main.py for risk‑adjusted CALM optimization.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Output from run_forecasting_suite().
    car_results : pd.DataFrame
        Capital-at-risk results from calculate_portfolio_car().
    lcr_results : pd.DataFrame
        Liquidity coverage / funding gap results from calculate_funding_gap().
    risk_aversion : float, optional
        Global risk-aversion parameter for the optimizer.

    Returns
    -------
    pd.DataFrame
        Optimal portfolio allocation.
    """
    from pathlib import Path
    Path("risk").mkdir(exist_ok=True)
    forecasts.to_csv("risk/forecasts_input.csv", index=False)
    car_results.to_csv("risk/car_results_input.csv", index=False)
    lcr_results.to_csv("risk/lcr_results_input.csv", index=False)

    risk_budget = 0.15 / risk_aversion
    optimizer = RiskAdjustedOptimizer(risk_budget=risk_budget)
    return optimizer.optimize_allocation()


if __name__ == '__main__':
    main()