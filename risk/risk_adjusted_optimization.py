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
            return -portfolio_return  # Minimize negative = maximize
        
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
        bounds = tuple((0, 0.5) for _ in range(n))
        
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

if __name__ == '__main__':
    main()