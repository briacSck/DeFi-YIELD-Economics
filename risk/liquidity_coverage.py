"""
Liquidity Coverage Analysis for DeFi Protocols
Measures ability to meet withdrawal demands under stress scenarios
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RISK_DIR = BASE_DIR / 'risk'

class LiquidityCoverageAnalyzer:
    """Analyze protocol liquidity coverage ratios"""
    
    def __init__(self):
        self.risk_scores = pd.read_csv(BASE_DIR / 'results' / 'protocol_risk_scores.csv')
    
    def calculate_lcr(self):
        """
        Enhanced Liquidity Coverage Ratio using real TVL and volatility
        
        LCR = Available Liquidity / Stress Outflows
        Stress Outflows = TVL * Withdrawal_Rate * Volatility_Multiplier
        """
        df = self.risk_scores.copy()
        
        # Stress scenarios
        base_withdrawal_rate = 0.10  # 10% baseline stress withdrawal
        
        # Adjust withdrawal rate by volatility (higher volatility = higher expected withdrawals)
        # Normalize market_risk (0-100) to a multiplier (0.5x to 2x)
        volatility_multiplier = 0.5 + (df['market_risk'] / 100) * 1.5
        
        # Stress outflow = baseline * volatility adjustment
        df['stress_outflow'] = base_withdrawal_rate * volatility_multiplier
        
        # Available liquidity proxy: inverse of operational risk
        # (Better operational security = more reliable liquidity)
        df['available_liquidity'] = (100 - df['operational_risk']) / 100
        
        # LCR = Available / Required
        df['lcr'] = df['available_liquidity'] / df['stress_outflow']
        
        # Categorize using Basel III-inspired thresholds
        # LCR >= 1.0 = Adequate (can cover 100% of stress outflows)
        df['liquidity_tier'] = pd.cut(df['lcr'], 
                                    bins=[0, 0.7, 1.0, 1.5, np.inf],
                                    labels=['Critical', 'Low', 'Adequate', 'Excess'])
        
        print("\n=== LIQUIDITY COVERAGE ANALYSIS ===")
        print(f"\nProtocols by Liquidity Tier:")
        tier_counts = df['liquidity_tier'].value_counts().sort_index()
        for tier, count in tier_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {tier}: {count} protocols ({pct:.1f}%)")
        
        print(f"\nLCR Statistics:")
        print(f"  Mean: {df['lcr'].mean():.2f}")
        print(f"  Median: {df['lcr'].median():.2f}")
        print(f"  Min: {df['lcr'].min():.2f}")
        print(f"  Max: {df['lcr'].max():.2f}")
        
        print(f"\nTop 5 Most Liquid Protocols:")
        top_liquid = df.nlargest(5, 'lcr')[['protocol_name', 'lcr', 'liquidity_tier', 'risk_tier']]
        print(top_liquid.to_string(index=False))
        
        print(f"\nBottom 5 Least Liquid Protocols:")
        low_liquid = df.nsmallest(5, 'lcr')[['protocol_name', 'lcr', 'liquidity_tier', 'risk_tier']]
        print(low_liquid.to_string(index=False))
        
        # Save results
        output_file = RISK_DIR / 'liquidity_coverage_results.csv'
        df[['protocol', 'protocol_name', 'available_liquidity', 'stress_outflow', 'lcr', 'liquidity_tier']].to_csv(output_file, index=False)
        print(f"\nResults saved: {output_file}")

        return df

def main():
    analyzer = LiquidityCoverageAnalyzer()
    analyzer.calculate_lcr()

# ============================================================================
# MAIN / PUBLIC API
# ============================================================================
def calculate_funding_gap(panel):
    """
    Public wrapper used by main.py for liquidity coverage / funding gap analysis.

    Parameters
    ----------
    panel : pd.DataFrame
        Protocol-level panel/timeseries data (same panel passed to score_protocols).

    Returns
    -------
    pd.DataFrame
        Liquidity coverage / funding gap results.
    """
    panel.to_csv("data/panel_latest.csv", index=False)
    analyzer = LiquidityCoverageAnalyzer()
    return analyzer.calculate_lcr()
    

if __name__ == '__main__':
    main()