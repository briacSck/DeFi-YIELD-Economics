"""
Protocol risk assessment for DeFi lending platforms
Implements credit, operational, and market risk scoring, and CALM framework methodology
"""

import pandas as pd
import numpy as np
from pathlib import Path

class ProtocolRiskScorer:
    def __init__(self, data_path=None):
        if isinstance(data_path, (pd.DataFrame, pd.Series)):
            self.df = data_path.copy()
            self.data_path = None
        else:
            if data_path is None:
                data_path = "data/panel_latest.csv"
            self.data_path = Path(data_path)
            self.df = None
        self.risk_scores = {}
        
    def load_data(self):
        """Load protocol data with metadata"""
        # If data was passed directly as a DataFrame, skip file loading
        if self.df is not None:
            self.apy_col = 'apy_base' if 'apy_base' in self.df.columns else 'apy_total'
            print(f"✓ Using in-memory DataFrame: {self.df['pool'].nunique()} pools")
            return self

        # File-based path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.apy_col = 'apy_base' if 'apy_base' in self.df.columns else 'apy_total'

        print(f"✓ Loaded data for {self.df['pool'].nunique()} protocols")
        print(f"  Columns available: {len(self.df.columns)}")
        print(f"  Using APY column: {self.apy_col}")
        return self
    
    def calculate_credit_risk(self, protocol_id):
        """
        Credit risk using protocol tier and APY volatility
        """
        protocol_data = self.df[self.df['pool'] == protocol_id].iloc[0]
        
        # 1. Use protocol_tier (with correct string matching!)
        if 'protocol_tier' in protocol_data and pd.notna(protocol_data['protocol_tier']):
            tier_str = str(protocol_data['protocol_tier']).lower()
            
            if 'tier 1' in tier_str or 'blue chip' in tier_str:
                tier_risk = 0.01  # 1% - Aave, Compound
            elif 'tier 2' in tier_str or 'established' in tier_str:
                tier_risk = 0.03  # 3% - Spark, Sparklend
            elif 'tier 3' in tier_str or 'emerging' in tier_str:
                tier_risk = 0.05  # 5% - Euler, newer protocols
            elif 'tier 4' in tier_str or 'niche' in tier_str or 'new' in tier_str:
                tier_risk = 0.08  # 8% - Tectonic, very small protocols
            else:
                tier_risk = 0.06  # Default for unknown tiers
        else:
            # Fallback: Use TVL
            if 'tvl_usd' in protocol_data and pd.notna(protocol_data['tvl_usd']):
                tvl = protocol_data['tvl_usd']
                if tvl > 500_000_000:  # >$500M
                    tier_risk = 0.01
                elif tvl > 100_000_000:  # $100-500M
                    tier_risk = 0.03
                elif tvl > 10_000_000:  # $10-100M
                    tier_risk = 0.05
                else:  # <$10M
                    tier_risk = 0.08
            else:
                tier_risk = 0.06
        
        # 2. Use APY volatility for stability assessment
        if 'apy_volatility' in protocol_data and pd.notna(protocol_data['apy_volatility']):
            apy_vol = protocol_data['apy_volatility']
            # Scale: 0.0-0.2 volatility = low risk, >0.5 = high risk
            if apy_vol < 0.1:
                volatility_risk = 0.005  # Very stable
            elif apy_vol < 0.3:
                volatility_risk = 0.015  # Moderate
            elif apy_vol < 0.6:
                volatility_risk = 0.030  # Elevated
            else:
                volatility_risk = 0.050  # High volatility
        else:
            volatility_risk = 0.020  # Default moderate
        
        # 3. Combined credit risk (NO arbitrary cap!)
        credit_risk = (
            0.70 * tier_risk +         # 70% weight on protocol tier
            0.30 * volatility_risk     # 30% weight on APY stability
        )
        
        return credit_risk  # Returns 0.01-0.09 range instead of 0.20!
    
    def calculate_duration_risk(self, protocol_id):
        """Duration risk for stablecoin lending (short duration)"""
        # Standard 3-month duration for stablecoin protocols
        duration_years = 0.25
        rate_shock_bps = 200
        return duration_years * (rate_shock_bps / 10000)
    
    def calculate_market_risk(self, protocol_id):
        """
        Market risk using pre-computed Sharpe proxy and IL risk
        """
        protocol_data = self.df[self.df['pool'] == protocol_id].iloc[0]
        
        # Use Sharpe proxy if available (lower Sharpe = higher risk)
        if 'sharpe_proxy' in protocol_data and pd.notna(protocol_data['sharpe_proxy']):
            sharpe = protocol_data['sharpe_proxy']
            # Invert Sharpe to risk: Low Sharpe = High Risk
            if sharpe > 2.0:
                market_risk = 0.01  # Excellent risk-adjusted returns
            elif sharpe > 1.0:
                market_risk = 0.03  # Good
            elif sharpe > 0.5:
                market_risk = 0.05  # Moderate
            else:
                market_risk = 0.08  # Poor risk-adjusted returns
        else:
            # Fallback: Use IL risk or chain type
            if 'il_risk' in protocol_data and pd.notna(protocol_data['il_risk']):
                il_risk = protocol_data['il_risk']
                market_risk = min(il_risk * 0.1, 0.10)  # Scale IL risk
            else:
                # Check if stablecoin (low market risk)
                if 'is_stablecoin_flag' in protocol_data and protocol_data['is_stablecoin_flag']:
                    market_risk = 0.01
                else:
                    market_risk = 0.05
        
        return min(market_risk, 0.10)
    
    def calculate_operational_risk(self, protocol_id):
        """Operational risk based on protocol name and type"""
        protocol_data = self.df[self.df['pool'] == protocol_id].iloc[0]
        
        base_risk = 0.01
        
        # Use protocol name/chain for complexity assessment
        protocol_name = str(protocol_data.get('protocol', '')).lower()
        
        # Well-established protocols
        if any(name in protocol_name for name in ['aave', 'compound', 'maker']):
            return 0.008  # Very low operational risk
        
        # Newer complex protocols
        elif any(name in protocol_name for name in ['morpho', 'radiant', 'venus']):
            return 0.012  # Slightly elevated
        
        return base_risk
    
    def score_protocol(self, protocol_id):
        """Calculate comprehensive risk score"""
        try:
            credit = self.calculate_credit_risk(protocol_id)
            duration = self.calculate_duration_risk(protocol_id)
            market = self.calculate_market_risk(protocol_id)
            operational = self.calculate_operational_risk(protocol_id)
            
            carr = credit + duration + market + operational
            
            # Risk tier classification
            if carr < 0.035:
                tier = "A (Low Risk)"
            elif carr < 0.06:
                tier = "B (Moderate Risk)"
            elif carr < 0.10:
                tier = "C (Elevated Risk)"
            else:
                tier = "D (High Risk)"
            
            # Get protocol metadata
            protocol_data = self.df[self.df['pool'] == protocol_id].iloc[0]
            
            return {
                'protocol': protocol_id,
                'protocol_name': protocol_data.get('protocol', 'unknown'),
                'chain': protocol_data.get('chain', 'unknown'),
                'tvl_usd': protocol_data.get('tvl_usd', 0),
                'credit_risk': credit,
                'duration_risk': duration,
                'market_risk': market,
                'operational_risk': operational,
                'total_carr': carr,
                'risk_tier': tier,
                'sharpe_proxy': protocol_data.get('sharpe_proxy', None)
            }
        except Exception as e:
            print(f"  ⚠️  Error scoring {protocol_id[:20]}...: {e}")
            return None
    
    def score_all_protocols(self, min_observations=1):
        """Score all protocols"""
        # Get unique protocols (just latest snapshot)
        unique_protocols = self.df['pool'].unique()
        
        print(f"\n{'='*70}")
        print(f"PROTOCOL RISK SCORING ({len(unique_protocols)} protocols)")
        print(f"{'='*70}")
        
        scores = []
        for protocol_id in unique_protocols:
            score = self.score_protocol(protocol_id)
            if score:
                scores.append(score)
                
                print(f"\n{score['protocol_name'][:30]:30s} | {score['chain'][:10]:10s}")
                print(f"  Credit: {score['credit_risk']:.3f} | Duration: {score['duration_risk']:.3f} | "
                      f"Market: {score['market_risk']:.3f} | Operational: {score['operational_risk']:.3f}")
                print(f"  Total CaRR: {score['total_carr']:.3f} | Tier: {score['risk_tier']} | "
                      f"Sharpe: {score['sharpe_proxy']:.2f}" if score['sharpe_proxy'] else f"  Total CaRR: {score['total_carr']:.3f} | Tier: {score['risk_tier']}")
        
        scores_df = pd.DataFrame(scores)
        
        print(f"\n{'='*70}")
        print("RISK DISTRIBUTION")
        print(f"{'='*70}")
        print(scores_df['risk_tier'].value_counts().sort_index())
        
        print(f"\n{'='*70}")
        print("TOP 10 SAFEST PROTOCOLS (by CaRR)")
        print(f"{'='*70}")
        top_safe = scores_df.nsmallest(10, 'total_carr')[['protocol_name', 'chain', 'total_carr', 'risk_tier', 'tvl_usd']]
        print(top_safe.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("TOP 10 RISKIEST PROTOCOLS (by CaRR)")
        print(f"{'='*70}")
        top_risky = scores_df.nlargest(10, 'total_carr')[['protocol_name', 'chain', 'total_carr', 'risk_tier', 'tvl_usd']]
        print(top_risky.to_string(index=False))
        
        return scores_df
    
    def save_scores(self, scores_df, output_path=None):
        """Save risk scores"""
        if output_path is None:
            script_dir = Path(__file__).parent.parent
            output_path = script_dir / 'results' / 'protocol_risk_scores.csv'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(exist_ok=True)
        scores_df.to_csv(output_path, index=False)
        print(f"\n✓ Risk scores saved to {output_path}")

def main():
    """Run protocol risk scoring"""
    scorer = ProtocolRiskScorer()
    scorer.load_data()
    scores = scorer.score_all_protocols()
    scorer.save_scores(scores)
    
    return scores


# ============================================================================
# MAIN / PUBLIC API
# ============================================================================
def score_protocols(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Public wrapper used by main.py.

    Parameters
    ----------
    panel : pd.DataFrame
        Protocol-level panel/timeseries data.

    Returns
    -------
    pd.DataFrame
        DataFrame of protocol risk scores.
    """
    scorer = ProtocolRiskScorer(panel)
    scores = scorer.score_all_protocols()
    scorer.save_scores(scores)   # writes results/protocol_risk_scores.csv
    return scores


if __name__ == '__main__':
    scores = main()