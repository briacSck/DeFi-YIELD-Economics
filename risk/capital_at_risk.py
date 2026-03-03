"""
Capital at Risk (CaR) Module for DeFi YIELD portfolio management
Calculates portfolio capital at risk based on the framework from 
Bluhm et al. (2024) using protocol risk scores and market conditions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RISK_DIR = BASE_DIR / 'risk'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

class CapitalAtRiskCalculator:
    """Calculate and analyze Capital at Risk for DeFi portfolio"""
    
    def __init__(self, confidence_level=0.95):
        """
        Initialize CaR calculator
        
        Args:
            confidence_level: Confidence level for CaR calculation (default 95%)
        """
        self.confidence_level = confidence_level
        self.risk_scores = None
        self.timeseries_data = None
        self.portfolio = None
        
    def load_data(self):
        """Load risk scores and timeseries data"""
        print("Loading data...")
        
        # Load protocol risk scores
        risk_file = BASE_DIR / 'results' / 'protocol_risk_scores.csv'
        if risk_file.exists():
            self.risk_scores = pd.read_csv(risk_file)
            print(f"Loaded {len(self.risk_scores)} protocol risk scores")
        else:
            raise FileNotFoundError(f"Risk scores file not found: {risk_file}")
        
        # Load timeseries data for historical volatility
        ts_file = DATA_DIR / 'processed' / 'timeseries_apy.csv'
        if ts_file.exists():
            self.timeseries_data = pd.read_csv(ts_file)
            self.timeseries_data['date'] = pd.to_datetime(self.timeseries_data['date'])
            print(f"Loaded {len(self.timeseries_data)} timeseries records")
        else:
            print(f"Warning: Timeseries file not found: {ts_file}")
            self.timeseries_data = None
    
    def create_sample_portfolio(self, total_capital=1000000):
        """
        Create a sample portfolio based on risk scores
        
        Args:
            total_capital: Total capital to allocate (default $1M)
        """
        print(f"\nCreating sample portfolio with ${total_capital:,.0f}...")
        
        # Select top protocols by risk-adjusted score
        # Weight by inverse of total risk score (lower risk = higher allocation)
        df = self.risk_scores.copy()
        
        # Calculate allocation weight (inverse of risk)
        df['risk_weight'] = 1 / (df['total_carr'] + 0.1)  # Add small constant to avoid division by zero
        df['allocation_pct'] = df['risk_weight'] / df['risk_weight'].sum()
        
        # Allocate capital
        df['capital_allocated'] = df['allocation_pct'] * total_capital
        
        # Focus on top 20 protocols by allocation
        self.portfolio = df.nlargest(20, 'capital_allocated').copy()
        
        print(f"Portfolio created with {len(self.portfolio)} protocols")
        print(f"Total allocated: ${self.portfolio['capital_allocated'].sum():,.2f}")
        
        return self.portfolio
    
    def calculate_position_car(self):
        """Calculate Capital at Risk for each position"""
        print("\nCalculating position-level CaR...")
        
        df = self.portfolio.copy()
        
        # Position CaR based on risk scores
        # CaR = Capital * (Risk Score / 100) * Confidence Factor
        confidence_factor = 1 + (self.confidence_level - 0.5)  # Scale by confidence level
        
        # Credit risk component (default probability)
        df['car_credit'] = df['capital_allocated'] * (df['credit_risk'] / 100) * confidence_factor
        
        # Duration risk component (interest rate sensitivity)
        df['car_duration'] = df['capital_allocated'] * (df['duration_risk'] / 100) * confidence_factor
        
        # Market risk component (volatility-based)
        df['car_market'] = df['capital_allocated'] * (df['market_risk'] / 100) * confidence_factor
        
        # Operational risk component
        df['car_operational'] = df['capital_allocated'] * (df['operational_risk'] / 100) * confidence_factor
        
        # Total position CaR
        df['car_total'] = df['car_credit'] + df['car_duration'] + df['car_market'] + df['car_operational']
        
        # CaR as percentage of position
        df['car_pct'] = (df['car_total'] / df['capital_allocated']) * 100
        
        self.portfolio = df
        
        print(f"Average position CaR: {df['car_pct'].mean():.2f}%")
        print(f"Total portfolio CaR: ${df['car_total'].sum():,.2f}")
        
        return df
    
    def calculate_portfolio_car(self):
        """Calculate aggregate portfolio Capital at Risk with diversification"""
        print("\nCalculating portfolio-level CaR with diversification...")
        
        # Simple diversification benefit (assumes some correlation < 1)
        # Diversification factor: sqrt(n) for equal weighted, adjusted by concentration
        n_positions = len(self.portfolio)
        concentration = (self.portfolio['capital_allocated'] ** 2).sum() / (self.portfolio['capital_allocated'].sum() ** 2)
        diversification_factor = np.sqrt(n_positions * concentration)
        
        # Undiversified CaR (sum of individual CaRs)
        undiversified_car = self.portfolio['car_total'].sum()
        
        # Diversified CaR (accounting for imperfect correlation)
        diversified_car = undiversified_car / diversification_factor
        
        diversification_benefit = undiversified_car - diversified_car
        diversification_benefit_pct = (diversification_benefit / undiversified_car) * 100
        
        portfolio_metrics = {
            'total_capital': self.portfolio['capital_allocated'].sum(),
            'undiversified_car': undiversified_car,
            'diversified_car': diversified_car,
            'diversification_benefit': diversification_benefit,
            'diversification_benefit_pct': diversification_benefit_pct,
            'car_pct_of_capital': (diversified_car / self.portfolio['capital_allocated'].sum()) * 100,
            'n_positions': n_positions,
            'concentration_hhi': concentration
        }
        
        print(f"\nPortfolio CaR Metrics:")
        print(f"  Total Capital: ${portfolio_metrics['total_capital']:,.2f}")
        print(f"  Undiversified CaR: ${portfolio_metrics['undiversified_car']:,.2f}")
        print(f"  Diversified CaR: ${portfolio_metrics['diversified_car']:,.2f}")
        print(f"  Diversification Benefit: ${portfolio_metrics['diversification_benefit']:,.2f} ({portfolio_metrics['diversification_benefit_pct']:.2f}%)")
        print(f"  CaR as % of Capital: {portfolio_metrics['car_pct_of_capital']:.2f}%")
        
        return portfolio_metrics
    
    def stress_test(self, scenarios=None):
        """
        Run stress test scenarios on portfolio
        
        Args:
            scenarios: Dict of scenario names and risk multipliers
        """
        print("\nRunning stress test scenarios...")
        
        if scenarios is None:
            scenarios = {
                'Baseline': 1.0,
                'Moderate Stress': 1.5,
                'Severe Stress': 2.0,
                'Extreme Stress': 3.0,
                'Market Crash': 4.0
            }
        
        stress_results = []
        
        for scenario_name, multiplier in scenarios.items():
            # Apply multiplier to risk scores
            stressed_car = self.portfolio['car_total'] * multiplier
            total_stressed_car = stressed_car.sum()
            
            # Calculate losses as percentage of capital
            loss_pct = (total_stressed_car / self.portfolio['capital_allocated'].sum()) * 100
            
            stress_results.append({
                'scenario': scenario_name,
                'multiplier': multiplier,
                'car_total': total_stressed_car,
                'loss_pct': loss_pct
            })
        
        stress_df = pd.DataFrame(stress_results)
        
        print("\nStress Test Results:")
        for _, row in stress_df.iterrows():
            print(f"  {row['scenario']}: ${row['car_total']:,.2f} ({row['loss_pct']:.2f}% of capital)")
        
        return stress_df
    
    def analyze_by_tier(self):
        """Analyze CaR by protocol risk tier"""
        print("\nAnalyzing CaR by risk tier...")
        
        tier_analysis = self.portfolio.groupby('risk_tier').agg({
            'capital_allocated': 'sum',
            'car_total': 'sum',
            'protocol': 'count'
        }).reset_index()
        
        tier_analysis.columns = ['risk_tier', 'capital', 'car_total', 'n_protocols']
        tier_analysis['car_pct'] = (tier_analysis['car_total'] / tier_analysis['capital']) * 100
        tier_analysis['capital_pct'] = (tier_analysis['capital'] / tier_analysis['capital'].sum()) * 100
        
        print("\nCaR by Risk Tier:")
        for _, row in tier_analysis.iterrows():
            print(f"  Tier {row['risk_tier']}: ${row['car_total']:,.2f} CaR on ${row['capital']:,.2f} "
                  f"({row['car_pct']:.2f}%) - {row['n_protocols']} protocols ({row['capital_pct']:.1f}% of capital)")
        
        return tier_analysis
    
    def visualize_car(self, portfolio_metrics, stress_df, tier_analysis):
        """Create comprehensive CaR visualization"""
        print("\nCreating CaR visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top 10 Positions by CaR
        ax1 = plt.subplot(2, 3, 1)
        top_car = self.portfolio.nlargest(10, 'car_total')
        ax1.barh(range(len(top_car)), top_car['car_total'], color='crimson', alpha=0.7)
        ax1.set_yticks(range(len(top_car)))
        ax1.set_yticklabels([f"{p[:20]}" for p in top_car['protocol']], fontsize=9)
        ax1.set_xlabel('Capital at Risk ($)', fontsize=10)
        ax1.set_title('Top 10 Positions by CaR', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. CaR Components Breakdown
        ax2 = plt.subplot(2, 3, 2)
        car_components = {
            'Credit': self.portfolio['car_credit'].sum(),
            'Duration': self.portfolio['car_duration'].sum(),
            'Market': self.portfolio['car_market'].sum(),
            'Operational': self.portfolio['car_operational'].sum()
        }
        colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        ax2.pie(car_components.values(), labels=car_components.keys(), autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('CaR by Risk Component', fontsize=12, fontweight='bold')
        
        # 3. Stress Test Results
        ax3 = plt.subplot(2, 3, 3)
        ax3.bar(range(len(stress_df)), stress_df['car_total'], color='darkred', alpha=0.7)
        ax3.set_xticks(range(len(stress_df)))
        ax3.set_xticklabels(stress_df['scenario'], rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Capital at Risk ($)', fontsize=10)
        ax3.set_title('Stress Test Scenarios', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. CaR vs Capital Allocated
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(self.portfolio['capital_allocated'], self.portfolio['car_total'],
                   c=self.portfolio['total_carr'], cmap='RdYlGn_r', s=100, alpha=0.6)
        ax4.set_xlabel('Capital Allocated ($)', fontsize=10)
        ax4.set_ylabel('Capital at Risk ($)', fontsize=10)
        ax4.set_title('Position Size vs CaR', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Risk Score', fontsize=9)
        
        # 5. CaR by Risk Tier
        ax5 = plt.subplot(2, 3, 5)
        tier_order = ['A', 'B', 'C', 'D']
        tier_analysis_sorted = tier_analysis.set_index('risk_tier').reindex(tier_order).reset_index()
        bars = ax5.bar(tier_analysis_sorted['risk_tier'], tier_analysis_sorted['car_total'],
                      color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax5.set_xlabel('Risk Tier', fontsize=10)
        ax5.set_ylabel('Total CaR ($)', fontsize=10)
        ax5.set_title('Capital at Risk by Tier', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Portfolio Composition
        ax6 = plt.subplot(2, 3, 6)
        tier_capital = tier_analysis_sorted[['risk_tier', 'capital']].dropna()  # Remove NaN rows
        if len(tier_capital) > 0:
            # Filter out zero/NaN values
            tier_capital_filtered = tier_capital[tier_capital['capital'] > 0]
            if len(tier_capital_filtered) > 0:
                tier_colors = []
                for tier in tier_capital_filtered['risk_tier']:
                    if tier == 'A':
                        tier_colors.append('green')
                    elif tier == 'B':
                        tier_colors.append('yellow')
                    elif tier == 'C':
                        tier_colors.append('orange')
                    else:
                        tier_colors.append('red')
                
                ax6.pie(tier_capital_filtered['capital'], 
                    labels=[f"Tier {t}" for t in tier_capital_filtered['risk_tier']],
                    autopct='%1.1f%%', colors=tier_colors, startangle=90)
                ax6.set_title('Capital Allocation by Tier', fontsize=12, fontweight='bold')
            else:
                ax6.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax6.set_title('Capital Allocation by Tier', fontsize=12, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax6.set_title('Capital Allocation by Tier', fontsize=12, fontweight='bold')

        ##### execution snippet
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save figure
        output_file = OUTPUT_DIR / f'car_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {output_file}")
        
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive CaR report"""
        print("\n" + "="*70)
        print("CAPITAL AT RISK (CaR) ANALYSIS REPORT")
        print("="*70)
        
        self.load_data()
        self.create_sample_portfolio()
        self.calculate_position_car()
        portfolio_metrics = self.calculate_portfolio_car()
        stress_df = self.stress_test()
        tier_analysis = self.analyze_by_tier()
        
        # Save results
        results_file = RISK_DIR / 'car_analysis_results.csv'
        self.portfolio.to_csv(results_file, index=False)
        print(f"\nDetailed results saved: {results_file}")
        
        # Create visualizations
        self.visualize_car(portfolio_metrics, stress_df, tier_analysis)
        
        return portfolio_metrics, stress_df, tier_analysis


def main():
    """Run Capital at Risk analysis"""
    calculator = CapitalAtRiskCalculator(confidence_level=0.95)
    portfolio_metrics, stress_df, tier_analysis = calculator.generate_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    

if __name__ == '__main__':
    main()
