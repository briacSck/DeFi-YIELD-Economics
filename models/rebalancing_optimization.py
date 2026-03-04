"""
Optimal rebalancing strategy under transaction costs
Tests whether AI-optimized portfolio management is economically viable for small depositors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class RebalancingSimulator:
    """Simulate portfolio rebalancing strategies with transaction costs"""
    
    def __init__(self, data_path='data/processed/yield_panel.csv'):
        self.data_path = Path(data_path)
        self.df = None
        self.results = []

    def load_data(self):
        """Load panel data and prepare for simulation"""
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['date', 'pool'])
        
        # Get APY column (different naming conventions)
        if 'apy' in self.df.columns:
            self.apy_col = 'apy'
        elif 'apyBase' in self.df.columns:
            self.apy_col = 'apyBase'
        elif 'apy_total' in self.df.columns:
            self.apy_col = 'apy_total'
        else:
            raise ValueError("No APY column found")
        
        print(f"✓ Loaded {len(self.df)} observations")
        print(f"  Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"  Unique dates: {self.df['date'].nunique()}")
        print(f"  Unique protocols: {self.df['pool'].nunique()}")
        
        return self
    
    def get_best_protocol(self, date, exclude_pools=None):
        """Find protocol with highest APY on given date"""
        day_data = self.df[self.df['date'] == date].copy()
        
        if exclude_pools:
            day_data = day_data[~day_data['pool'].isin(exclude_pools)]
        
        if len(day_data) == 0:
            return None, 0
        
        best = day_data.loc[day_data[self.apy_col].idxmax()]
        return best['pool'], best[self.apy_col]
    
    def calculate_daily_earnings(self, deposit, apy_annual):
        """Convert annual APY to daily earnings"""
        return deposit * (apy_annual / 100) / 365
    
    def strategy_always_best(self, initial_deposit, holding_days, gas_cost):
        """Naive strategy: Always move to highest APY protocol"""
        current_balance = initial_deposit
        current_protocol = None
        total_gas_paid = 0
        rebalance_count = 0
        daily_log = []
        
        dates = sorted(self.df['date'].unique())[:holding_days]
        
        for date in dates:
            best_pool, best_apy = self.get_best_protocol(date)
            
            # Rebalance if better protocol exists
            if current_protocol != best_pool:
                if current_protocol is not None:  # Not first deposit
                    current_balance -= gas_cost
                    total_gas_paid += gas_cost
                    rebalance_count += 1
                
                current_protocol = best_pool
            
            # Earn daily yield
            if current_protocol:
                daily_earnings = self.calculate_daily_earnings(current_balance, best_apy)
                current_balance += daily_earnings
                
                daily_log.append({
                    'date': date,
                    'protocol': current_protocol,
                    'apy': best_apy,
                    'balance': current_balance,
                    'daily_earnings': daily_earnings
                })
        
        final_return = current_balance - initial_deposit
        net_return_pct = (final_return / initial_deposit) * 100
        
        return {
            'strategy': 'always_best',
            'final_balance': current_balance,
            'gross_earnings': final_return + total_gas_paid,
            'gas_paid': total_gas_paid,
            'net_earnings': final_return,
            'rebalance_count': rebalance_count,
            'net_return_pct': net_return_pct,
            'daily_log': daily_log
        }
    
    def strategy_threshold(self, initial_deposit, holding_days, gas_cost, threshold_pct=1.5):
        """Smart strategy: Only rebalance if APY improvement > threshold + gas cost"""
        current_balance = initial_deposit
        current_protocol = None
        current_apy = 0
        total_gas_paid = 0
        rebalance_count = 0
        daily_log = []
        
        dates = sorted(self.df['date'].unique())[:holding_days]
        
        for date in dates:
            best_pool, best_apy = self.get_best_protocol(date)
            
            # Calculate if rebalancing is worth it
            apy_improvement = best_apy - current_apy
            gas_cost_as_apy = (gas_cost / current_balance) * 100 * 365  # Annualized gas cost
            
            should_rebalance = (
                current_protocol is None or  # First deposit
                (apy_improvement > threshold_pct and apy_improvement > gas_cost_as_apy)
            )
            
            if should_rebalance:
                if current_protocol is not None:
                    current_balance -= gas_cost
                    total_gas_paid += gas_cost
                    rebalance_count += 1
                
                current_protocol = best_pool
                current_apy = best_apy
            
            # Earn daily yield at current protocol
            if current_protocol:
                daily_earnings = self.calculate_daily_earnings(current_balance, current_apy)
                current_balance += daily_earnings
                
                daily_log.append({
                    'date': date,
                    'protocol': current_protocol,
                    'apy': current_apy,
                    'balance': current_balance,
                    'daily_earnings': daily_earnings
                })
        
        final_return = current_balance - initial_deposit
        net_return_pct = (final_return / initial_deposit) * 100
        
        return {
            'strategy': f'threshold_{threshold_pct}pct',
            'final_balance': current_balance,
            'gross_earnings': final_return + total_gas_paid,
            'gas_paid': total_gas_paid,
            'net_earnings': final_return,
            'rebalance_count': rebalance_count,
            'net_return_pct': net_return_pct,
            'daily_log': daily_log
        }
    
    def strategy_fixed_schedule(self, initial_deposit, holding_days, gas_cost, rebalance_every=7):
        """Fixed schedule: Rebalance every N days"""
        current_balance = initial_deposit
        current_protocol = None
        current_apy = 0
        total_gas_paid = 0
        rebalance_count = 0
        daily_log = []
        
        dates = sorted(self.df['date'].unique())[:holding_days]
        
        for i, date in enumerate(dates):
            best_pool, best_apy = self.get_best_protocol(date)
            
            # Rebalance on schedule
            if i % rebalance_every == 0:
                if current_protocol is not None:
                    current_balance -= gas_cost
                    total_gas_paid += gas_cost
                    rebalance_count += 1
                
                current_protocol = best_pool
                current_apy = best_apy
            
            # Earn daily yield
            if current_protocol:
                daily_earnings = self.calculate_daily_earnings(current_balance, current_apy)
                current_balance += daily_earnings
                
                daily_log.append({
                    'date': date,
                    'protocol': current_protocol,
                    'apy': current_apy,
                    'balance': current_balance,
                    'daily_earnings': daily_earnings
                })
        
        final_return = current_balance - initial_deposit
        net_return_pct = (final_return / initial_deposit) * 100
        
        return {
            'strategy': f'fixed_{rebalance_every}day',
            'final_balance': current_balance,
            'gross_earnings': final_return + total_gas_paid,
            'gas_paid': total_gas_paid,
            'net_earnings': final_return,
            'rebalance_count': rebalance_count,
            'net_return_pct': net_return_pct,
            'daily_log': daily_log
        }
    
    def strategy_buy_hold(self, initial_deposit, holding_days):
        """Baseline: Pick best protocol initially and never rebalance"""
        dates = sorted(self.df['date'].unique())[:holding_days]
        first_date = dates[0]
        
        # Pick best protocol on day 1
        best_pool, best_apy = self.get_best_protocol(first_date)
        
        current_balance = initial_deposit
        daily_log = []
        
        for date in dates:
            # Check current APY of our protocol on this date
            day_data = self.df[(self.df['date'] == date) & (self.df['pool'] == best_pool)]
            
            if len(day_data) > 0:
                current_apy = day_data[self.apy_col].iloc[0]
                daily_earnings = self.calculate_daily_earnings(current_balance, current_apy)
                current_balance += daily_earnings
                
                daily_log.append({
                    'date': date,
                    'protocol': best_pool,
                    'apy': current_apy,
                    'balance': current_balance,
                    'daily_earnings': daily_earnings
                })
        
        final_return = current_balance - initial_deposit
        net_return_pct = (final_return / initial_deposit) * 100
        
        return {
            'strategy': 'buy_and_hold',
            'final_balance': current_balance,
            'gross_earnings': final_return,
            'gas_paid': 0,
            'net_earnings': final_return,
            'rebalance_count': 0,
            'net_return_pct': net_return_pct,
            'daily_log': daily_log
        }

    def run_simulation(self, deposit_sizes=[100, 500, 1000, 5000, 10000], 
                      holding_days=14, gas_costs={'L2': 0.50, 'L1': 5.00}):
        """Run all strategies across different deposit sizes and gas costs"""
        
        print(f"\n{'='*70}")
        print("REBALANCING OPTIMIZATION SIMULATION")
        print(f"{'='*70}")
        print(f"Holding period: {holding_days} days")
        print(f"Gas costs: L2 ${gas_costs['L2']}, L1 ${gas_costs['L1']}")
        print(f"Deposit sizes: ${min(deposit_sizes)} - ${max(deposit_sizes)}")
        
        all_results = []
        
        for deposit in deposit_sizes:
            for gas_label, gas_cost in gas_costs.items():
                print(f"\n--- Deposit: ${deposit} | Gas: {gas_label} (${gas_cost}) ---")
                
                # Run all strategies
                buy_hold = self.strategy_buy_hold(deposit, holding_days)
                always_best = self.strategy_always_best(deposit, holding_days, gas_cost)
                threshold_1 = self.strategy_threshold(deposit, holding_days, gas_cost, threshold_pct=1.0)
                threshold_2 = self.strategy_threshold(deposit, holding_days, gas_cost, threshold_pct=2.0)
                fixed_7 = self.strategy_fixed_schedule(deposit, holding_days, gas_cost, rebalance_every=7)
                
                # Combine results
                for result in [buy_hold, always_best, threshold_1, threshold_2, fixed_7]:
                    result['deposit'] = deposit
                    result['gas_type'] = gas_label
                    result['gas_cost'] = gas_cost
                    all_results.append(result)
                    
                    print(f"  {result['strategy']:20s} | "
                          f"Net: ${result['net_earnings']:6.2f} ({result['net_return_pct']:5.2f}%) | "
                          f"Rebalances: {result['rebalance_count']:2d} | "
                          f"Gas: ${result['gas_paid']:5.2f}")
        
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'daily_log'} 
                                   for r in all_results])
        
        return results_df, all_results
    
    def analyze_results(self, results_df):
        """Generate insights from simulation results"""
        print(f"\n{'='*70}")
        print("KEY FINDINGS")
        print(f"{'='*70}")
        
        # Finding 1: Minimum viable deposit size
        for gas_type in results_df['gas_type'].unique():
            gas_results = results_df[results_df['gas_type'] == gas_type]
            
            # Find where rebalancing beats buy-and-hold
            for deposit in sorted(gas_results['deposit'].unique()):
                deposit_results = gas_results[gas_results['deposit'] == deposit]
                buy_hold_return = deposit_results[deposit_results['strategy'] == 'buy_and_hold']['net_earnings'].iloc[0]
                best_active = deposit_results[deposit_results['strategy'] != 'buy_and_hold']['net_earnings'].max()
                
                if best_active > buy_hold_return:
                    print(f"\n✓ {gas_type}: Rebalancing profitable at ${deposit}+ deposits")
                    print(f"  Buy-and-hold: ${buy_hold_return:.2f}")
                    print(f"  Best active:  ${best_active:.2f} (+" +
                          f"{((best_active - buy_hold_return) / buy_hold_return * 100):.1f}%)")
                    break
            else:
                print(f"\n✗ {gas_type}: Rebalancing NOT profitable at tested deposit sizes")
        
        # Finding 2: Best strategy by deposit size
        print(f"\n{'='*70}")
        print("RECOMMENDED STRATEGIES")
        print(f"{'='*70}")
        
        for deposit in sorted(results_df['deposit'].unique()):
            deposit_results = results_df


# ============================================================================
# MAIN / PUBLIC API
# ============================================================================
def run_backtests(optimal_alloc, panel, deposit_sizes=None):
    """
    Public wrapper used by main.py for the rebalancing backtests.

    Parameters
    ----------
    optimal_alloc : pd.DataFrame
        Output from optimize_with_calm_constraints(), with target allocations.
        (Currently unused in this simulator-based backtest.)
    panel : pd.DataFrame
        Full panel/timeseries dataset. (Currently unused; simulator reads CSV.)
    deposit_sizes : list[float], optional
        Deposit sizes to simulate. If None, use the module's default set.

    Returns
    -------
    pd.DataFrame
        Backtest results across all deposit sizes.
    """
    from pathlib import Path
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    panel.to_csv("data/processed/yield_panel.csv", index=False)

    if deposit_sizes is None:
        deposit_sizes = [100, 500, 1000, 5000, 10000]

    sim = RebalancingSimulator()
    sim.load_data()
    results_df, _ = sim.run_simulation(
        deposit_sizes=deposit_sizes,
        holding_days=14,
        gas_costs={"L2": 0.50, "L1": 5.00},
    )
    return results_df


if __name__ == "__main__":
    sim = RebalancingSimulator()
    sim.load_data()
    results_df, all_results = sim.run_simulation(
        deposit_sizes=[100, 500, 1000, 5000, 10000],
        holding_days=14,
        gas_costs={"L2": 0.50, "L1": 5.00},
    )
    print(results_df.head())
