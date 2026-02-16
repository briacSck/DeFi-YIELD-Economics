"""
Aggregate daily data into time-series panel for forecasting
"""

import pandas as pd
import glob
from datetime import datetime
from pathlib import Path

def build_panel_dataset():
    """Combine all daily snapshots into panel data"""
    
    # Load all snapshots
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir / 'raw'
    processed_dir = script_dir / 'processed'
    
    # Load all snapshots
    files = list(raw_data_dir.glob('defi_yields_*.csv'))
    files = [f for f in files if 'latest' not in f.name]  # Exclude 'latest'

    
    if len(files) == 0:
        print("⚠️ No historical data yet. Run collect_apy_data.py daily.")
        return None
    
    # Combine all snapshots
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    panel = pd.concat(dfs, ignore_index=True)
    
    # Create panel structure: (pool_id, date) pairs
    panel = panel.sort_values(['pool', 'date'])
    
    # Save processed panel
    processed_dir.mkdir(exist_ok=True)
    panel.to_csv(processed_dir / 'yield_panel.csv', index=False)

    
    print(f"✓ Built panel with {len(panel)} observations")
    print(f"  Time span: {panel['date'].min()} to {panel['date'].max()}")
    print(f"  Unique pools: {panel['pool'].nunique()}")
    
    return panel

if __name__ == '__main__':
    build_panel_dataset()