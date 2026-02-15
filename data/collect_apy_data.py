"""
Comprehensive DeFi Stablecoin Yield Data Collection
Research focus: Unbiased analysis of all major lending protocols
Covers all EVM chains + Flow, Solana where data available
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict


# ============================================================================
# COMPREHENSIVE PROTOCOL COVERAGE
# ============================================================================

LENDING_PROTOCOLS = {
    'tier_1_established': [
        'aave-v3', 'aave-v2',
        'compound-v3', 'compound-v2', 'compound-finance',
        'maker', 'makerdao',
    ],
    'tier_2_morpho_ecosystem': [
        'morpho-blue', 'morpho-aave', 'morpho-compound',
    ],
    'tier_3_emerging': [
        'spark', 'radiant', 'venus', 'benqi',
        'euler', 'silo', 'ajna',
        'granary', 'dforce', 'tectonic',
        'moonwell', 'angle', 'sturdy',
    ],
    'cross_chain_native': [
        'stargate', 'layerzero',
    ],
    'flow_ecosystem': [
        'kittypunk', 'moremarkets', 'increment', 'flowty',
    ],
    'other_chains': [
        'solend', 'port-finance',  # Solana
        'apricot', 'tulip',  # Solana
    ]
}

# Flatten all protocols
ALL_PROTOCOLS = [p for tier in LENDING_PROTOCOLS.values() for p in tier]

# Comprehensive stablecoin list
STABLECOINS = [
    # USD pegged
    'USDC', 'USDT', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'LUSD',
    'FRAX', 'UST', 'USDD', 'USDC.E', 'USDbC', 'stgUSDC',
    # EUR pegged
    'EURC', 'EURS', 'EURT', 'agEUR',
    # Decentralized
    'sUSD', 'MIM', 'USDF', 'crvUSD', 'GHO',
]

# All relevant chains - treat equally
BLOCKCHAIN_NETWORKS = [
    # Ethereum mainnet
    'Ethereum',
    
    # L2s
    'Arbitrum', 'Optimism', 'Base', 'zkSync Era', 'Polygon zkEVM',
    'Linea', 'Scroll', 'Mantle', 'Blast',
    
    # Sidechains
    'Polygon', 'BNB Chain', 'Avalanche', 'Fantom',
    'Gnosis', 'Celo', 'Harmony', 'Moonbeam', 'Moonriver',
    
    # Alt L1s
    'Flow', 'Solana', 'Near', 'Aurora',
    
    # Others
    'Metis', 'Boba', 'Cronos',
]


# ============================================================================
# DATA COLLECTION
# ============================================================================

def fetch_defillama_pools() -> List[Dict]:
    """Fetch all available yield pools from DeFiLlama"""
    try:
        print("📡 Fetching comprehensive DeFi yield data from DeFiLlama...")
        url = "https://yields.llama.fi/pools"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        pools = data.get('data', [])
        print(f"✓ Retrieved {len(pools):,} total pools across all DeFi")
        return pools
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching DeFiLlama: {e}")
        return []


def filter_stablecoin_lending_pools(pools: List[Dict]) -> pd.DataFrame:
    """
    Filter for stablecoin lending/supply pools only
    Exclude: LP pools, leveraged farming, derivatives
    """
    filtered = []
    excluded_types = ['lp', 'vault', 'farm', 'leverage', 'perp', 'options']
    
    for pool in pools:
        project = pool.get('project', '').lower()
        symbol = pool.get('symbol', '').upper()
        chain = pool.get('chain', '')
        apy = pool.get('apy')
        pool_id = pool.get('pool', '')
        
        # Skip non-lending pools
        if any(ex_type in pool_id.lower() for ex_type in excluded_types):
            continue
        
        # Must be recognized lending protocol
        is_lending = any(proto in project for proto in ALL_PROTOCOLS)
        
        # Must be stablecoin
        is_stable = any(stable in symbol for stable in STABLECOINS)
        
        # Must be on tracked chain
        is_tracked_chain = chain in BLOCKCHAIN_NETWORKS
        
        # Must have valid APY
        has_apy = apy is not None and isinstance(apy, (int, float)) and apy > 0
        
        if is_lending and is_stable and is_tracked_chain and has_apy:
            filtered.append(pool)
    
    if not filtered:
        print("⚠️  No matching stablecoin lending pools found")
        return pd.DataFrame()
    
    df = pd.DataFrame(filtered)
    print(f"✓ Filtered to {len(df):,} stablecoin lending pools")
    print(f"   Across {df['chain'].nunique()} chains")
    print(f"   From {df['project'].nunique()} protocols")
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed fields and metadata"""
    if df.empty:
        return df
    
    # Timestamps
    df['timestamp'] = datetime.now().isoformat()
    df['date'] = datetime.now().strftime('%Y-%m-%d')
    df['collection_hour'] = datetime.now().hour
    
    # Standardize column names
    rename_map = {
        'project': 'protocol',
        'symbol': 'asset',
        'tvlUsd': 'tvl_usd',
        'apy': 'apy_total',
        'apyBase': 'apy_base',
        'apyReward': 'apy_reward',
        'apyMean30d': 'apy_30d_avg',
        'apyPct1D': 'apy_change_1d',
        'apyPct7D': 'apy_change_7d',
        'apyPct30D': 'apy_change_30d',
        'stablecoin': 'is_stablecoin_flag',
        'ilRisk': 'il_risk',
        'exposure': 'token_exposure',
    }
    
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Computed metrics
    
    # 1. APY volatility (deviation from 30d mean)
    if 'apy_total' in df.columns and 'apy_30d_avg' in df.columns:
        df['apy_volatility'] = abs(df['apy_30d_avg'] - df['apy_total'])
    
    # 2. TVL metrics
    if 'tvl_usd' in df.columns:
        df['tvl_millions'] = df['tvl_usd'] / 1_000_000
        df['tvl_log'] = np.log1p(df['tvl_usd'].fillna(0))
        
        # TVL categories
        df['tvl_category'] = pd.cut(
            df['tvl_usd'], 
            bins=[0, 1e6, 10e6, 100e6, np.inf],
            labels=['<$1M', '$1-10M', '$10-100M', '>$100M']
        )
    
    # 3. Risk-adjusted return (Sharpe proxy)
    if 'apy_total' in df.columns and 'apy_volatility' in df.columns:
        df['sharpe_proxy'] = df['apy_total'] / (df['apy_volatility'] + 0.5)
    
    # 4. Protocol tier classification
    df['protocol_tier'] = df['protocol'].apply(classify_protocol_tier)
    
    # 5. Chain type
    df['chain_type'] = df['chain'].apply(classify_chain_type)
    
    # 6. Stablecoin type
    df['stable_type'] = df['asset'].apply(classify_stablecoin_type)
    
    return df


def classify_protocol_tier(protocol: str) -> str:
    """Classify protocols by maturity/adoption"""
    protocol = protocol.lower()
    
    if any(p in protocol for p in ['aave', 'compound', 'maker']):
        return 'Tier 1 - Blue Chip'
    elif any(p in protocol for p in ['morpho', 'spark']):
        return 'Tier 2 - Established'
    elif any(p in protocol for p in ['radiant', 'venus', 'benqi', 'euler']):
        return 'Tier 3 - Emerging'
    else:
        return 'Tier 4 - New/Niche'


def classify_chain_type(chain: str) -> str:
    """Classify chains by type"""
    l2s = ['Arbitrum', 'Optimism', 'Base', 'zkSync', 'Polygon zkEVM', 'Linea', 'Scroll', 'Mantle', 'Blast']
    sidechains = ['Polygon', 'BNB Chain', 'Avalanche', 'Fantom', 'Gnosis']
    alt_l1s = ['Flow', 'Solana', 'Near', 'Aurora']
    
    if chain == 'Ethereum':
        return 'L1 - Ethereum'
    elif chain in l2s:
        return 'L2 - Rollup'
    elif chain in sidechains:
        return 'Sidechain'
    elif chain in alt_l1s:
        return 'Alt L1'
    else:
        return 'Other'


def classify_stablecoin_type(asset: str) -> str:
    """Classify stablecoins by backing type"""
    asset_upper = asset.upper()
    
    # Fiat-backed centralized
    if any(s in asset_upper for s in ['USDC', 'USDT', 'BUSD', 'EURC', 'TUSD', 'GUSD']):
        return 'Fiat-backed'
    
    # Decentralized overcollateralized
    elif any(s in asset_upper for s in ['DAI', 'LUSD', 'FRAX', 'SUSD', 'CRVUSD', 'GHO']):
        return 'Decentralized'
    
    # Algorithmic/hybrid
    elif any(s in asset_upper for s in ['UST', 'USDD', 'MIM']):
        return 'Algorithmic'
    
    else:
        return 'Other'


def save_comprehensive_dataset(df: pd.DataFrame, output_dir: str = 'raw'):
    """Save timestamped and latest versions"""
    if df.empty:
        print("⚠️  No data to save")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Timestamped version
    filename = f"{output_dir}/defi_yields_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 Saved: {filename}")
    
    # Latest version (overwrite)
    latest = f"{output_dir}/defi_yields_latest.csv"
    df.to_csv(latest, index=False)
    print(f"💾 Saved: {latest}")
    
    return filename


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_market_landscape(df: pd.DataFrame):
    """Comprehensive market analysis - no bias"""
    if df.empty:
        return
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE DEFI STABLECOIN LENDING MARKET ANALYSIS")
    print("="*80)
    
    # === OVERALL MARKET STATISTICS ===
    print("\n🌐 MARKET OVERVIEW:")
    print(f"   Total pools: {len(df):,}")
    print(f"   Chains covered: {df['chain'].nunique()}")
    print(f"   Protocols: {df['protocol'].nunique()}")
    print(f"   Assets: {df['asset'].nunique()}")
    
    if 'apy_total' in df.columns:
        print(f"\n   APY Statistics:")
        print(f"   • Mean: {df['apy_total'].mean():.2f}%")
        print(f"   • Median: {df['apy_total'].median():.2f}%")
        print(f"   • Std Dev: {df['apy_total'].std():.2f}%")
        print(f"   • Range: {df['apy_total'].min():.2f}% - {df['apy_total'].max():.2f}%")
        print(f"   • IQR: {df['apy_total'].quantile(0.25):.2f}% - {df['apy_total'].quantile(0.75):.2f}%")
    
    if 'tvl_usd' in df.columns:
        total_tvl = df['tvl_usd'].sum()
        print(f"\n   Total TVL: ${total_tvl:,.0f}")
        print(f"   Mean TVL per pool: ${df['tvl_usd'].mean():,.0f}")
        print(f"   Median TVL per pool: ${df['tvl_usd'].median():,.0f}")
    
    # === BY CHAIN ===
    print("\n⛓️  BLOCKCHAIN COMPARISON (All chains):")
    if 'chain' in df.columns and 'apy_total' in df.columns:
        chain_stats = df.groupby('chain').agg({
            'apy_total': ['mean', 'median', 'std', 'count'],
            'tvl_usd': 'sum'
        }).round(2)
        chain_stats.columns = ['apy_mean', 'apy_median', 'apy_std', 'pool_count', 'total_tvl']
        chain_stats = chain_stats.sort_values('apy_mean', ascending=False)
        
        print(f"\n   Ranked by mean APY:")
        for idx, (chain, row) in enumerate(chain_stats.head(15).iterrows(), 1):
            tvl_str = f"${row['total_tvl']/1e6:.1f}M" if pd.notna(row['total_tvl']) else 'N/A'
            print(f"   {idx:2}. {chain:15} | {row['apy_mean']:5.2f}% avg | "
                  f"{row['apy_median']:5.2f}% med | {int(row['pool_count']):3} pools | TVL: {tvl_str}")
    
    # === BY PROTOCOL ===
    print("\n🏦 PROTOCOL COMPARISON (Top 20 by average APY):")
    if 'protocol' in df.columns and 'apy_total' in df.columns:
        protocol_stats = df.groupby('protocol').agg({
            'apy_total': ['mean', 'median', 'count'],
            'tvl_usd': 'sum',
            'chain': lambda x: x.nunique()
        }).round(2)
        protocol_stats.columns = ['apy_mean', 'apy_median', 'pool_count', 'total_tvl', 'chain_count']
        protocol_stats = protocol_stats[protocol_stats['pool_count'] >= 2]  # Filter noise
        protocol_stats = protocol_stats.sort_values('apy_mean', ascending=False)
        
        for idx, (protocol, row) in enumerate(protocol_stats.head(20).iterrows(), 1):
            tvl_str = f"${row['total_tvl']/1e6:.1f}M" if pd.notna(row['total_tvl']) else 'N/A'
            print(f"   {idx:2}. {protocol:20} | {row['apy_mean']:5.2f}% avg | "
                  f"{int(row['pool_count']):2} pools | {int(row['chain_count'])} chains | TVL: {tvl_str}")
    
    # === BY STABLECOIN ===
    print("\n💰 STABLECOIN COMPARISON:")
    if 'asset' in df.columns and 'apy_total' in df.columns:
        stable_stats = df.groupby('asset').agg({
            'apy_total': ['mean', 'median', 'count'],
            'tvl_usd': 'sum'
        }).round(2)
        stable_stats.columns = ['apy_mean', 'apy_median', 'pool_count', 'total_tvl']
        stable_stats = stable_stats[stable_stats['pool_count'] >= 3]
        stable_stats = stable_stats.sort_values('apy_mean', ascending=False)
        
        for idx, (asset, row) in enumerate(stable_stats.head(15).iterrows(), 1):
            tvl_str = f"${row['total_tvl']/1e6:.1f}M" if pd.notna(row['total_tvl']) else 'N/A'
            print(f"   {idx:2}. {asset:12} | {row['apy_mean']:5.2f}% avg | "
                  f"{int(row['pool_count']):3} pools | TVL: {tvl_str}")
    
    # === RISK-ADJUSTED RETURNS ===
    print("\n📈 RISK-ADJUSTED PERFORMANCE (Top 15 by Sharpe proxy):")
    if 'sharpe_proxy' in df.columns and pd.notna(df['sharpe_proxy']).any():
        top_sharpe = df.nlargest(15, 'sharpe_proxy')[
            ['protocol', 'chain', 'asset', 'apy_total', 'apy_volatility', 'sharpe_proxy']
        ].round(2)
        
        for idx, row in enumerate(top_sharpe.itertuples(), 1):
            print(f"   {idx:2}. {row.protocol:18} on {row.chain:15}")
            print(f"       {row.asset:10} | {row.apy_total:5.2f}% APY | "
                  f"Vol: {row.apy_volatility:4.2f}% | Sharpe: {row.sharpe_proxy:5.2f}")
    
    # === PROTOCOL TIER ANALYSIS ===
    if 'protocol_tier' in df.columns:
        print("\n🎯 PERFORMANCE BY PROTOCOL MATURITY:")
        tier_stats = df.groupby('protocol_tier')['apy_total'].agg(['mean', 'median', 'count']).round(2)
        tier_stats = tier_stats.sort_values('mean', ascending=False)
        
        for tier, row in tier_stats.iterrows():
            print(f"   {tier:25} | {row['mean']:5.2f}% avg | {int(row['count']):3} pools")
    
    # === CHAIN TYPE ANALYSIS ===
    if 'chain_type' in df.columns:
        print("\n🔗 PERFORMANCE BY CHAIN TYPE:")
        chain_type_stats = df.groupby('chain_type')['apy_total'].agg(['mean', 'median', 'count']).round(2)
        chain_type_stats = chain_type_stats.sort_values('mean', ascending=False)
        
        for chain_type, row in chain_type_stats.iterrows():
            print(f"   {chain_type:20} | {row['mean']:5.2f}% avg | {int(row['count']):3} pools")
    
    # === STABLECOIN TYPE ANALYSIS ===
    if 'stable_type' in df.columns:
        print("\n💵 PERFORMANCE BY STABLECOIN TYPE:")
        stable_type_stats = df.groupby('stable_type')['apy_total'].agg(['mean', 'median', 'count']).round(2)
        stable_type_stats = stable_type_stats.sort_values('mean', ascending=False)
        
        for stable_type, row in stable_type_stats.iterrows():
            print(f"   {stable_type:20} | {row['mean']:5.2f}% avg | {int(row['count']):3} pools")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute data collection"""
    print("\n🔬 Comprehensive DeFi Stablecoin Yield Research")
    print(f"⏰ Collection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    # Fetch all data
    all_pools = fetch_defillama_pools()
    if not all_pools:
        print("❌ Data fetch failed")
        return
    
    # Filter for stablecoin lending
    df_filtered = filter_stablecoin_lending_pools(all_pools)
    if df_filtered.empty:
        print("❌ No stablecoin lending pools found")
        return
    
    # Enrich with computed metrics
    df_enriched = enrich_data(df_filtered)
    
    # Save datasets
    saved_file = save_comprehensive_dataset(df_enriched)
    
    # Analyze
    analyze_market_landscape(df_enriched)
    
    # Summary
    print(f"\n✅ Data collection complete!")
    print(f"   📊 Total records: {len(df_enriched):,}")
    print(f"   💾 Saved to: {saved_file}")
    print(f"\n📁 Ready for analysis in: raw/defi_yields_latest.csv")


if __name__ == '__main__':
    main()
