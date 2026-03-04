"""
Time-series forecasting for DeFi yields
Compares ARIMA, XGBoost, and LSTM against naive baselines
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ML models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Deep learning (optional - only if you have enough data)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️  TensorFlow not available. LSTM models will be skipped.")

class YieldForecaster:
    """Forecast DeFi protocol yields using multiple methods"""
    
    def __init__(self, data_path='data/processed/yield_panel.csv'):
        self.data_path = Path(data_path)
        self.df = None
        self.results = []
        
    def load_data(self):
        """Load and prepare panel data"""
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['pool', 'date'])
        
        print(f"✓ Loaded {len(self.df)} observations")
        print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Unique protocols: {self.df['pool'].nunique()}")
        print(f"  Days of data: {self.df['date'].nunique()}")
        
        return self
    
    def select_protocols(self, min_observations=10, top_n=5):
        """Select protocols with sufficient data for forecasting"""
        protocol_counts = self.df.groupby('pool').size()
        valid_protocols = protocol_counts[protocol_counts >= min_observations]
        
        # Prioritize by TVL (if available)
        if 'tvlUsd' in self.df.columns:
            protocol_tvl = self.df.groupby('pool')['tvlUsd'].mean()
            selected = protocol_tvl[protocol_tvl.index.isin(valid_protocols.index)].nlargest(top_n)
        else:
            selected = valid_protocols.nlargest(top_n)
        
        print(f"\n✓ Selected {len(selected)} protocols with {min_observations}+ observations")
        return selected.index.tolist()
    
    def naive_forecast(self, train, test):
        """Baseline: Tomorrow's yield = today's yield"""
        last_value = train['apy'].iloc[-1]
        predictions = [last_value] * len(test)
        mae = mean_absolute_error(test['apy'], predictions)
        rmse = np.sqrt(mean_squared_error(test['apy'], predictions))
        return {'mae': mae, 'rmse': rmse, 'predictions': predictions}
    
    def historical_mean_forecast(self, train, test, window=7):
        """Baseline: Use rolling mean as forecast"""
        mean_value = train['apy'].tail(window).mean()
        predictions = [mean_value] * len(test)
        mae = mean_absolute_error(test['apy'], predictions)
        rmse = np.sqrt(mean_squared_error(test['apy'], predictions))
        return {'mae': mae, 'rmse': rmse, 'predictions': predictions}
    
    def arima_forecast(self, train, test, order=(1,0,1)):
        """ARIMA time-series model"""
        try:
            model = ARIMA(train['apy'], order=order)
            fitted = model.fit()
            predictions = fitted.forecast(steps=len(test))
            mae = mean_absolute_error(test['apy'], predictions)
            rmse = np.sqrt(mean_squared_error(test['apy'], predictions))
            return {'mae': mae, 'rmse': rmse, 'predictions': predictions.tolist()}
        except Exception as e:
            print(f"    ARIMA failed: {e}")
            return None
    
    def xgboost_forecast(self, train, test):
        """XGBoost with engineered features"""
        def create_features(df):
            """Engineer time-series features"""
            features = pd.DataFrame()
            features['apy_lag1'] = df['apy'].shift(1)
            features['apy_lag7'] = df['apy'].shift(7)
            features['apy_rolling_mean_7'] = df['apy'].rolling(7, min_periods=1).mean()
            features['apy_rolling_std_7'] = df['apy'].rolling(7, min_periods=1).std()
            if 'tvlUsd' in df.columns:
                features['tvl_change'] = df['tvlUsd'].pct_change()
            return features.bfill()
        
        try:
            train_features = create_features(train)
            test_features = create_features(pd.concat([train, test]))[-len(test):]
            
            # Train model
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            model.fit(train_features, train['apy'])
            
            # Predict
            predictions = model.predict(test_features)
            mae = mean_absolute_error(test['apy'], predictions)
            rmse = np.sqrt(mean_squared_error(test['apy'], predictions))
            return {'mae': mae, 'rmse': rmse, 'predictions': predictions.tolist()}
        except Exception as e:
            print(f"    XGBoost failed: {e}")
            return None
    
    def evaluate_protocol(self, protocol_id, test_size=3):
        """Run all forecasting methods on a single protocol"""
        # Filter data for this protocol
        protocol_data = self.df[self.df['pool'] == protocol_id].copy()
        protocol_data = protocol_data.sort_values('date').reset_index(drop=True)
        
        # Rename column to standard 'apy'
        if "apyBase" in protocol_data.columns:
            apy_col = "apyBase"
        elif "apy" in protocol_data.columns:
            apy_col = "apy"
        elif "apy_total" in protocol_data.columns:
            apy_col = "apy_total"
        else:
            raise ValueError(
                f"No APY column found for protocol {protocol_id}: {protocol_data.columns}"
            )

        protocol_data["apy"] = protocol_data[apy_col]

        
        # Train/test split
        train = protocol_data.iloc[:-test_size]
        test = protocol_data.iloc[-test_size:]
        
        if len(train) < 7:
            print(f"  ⚠️  Insufficient training data ({len(train)} days)")
            return None
        
        print(f"\n  Protocol: {protocol_id[:20]}...")
        print(f"  Train: {len(train)} days | Test: {len(test)} days")
        print(f"  APY range: {protocol_data['apy'].min():.2f}% - {protocol_data['apy'].max():.2f}%")
        
        # Test all models
        results = {
            'protocol': protocol_id,
            'train_size': len(train),
            'test_size': len(test),
            'apy_mean': protocol_data['apy'].mean(),
            'apy_std': protocol_data['apy'].std()
        }
        
        # Naive baseline
        naive = self.naive_forecast(train, test)
        results['naive_mae'] = naive['mae']
        results['naive_rmse'] = naive['rmse']
        
        # Historical mean
        hist_mean = self.historical_mean_forecast(train, test)
        results['hist_mean_mae'] = hist_mean['mae']
        
        # ARIMA
        arima = self.arima_forecast(train, test)
        if arima:
            results['arima_mae'] = arima['mae']
            results['arima_rmse'] = arima['rmse']
            results['arima_improvement'] = (naive['mae'] - arima['mae']) / naive['mae'] * 100
        
        # XGBoost
        xgb = self.xgboost_forecast(train, test)
        if xgb:
            results['xgb_mae'] = xgb['mae']
            results['xgb_rmse'] = xgb['rmse']
            results['xgb_improvement'] = (naive['mae'] - xgb['mae']) / naive['mae'] * 100
        
        print(f"  Naive MAE: {naive['mae']:.4f} | ARIMA: {results.get('arima_mae', 'N/A')} | XGBoost: {results.get('xgb_mae', 'N/A')}")
        
        return results
    
    def run_evaluation(self, protocols=None):
        """Evaluate forecasting performance across protocols"""
        if protocols is None:
            protocols = self.select_protocols()
        
        print(f"\n{'='*60}")
        print("FORECASTING EVALUATION")
        print(f"{'='*60}")
        
        for protocol_id in protocols:
            result = self.evaluate_protocol(protocol_id)
            if result:
                self.results.append(result)
        
        # Aggregate results
        results_df = pd.DataFrame(self.results)
        
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Protocols evaluated: {len(results_df)}")
        print(f"\nAverage MAE:")
        print(f"  Naive:     {results_df['naive_mae'].mean():.4f}%")
        if 'arima_mae' in results_df.columns:
            print(f"  ARIMA:     {results_df['arima_mae'].mean():.4f}%")
            print(f"  Improvement: {results_df['arima_improvement'].mean():.2f}%")
        if 'xgb_mae' in results_df.columns:
            print(f"  XGBoost:   {results_df['xgb_mae'].mean():.4f}%")
            print(f"  Improvement: {results_df['xgb_improvement'].mean():.2f}%")
        
        return results_df
    
    def save_results(self, results_df, output_path='results/forecasting_performance.csv'):
        """Save evaluation results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")

def main():
    """Run forecasting evaluation"""
    forecaster = YieldForecaster()
    forecaster.load_data()
    results = forecaster.run_evaluation()
    forecaster.save_results(results)
    
    return results

# ============================================================================
# MAIN / PUBLIC API
# ============================================================================

# ============================================================================
# MAIN / PUBLIC API
# ============================================================================

def run_forecasting_suite(panel, skip_lstm: bool = False):
    """
    Public wrapper used by main.py for the full yield forecasting stage.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel/timeseries dataset with protocol-level yields.
    skip_lstm : bool, optional
        If True, skip slow LSTM training and run only faster models.

    Returns
    -------
    pd.DataFrame
        Forecast results for all models/horizons used in the pipeline.
    """
    from pathlib import Path
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    panel.to_csv("data/processed/yield_panel.csv", index=False)

    forecaster = YieldForecaster(panel, skip_lstm=skip_lstm)
    return forecaster.run()


if __name__ == '__main__':
    results = main()