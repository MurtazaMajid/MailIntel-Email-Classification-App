"""
Automated Model Retraining Pipeline for Scotland Birth Rate Forecasting
======================================================================

This module provides automated retraining capabilities including:
- Scheduled model retraining
- Data freshness validation
- Model performance monitoring
- Automatic model deployment
- Retraining history tracking
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_model import train_model, prepare_data
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRetrainingPipeline:
    """Automated model retraining and monitoring pipeline"""
    
    def __init__(self, config_file='retraining_config.json'):
        """Initialize the retraining pipeline"""
        self.config_file = config_file
        self.config = self._load_config()
        self.history_file = 'retraining_history.json'
        self.model_path = 'birth_model.pkl'
        self.backup_model_path = 'birth_model_backup.pkl'
        
    def _load_config(self):
        """Load retraining configuration"""
        default_config = {
            "retraining_frequency_days": 30,
            "min_data_age_days": 7,
            "performance_threshold": 0.15,  # 15% performance degradation triggers retraining
            "min_improvement_threshold": 0.05,  # 5% minimum improvement to deploy new model
            "data_validation": {
                "min_records": 100,
                "max_missing_rate": 0.1,
                "seasonal_pattern_check": True
            },
            "model_validation": {
                "cross_validation_folds": 5,
                "test_size": 0.2,
                "min_r2_score": 0.3
            },
            "auto_deploy": True,
            "backup_previous_model": True
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
                return default_config
        else:
            # Save default config
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _save_history(self, entry):
        """Save retraining history entry"""
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        
        history.append(entry)
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _load_history(self):
        """Load retraining history"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def load_data(self):
        """Load the three main datasets"""
        try:
            births_df = pd.read_csv('births.csv')
            births_df['Date'] = pd.to_datetime(births_df['Date'])
            
            unemployment_df = pd.read_csv('unemployment.csv')
            
            holidays_df = pd.read_csv('holidays.csv')
            holidays_df['date'] = pd.to_datetime(holidays_df['date'])
            
            return births_df, unemployment_df, holidays_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None, None
    
    def check_data_freshness(self):
        """Check if new data is available for retraining"""
        try:
            # Load current data
            births_df, _, _ = self.load_data()
            
            if births_df is None:
                return {'has_fresh_data': False, 'error': 'Could not load data'}
            
            # Check latest data date
            latest_data_date = births_df['Date'].max()
            days_since_latest = (datetime.now() - latest_data_date).days
            
            logger.info(f"Latest data date: {latest_data_date}")
            logger.info(f"Days since latest data: {days_since_latest}")
            
            return {
                'has_fresh_data': days_since_latest <= self.config['min_data_age_days'],
                'latest_date': latest_data_date.isoformat(),
                'days_since_latest': days_since_latest,
                'record_count': len(births_df)
            }
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return {'has_fresh_data': False, 'error': str(e)}
    
    def validate_data_quality(self):
        """Validate data quality for retraining"""
        try:
            # Load and prepare data
            births_df, unemployment_df, holidays_df = self.load_data()
            
            validation_results = {
                'passed': True,
                'issues': [],
                'metrics': {}
            }
            
            # Check minimum record count
            if len(births_df) < self.config['data_validation']['min_records']:
                validation_results['passed'] = False
                validation_results['issues'].append(
                    f"Insufficient records: {len(births_df)} < {self.config['data_validation']['min_records']}"
                )
            
            # Check missing data rate
            missing_rate = births_df['Births'].isnull().sum() / len(births_df)
            validation_results['metrics']['missing_rate'] = missing_rate
            
            if missing_rate > self.config['data_validation']['max_missing_rate']:
                validation_results['passed'] = False
                validation_results['issues'].append(
                    f"High missing data rate: {missing_rate:.2%} > {self.config['data_validation']['max_missing_rate']:.2%}"
                )
            
            # Check for data anomalies
            births_stats = births_df['Births'].describe()
            q1, q3 = births_stats['25%'], births_stats['75%']
            iqr = q3 - q1
            outlier_threshold = q3 + 1.5 * iqr
            outliers = (births_df['Births'] > outlier_threshold).sum()
            outlier_rate = outliers / len(births_df)
            
            validation_results['metrics']['outlier_rate'] = outlier_rate
            validation_results['metrics']['births_stats'] = births_stats.to_dict()
            
            if outlier_rate > 0.1:  # More than 10% outliers
                validation_results['issues'].append(
                    f"High outlier rate: {outlier_rate:.2%}"
                )
            
            logger.info(f"Data validation results: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return {'passed': False, 'error': str(e)}
    
    def evaluate_current_model_performance(self):
        """Evaluate current model performance on latest data"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning("No current model found")
                return {'model_exists': False}
            
            # Load current model
            current_model = joblib.load(self.model_path)
            
            # Prepare latest data  
            data_prepared = prepare_data()
            if data_prepared is None:
                return {'model_exists': False, 'error': 'Could not prepare data'}
            
            births_df, X, y = data_prepared
            
            # Use last 20% of data for evaluation
            eval_size = max(10, int(len(X) * 0.2))
            X_eval = X.tail(eval_size)
            y_eval = y.tail(eval_size)
            
            # Make predictions
            y_pred = current_model.predict(X_eval)
            
            # Calculate metrics
            mae = mean_absolute_error(y_eval, y_pred)
            rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
            mape = np.mean(np.abs((y_eval - y_pred) / y_eval)) * 100
            
            performance = {
                'model_exists': True,
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'eval_samples': len(X_eval),
                'eval_period': f"{births_df['Date'].iloc[-eval_size]} to {births_df['Date'].max()}"
            }
            
            logger.info(f"Current model performance: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating current model: {e}")
            return {'model_exists': False, 'error': str(e)}
    
    def should_retrain(self):
        """Determine if model should be retrained"""
        reasons = []
        
        # Check last retraining date
        history = self._load_history()
        if history:
            last_training = datetime.fromisoformat(history[-1]['timestamp'])
            days_since_training = (datetime.now() - last_training).days
            
            if days_since_training >= self.config['retraining_frequency_days']:
                reasons.append(f"Scheduled retraining due ({days_since_training} days)")
        else:
            reasons.append("No previous training history found")
        
        # Check data freshness
        freshness = self.check_data_freshness()
        if not freshness.get('has_fresh_data', False):
            reasons.append("Fresh data available")
        
        # Check current model performance
        current_performance = self.evaluate_current_model_performance()
        if current_performance.get('model_exists', False):
            # Compare with historical performance
            if history:
                last_mae = history[-1].get('new_model_performance', {}).get('mae', float('inf'))
                current_mae = current_performance.get('mae', float('inf'))
                
                if current_mae > last_mae * (1 + self.config['performance_threshold']):
                    reasons.append(f"Performance degradation detected: {current_mae:.2f} vs {last_mae:.2f}")
        
        should_retrain = len(reasons) > 0
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'data_freshness': freshness,
            'current_performance': current_performance
        }
    
    def retrain_model(self):
        """Retrain the model with latest data"""
        logger.info("Starting model retraining...")
        start_time = datetime.now()
        
        try:
            # Validate data quality first
            validation = self.validate_data_quality()
            if not validation['passed']:
                raise Exception(f"Data validation failed: {validation['issues']}")
            
            # Backup current model if it exists
            if os.path.exists(self.model_path) and self.config['backup_previous_model']:
                import shutil
                shutil.copy2(self.model_path, self.backup_model_path)
                logger.info("Backed up current model")
            
            # Train new model
            model_info = train_model()
            
            # Evaluate new model performance
            new_performance = self.evaluate_current_model_performance()
            
            # Compare with previous model if available
            should_deploy = True
            comparison = {}
            
            if os.path.exists(self.backup_model_path):
                # Load backup model and compare
                backup_model = joblib.load(self.backup_model_path)
                
                # Prepare evaluation data
                data_prepared = prepare_data()
                if data_prepared is None:
                    should_deploy = True  # Deploy if we can't compare
                else:
                    births_df, X, y = data_prepared
                    eval_size = max(10, int(len(X) * 0.2))
                    X_eval = X.tail(eval_size)
                    y_eval = y.tail(eval_size)
                
                    # Compare predictions
                    new_pred = joblib.load(self.model_path).predict(X_eval)
                    old_pred = backup_model.predict(X_eval)
                    
                    new_mae = mean_absolute_error(y_eval, new_pred)
                    old_mae = mean_absolute_error(y_eval, old_pred)
                    
                    improvement = (old_mae - new_mae) / old_mae
                    
                    comparison = {
                        'old_mae': float(old_mae),
                        'new_mae': float(new_mae),
                        'improvement': float(improvement)
                    }
                    
                    # Check if improvement meets threshold
                    if improvement < self.config['min_improvement_threshold']:
                        should_deploy = False
                        logger.warning(f"New model improvement ({improvement:.2%}) below threshold ({self.config['min_improvement_threshold']:.2%})")
            
            # Deploy new model or restore backup
            if should_deploy and self.config['auto_deploy']:
                logger.info("Deploying new model")
                deployment_status = "deployed"
            else:
                if os.path.exists(self.backup_model_path):
                    import shutil
                    shutil.copy2(self.backup_model_path, self.model_path)
                    logger.info("Restored previous model")
                deployment_status = "reverted"
            
            # Record retraining history
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            history_entry = {
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration,
                'trigger_reasons': self.should_retrain()['reasons'],
                'data_validation': validation,
                'model_info': model_info,
                'new_model_performance': new_performance,
                'model_comparison': comparison,
                'deployment_status': deployment_status,
                'success': True
            }
            
            self._save_history(history_entry)
            
            logger.info(f"Model retraining completed in {duration:.2f} seconds")
            logger.info(f"Deployment status: {deployment_status}")
            
            return history_entry
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            
            # Record failure
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            history_entry = {
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration,
                'error': str(e),
                'success': False
            }
            
            self._save_history(history_entry)
            raise e
    
    def run_pipeline(self, force_retrain=False):
        """Run the complete retraining pipeline"""
        logger.info("Starting automated retraining pipeline")
        
        try:
            # Check if retraining is needed
            retrain_check = self.should_retrain()
            
            if force_retrain or retrain_check['should_retrain']:
                logger.info(f"Retraining triggered. Reasons: {retrain_check['reasons']}")
                result = self.retrain_model()
                
                return {
                    'pipeline_executed': True,
                    'retraining_performed': True,
                    'retrain_check': retrain_check,
                    'result': result
                }
            else:
                logger.info("No retraining needed")
                return {
                    'pipeline_executed': True,
                    'retraining_performed': False,
                    'retrain_check': retrain_check
                }
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                'pipeline_executed': False,
                'error': str(e)
            }
    
    def get_retraining_status(self):
        """Get current retraining status and history"""
        history = self._load_history()
        
        status = {
            'total_retrainings': len(history),
            'last_retraining': history[-1] if history else None,
            'next_scheduled_check': None,
            'current_model_age_days': None
        }
        
        if history:
            last_training = datetime.fromisoformat(history[-1]['timestamp'])
            status['current_model_age_days'] = (datetime.now() - last_training).days
            
            next_check = last_training + timedelta(days=self.config['retraining_frequency_days'])
            status['next_scheduled_check'] = next_check.isoformat()
        
        return status

def run_retraining_pipeline(force=False):
    """Convenience function to run the retraining pipeline"""
    pipeline = ModelRetrainingPipeline()
    return pipeline.run_pipeline(force_retrain=force)

def get_retraining_status():
    """Convenience function to get retraining status"""
    pipeline = ModelRetrainingPipeline()
    return pipeline.get_retraining_status()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Model Retraining Pipeline")
    parser.add_argument("--force", action="store_true", help="Force retraining even if not scheduled")
    parser.add_argument("--status", action="store_true", help="Show retraining status")
    parser.add_argument("--config", help="Path to config file")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_retraining_status()
        print(json.dumps(status, indent=2))
    else:
        result = run_retraining_pipeline(force=args.force)
        print(json.dumps(result, indent=2))