import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def check_data_integrity():
    """Check the integrity and completeness of loaded data"""
    integrity_report = {}
    
    # Check births data
    if os.path.exists('births.csv'):
        births = pd.read_csv('births.csv')
        births['Date'] = pd.to_datetime(births['Date'])
        
        integrity_report['births'] = {
            'file_exists': True,
            'record_count': len(births),
            'date_range': f"{births['Date'].min()} to {births['Date'].max()}",
            'missing_values': births.isnull().sum().sum(),
            'avg_monthly_births': births['Births'].mean()
        }
    else:
        integrity_report['births'] = {'file_exists': False}
    
    # Check unemployment data
    if os.path.exists('unemployment.csv'):
        unemployment = pd.read_csv('unemployment.csv')
        
        integrity_report['unemployment'] = {
            'file_exists': True,
            'record_count': len(unemployment),
            'year_range': f"{unemployment['Year'].min()} to {unemployment['Year'].max()}",
            'missing_values': unemployment.isnull().sum().sum(),
            'avg_rate': unemployment['Rate'].mean()
        }
    else:
        integrity_report['unemployment'] = {'file_exists': False}
    
    # Check holidays data
    if os.path.exists('holidays.csv'):
        holidays = pd.read_csv('holidays.csv')
        holidays['date'] = pd.to_datetime(holidays['date'])
        
        integrity_report['holidays'] = {
            'file_exists': True,
            'record_count': len(holidays),
            'date_range': f"{holidays['date'].min()} to {holidays['date'].max()}",
            'missing_values': holidays.isnull().sum().sum()
        }
    else:
        integrity_report['holidays'] = {'file_exists': False}
    
    # Check model file
    integrity_report['model'] = {
        'file_exists': os.path.exists('birth_model.pkl'),
        'file_size': os.path.getsize('birth_model.pkl') if os.path.exists('birth_model.pkl') else 0
    }
    
    # Check predictions
    if os.path.exists('predictions.csv'):
        predictions = pd.read_csv('predictions.csv')
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        integrity_report['predictions'] = {
            'file_exists': True,
            'record_count': len(predictions),
            'date_range': f"{predictions['Date'].min()} to {predictions['Date'].max()}",
            'missing_values': predictions.isnull().sum().sum()
        }
    else:
        integrity_report['predictions'] = {'file_exists': False}
    
    return integrity_report

def validate_date_range(start_date, end_date):
    """Validate that date range is reasonable"""
    if start_date > end_date:
        return False, "Start date must be before end date"
    
    if (end_date - start_date).days > 3650:  # More than 10 years
        return False, "Date range too large (max 10 years)"
    
    return True, "Valid date range"

def calculate_birth_statistics(births_df):
    """Calculate comprehensive birth statistics"""
    stats = {}
    
    if len(births_df) == 0:
        return stats
    
    # Basic statistics
    stats['total_births'] = births_df['Births'].sum()
    stats['avg_monthly_births'] = births_df['Births'].mean()
    stats['median_monthly_births'] = births_df['Births'].median()
    stats['std_monthly_births'] = births_df['Births'].std()
    stats['min_monthly_births'] = births_df['Births'].min()
    stats['max_monthly_births'] = births_df['Births'].max()
    
    # Date range
    births_df['Date'] = pd.to_datetime(births_df['Date'])
    stats['date_range_start'] = births_df['Date'].min()
    stats['date_range_end'] = births_df['Date'].max()
    stats['total_months'] = len(births_df)
    
    # Seasonal analysis
    births_df['Month'] = births_df['Date'].dt.month
    monthly_avg = births_df.groupby('Month')['Births'].mean()
    stats['peak_birth_month'] = monthly_avg.idxmax()
    stats['lowest_birth_month'] = monthly_avg.idxmin()
    stats['seasonal_variation'] = monthly_avg.max() - monthly_avg.min()
    
    # Year-over-year trends
    births_df['Year'] = births_df['Date'].dt.year
    yearly_totals = births_df.groupby('Year')['Births'].sum()
    if len(yearly_totals) > 1:
        stats['year_over_year_change'] = yearly_totals.pct_change().mean()
        stats['trend_direction'] = 'increasing' if stats['year_over_year_change'] > 0 else 'decreasing'
    else:
        stats['year_over_year_change'] = 0
        stats['trend_direction'] = 'stable'
    
    return stats

def format_number(number, decimal_places=0):
    """Format numbers for display with appropriate commas and decimals"""
    if decimal_places == 0:
        return f"{number:,.0f}"
    else:
        return f"{number:,.{decimal_places}f}"

def get_data_freshness():
    """Check how fresh the data files are"""
    freshness = {}
    files_to_check = ['births.csv', 'unemployment.csv', 'holidays.csv', 'predictions.csv']
    
    for file in files_to_check:
        if os.path.exists(file):
            mod_time = os.path.getmtime(file)
            mod_datetime = datetime.fromtimestamp(mod_time)
            age_hours = (datetime.now() - mod_datetime).total_seconds() / 3600
            
            freshness[file] = {
                'exists': True,
                'last_modified': mod_datetime,
                'age_hours': age_hours,
                'is_fresh': age_hours < 24  # Consider fresh if less than 24 hours old
            }
        else:
            freshness[file] = {
                'exists': False,
                'last_modified': None,
                'age_hours': None,
                'is_fresh': False
            }
    
    return freshness

def generate_summary_report():
    """Generate a comprehensive summary report of the system status"""
    report = {
        'timestamp': datetime.now(),
        'data_integrity': check_data_integrity(),
        'data_freshness': get_data_freshness()
    }
    
    # Add birth statistics if data exists
    if os.path.exists('births.csv'):
        births = pd.read_csv('births.csv')
        report['birth_statistics'] = calculate_birth_statistics(births)
    
    return report

def clean_data_files():
    """Clean up temporary or corrupted data files"""
    files_to_clean = ['births_prepared.csv']  # Add other temp files as needed
    
    cleaned = []
    for file in files_to_clean:
        if os.path.exists(file):
            try:
                os.remove(file)
                cleaned.append(file)
            except Exception:
                pass
    
    return cleaned

if __name__ == "__main__":
    # Run diagnostics when script is executed directly
    print("🔍 Running system diagnostics...")
    
    report = generate_summary_report()
    print(f"📊 Report generated at: {report['timestamp']}")
    
    for dataset, info in report['data_integrity'].items():
        if info.get('file_exists', False):
            print(f"✅ {dataset}: {info.get('record_count', 'N/A')} records")
        else:
            print(f"❌ {dataset}: Missing")
    
    print("\n🔧 System ready for operation")
