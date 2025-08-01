import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def prepare_data():
    """Load and feature-engineer the data for time-series split"""
    print("📊 Loading and preparing data...")
    try:
        births = pd.read_csv('births.csv', parse_dates=['Date'])
        unemployment = pd.read_csv('unemployment.csv')
        holidays = pd.read_csv('holidays.csv', parse_dates=['date'])
    except FileNotFoundError as e:
        print(f"❌ Missing file: {e}")
        print("💡 Run the data fetcher first to download required datasets")
        return None

    print(f"   - Births: {len(births)} records")
    print(f"   - Unemployment: {len(unemployment)} records")
    print(f"   - Holidays: {len(holidays)} records")

    # Feature engineer births data
    births['Year'] = births['Date'].dt.year
    births['Month'] = births['Date'].dt.month
    births['Quarter'] = births['Date'].dt.quarter

    # Map unemployment rate by year
    unemployment['Year'] = unemployment['Year'].astype(int)
    rate_map = dict(zip(unemployment['Year'], unemployment['Rate']))
    births['Unemployment_Rate'] = births['Year'].map(rate_map)
    
    # Fill missing unemployment rates with mean
    missing_unemployment = births['Unemployment_Rate'].isna().sum()
    if missing_unemployment > 0:
        print(f"   - Filling {missing_unemployment} missing unemployment values with mean")
        births['Unemployment_Rate'].fillna(births['Unemployment_Rate'].mean(), inplace=True)

    # Count holidays per month
    holidays['Year'] = holidays['date'].dt.year
    holidays['Month'] = holidays['date'].dt.month
    holiday_counts = holidays.groupby(['Year', 'Month']).size().reset_index(name='Holiday_Count')
    births = births.merge(holiday_counts, on=['Year', 'Month'], how='left')
    births['Holiday_Count'].fillna(0, inplace=True)

    # Create cyclical month features for seasonality
    births['Month_Sin'] = np.sin(2 * np.pi * births['Month'] / 12)
    births['Month_Cos'] = np.cos(2 * np.pi * births['Month'] / 12)
    
    # Scale year feature
    year_min, year_max = births['Year'].min(), births['Year'].max()
    if year_max > year_min:
        births['Year_Scaled'] = (births['Year'] - year_min) / (year_max - year_min)
    else:
        births['Year_Scaled'] = 0

    # Sort by date for proper time-series split
    births.sort_values('Date', inplace=True)
    
    # Save prepared data for inspection
    births.to_csv('births_prepared.csv', index=False)
    
    # Define features and target
    features = ['Year_Scaled', 'Month', 'Quarter', 'Month_Sin', 'Month_Cos', 
                'Unemployment_Rate', 'Holiday_Count']
    X = births[features]
    y = births['Births']

    print(f"✅ Prepared {len(X)} samples with {len(features)} features")
    print(f"   Features: {', '.join(features)}")
    return births, X, y

def train_model():
    """Train XGBoost model with time-series split and report performance"""
    data = prepare_data()
    if data is None:
        return None
    
    births, X, y = data

    # Time-series split: use first 80% for training, last 20% for testing
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\n📊 Data Split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    if len(X_test) == 0:
        print("❌ Test set is empty. Need more data or smaller test_size.")
        return None

    # Configure XGBoost with regularization to prevent overfitting
    model = XGBRegressor(
        n_estimators=50,        # Fewer trees to prevent overfitting
        max_depth=3,            # Shallow trees
        learning_rate=0.05,     # Lower learning rate
        reg_alpha=0.1,          # L1 regularization
        reg_lambda=1.0,         # L2 regularization
        random_state=42,
        verbosity=0
    )
    
    print("🤖 Training XGBoost model...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate performance metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\n📈 Model Performance:")
    print(f"   Training MAE: {train_mae:.2f}")
    print(f"   Test MAE: {test_mae:.2f}")
    print(f"   Overfitting ratio: {test_mae/train_mae:.2f}")

    # Show feature importance
    feature_names = X.columns
    importance = model.feature_importances_
    print(f"\n🎯 Feature Importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"   {name}: {imp:.3f}")

    # Show sample predictions
    print(f"\n🔍 Sample Test Predictions:")
    test_births = births.iloc[split_index:split_index+min(5, len(X_test))].copy()
    test_births['Predicted'] = y_pred_test[:len(test_births)]
    test_births['Error'] = abs(test_births['Births'] - test_births['Predicted'])
    
    display_cols = ['Date', 'Births', 'Predicted', 'Error']
    print(test_births[display_cols].to_string(index=False))

    # Generate future predictions
    generate_predictions(model, births, X.columns)

    # Save the trained model
    joblib.dump(model, 'birth_model.pkl')
    print(f"\n💾 Model saved as 'birth_model.pkl'")
    
    return model

def generate_predictions(model, births, feature_columns):
    """Generate future predictions for the next 12 months"""
    print("\n🔮 Generating future predictions...")
    
    # Get the last known values for feature engineering
    last_date = births['Date'].max()
    last_year = births['Year'].max()
    last_unemployment = births['Unemployment_Rate'].iloc[-1]
    
    # Create future dates (next 12 months)
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq='MS'  # Month start
    )
    
    future_data = []
    for date in future_dates:
        year = date.year
        month = date.month
        quarter = date.quarter
        
        # Scale year based on training data range
        year_min = births['Year'].min()
        year_max = births['Year'].max()
        if year_max > year_min:
            year_scaled = (year - year_min) / (year_max - year_min)
        else:
            year_scaled = 0
        
        # Cyclical features
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Assume unemployment rate remains stable
        unemployment_rate = last_unemployment
        
        # Assume no holidays for simplicity (could be enhanced)
        holiday_count = 0
        
        future_data.append({
            'Date': date,
            'Year_Scaled': year_scaled,
            'Month': month,
            'Quarter': quarter,
            'Month_Sin': month_sin,
            'Month_Cos': month_cos,
            'Unemployment_Rate': unemployment_rate,
            'Holiday_Count': holiday_count
        })
    
    future_df = pd.DataFrame(future_data)
    X_future = future_df[feature_columns]
    
    # Make predictions
    future_predictions = model.predict(X_future)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Births': future_predictions.astype(int)
    })
    
    # Save predictions
    predictions_df.to_csv('predictions.csv', index=False)
    print(f"✅ Generated {len(predictions_df)} future predictions")
    print(f"   Saved to 'predictions.csv'")
    
    # Show sample future predictions
    print(f"\n📅 Next 6 Months Forecast:")
    sample_predictions = predictions_df.head(6)
    for _, row in sample_predictions.iterrows():
        print(f"   {row['Date'].strftime('%Y-%m')}: {row['Predicted_Births']:,} births")

def main():
    """Main training pipeline"""
    print("🚀 Starting birth rate forecasting model training")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    try:
        model = train_model()
        if model is not None:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"\n✅ Training completed successfully in {elapsed_time:.2f} seconds")
            print(f"🎯 Model ready for forecasting Scotland birth rates")
        else:
            print(f"\n❌ Training failed - check data files and try again")
    except Exception as e:
        print(f"\n💥 Training error: {e}")
        print(f"💡 Ensure data files are available and valid")

if __name__ == "__main__":
    main()
