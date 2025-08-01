# Overview

This is a Scotland Birth Rate Forecasting System built with Streamlit that predicts birth rates using machine learning. The application fetches real-world data from government APIs, processes it with feature engineering, trains an XGBoost regression model, and provides an interactive dashboard for visualizing historical trends and future predictions. The system combines birth statistics with economic indicators (unemployment rates) and calendar events (holidays) to make informed forecasts.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit-based web application with wide layout configuration
- **Visualization**: Plotly for interactive charts and graphs including time series plots, bar charts, line plots, and heatmaps
- **Caching**: Streamlit's `@st.cache_data` decorator for performance optimization of data loading operations
- **User Interface**: Multi-section dashboard with EDA analysis, prediction comparisons, data integrity checks and model training controls
- **EDA Features**: Comprehensive exploratory data analysis with births by month/quarter/year, statistical summaries, and trend visualizations
- **Prediction Analysis**: Side-by-side comparison charts showing actual vs predicted values with forecast statistics

## Backend Architecture
- **Data Processing**: Pandas for data manipulation and feature engineering
- **Machine Learning**: XGBoost regression model for time series forecasting
- **Model Persistence**: Joblib for saving and loading trained models
- **Feature Engineering**: Cyclical encoding for seasonal patterns, unemployment rate mapping, and holiday counting

## Data Pipeline
- **Data Fetching**: Automated scripts to pull data from external APIs
- **Data Storage**: CSV files for local data persistence (births.csv, unemployment.csv, holidays.csv, predictions.csv)
- **Data Preparation**: Feature engineering pipeline that creates temporal features, maps economic indicators, and handles missing values
- **Time Series Processing**: Conversion from yearly to monthly data with realistic seasonal distribution patterns

## Model Architecture
- **Algorithm**: XGBoost Regressor for handling non-linear relationships and feature interactions
- **Features**: Year scaling, monthly cyclical encoding, unemployment rates, holiday counts, and quarterly indicators
- **Validation**: Time-series aware train/test split to prevent data leakage
- **Prediction Pipeline**: Generates future forecasts and saves results to CSV

# External Dependencies

## APIs and Data Sources
- **UK Government Bank Holidays API**: `https://www.gov.uk/bank-holidays.json` for Scotland holiday data
- **World Bank API**: `http://api.worldbank.org/v2/country/GB/indicator/SL.UEM.TOTL.ZS` for UK unemployment statistics

## Python Libraries
- **Web Framework**: Streamlit for dashboard creation
- **Data Science**: Pandas, NumPy for data manipulation
- **Machine Learning**: XGBoost, scikit-learn for model training and evaluation
- **Visualization**: Plotly Express and Plotly Graph Objects for interactive charts
- **HTTP Requests**: Requests library for API data fetching
- **Model Persistence**: Joblib for model serialization

## Data Files
- **births.csv**: Historical birth records with dates and counts
- **unemployment.csv**: Economic indicator data by year
- **holidays.csv**: Scotland public holiday calendar
- **predictions.csv**: Model-generated forecasts
- **births_prepared.csv**: Feature-engineered dataset for training