# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
import json
try:
    from model_retraining import ModelRetrainingPipeline, get_retraining_status, run_retraining_pipeline
except ImportError:
    # Fallback for when retraining module is not available
    ModelRetrainingPipeline = None
    get_retraining_status = lambda: {"error": "Retraining module not available"}
    run_retraining_pipeline = lambda force=False: {"error": "Retraining module not available"}

# Import the new data fetcher
try:
    from data_fetcher import fetch_latest_data
except ImportError:
    # Fallback if the data fetcher script is not available
    def fetch_latest_data():
        st.error("❌ Data fetcher script `data_fetcher.py` not found!")
        return False, 0

@st.cache_data
def load_data():
    """Load all datasets with comprehensive error handling"""
    try:
        # Load births data
        births = pd.read_csv('births.csv')
        births['Date'] = pd.to_datetime(births['Date'])
        
        # Load unemployment data
        try:
            unemployment = pd.read_csv('unemployment.csv')
            unemployment['Year'] = pd.to_numeric(unemployment['Year'], errors='coerce').astype('Int64')
            unemployment = unemployment.dropna(subset=['Year', 'Rate'])
        except FileNotFoundError:
            st.warning("⚠️ Unemployment data not found")
            unemployment = pd.DataFrame(columns=['Year', 'Rate'])
        
        # Load holidays data
        try:
            holidays = pd.read_csv('holidays.csv')
            holidays['date'] = pd.to_datetime(holidays['date'], errors='coerce')
            holidays = holidays.dropna(subset=['date'])
        except FileNotFoundError:
            st.warning("⚠️ Holiday data not found")
            holidays = pd.DataFrame(columns=['date', 'title'])
        
        # Load predictions data
        try:
            predictions = pd.read_csv('predictions.csv')
            predictions['Date'] = pd.to_datetime(predictions['Date'])
            # Ensure the required column exists
            if 'Predicted_Births' not in predictions.columns:
                predictions = pd.DataFrame(columns=['Date', 'Predicted_Births'])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            predictions = pd.DataFrame(columns=['Date', 'Predicted_Births'])
        except Exception as e:
            st.warning(f"⚠️ Error loading predictions: {e}")
            predictions = pd.DataFrame(columns=['Date', 'Predicted_Births'])
        
        return births, unemployment, holidays, predictions
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def load_model():
    """Load the trained model if available"""
    try:
        if os.path.exists('birth_model.pkl'):
            model = joblib.load('birth_model.pkl')
            return model
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def create_unemployment_correlation(births, unemployment):
    """Create unemployment vs births correlation analysis"""
    try:
        if len(unemployment) == 0:
            st.info("📊 No unemployment data available for correlation analysis")
            return None
        
        # Aggregate births by year
        births_yearly = births.copy()
        births_yearly['Year'] = births_yearly['Date'].dt.year
        births_yearly = births_yearly.groupby('Year')['Births'].sum().reset_index()
        births_yearly.columns = ['Year', 'Total_Births']
        
        # Ensure compatible data types
        births_yearly['Year'] = births_yearly['Year'].astype('Int64')
        unemployment['Year'] = unemployment['Year'].astype('Int64')
        
        # Merge datasets
        merged = pd.merge(births_yearly, unemployment, on='Year', how='inner')
        
        if len(merged) == 0:
            st.info("📊 No overlapping years found between births and unemployment data")
            return None
        
        # Create scatter plot with trendline
        fig = px.scatter(
            merged,
            x='Rate',
            y='Total_Births',
            trendline='ols',
            hover_data=['Year'],
            labels={
                'Rate': 'Unemployment Rate (%)',
                'Total_Births': 'Total Births per Year'
            }
        )
        
        # Update layout to match heatmap formatting
        fig.update_layout(
            title=dict(
                text="Unemployment Rate vs Birth Rate Correlation",
                font=dict(size=14, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=10, color='#2c3e50'),
            height=350,  # Match heatmap height
            margin=dict(l=50, r=50, t=60, b=50),
            xaxis=dict(
                title="Unemployment Rate (%)",
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Total Births per Year",
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=10)
            )
        )
        
        # Add correlation coefficient
        correlation = merged['Rate'].corr(merged['Total_Births'])
        fig.add_annotation(
            x=0.05, y=0.95,
            xref="paper", yref="paper",
            text=f"Correlation: {correlation:.3f}",
            showarrow=False,
            font=dict(size=10, color='#2c3e50'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            borderpad=4
        )
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating unemployment correlation: {e}")
        return None

def create_comprehensive_time_series(births):
    """Create a comprehensive time series analysis chart"""
    try:
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Monthly Birth Counts Over Time', 'Yearly Birth Trends'],
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Monthly time series (top plot)
        fig.add_trace(
            go.Scatter(
                x=births['Date'],
                y=births['Births'],
                mode='lines+markers',
                name='Monthly Births',
                line=dict(color='#005EB8', width=2),
                marker=dict(size=4, color='#005EB8', line=dict(width=1, color='white')),
                hovertemplate='<b>%{x}</b><br>Births: %{y:,}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving average
        births_sorted = births.sort_values('Date')
        births_sorted['MA_12'] = births_sorted['Births'].rolling(window=12, center=True).mean()
        
        fig.add_trace(
            go.Scatter(
                x=births_sorted['Date'],
                y=births_sorted['MA_12'],
                mode='lines',
                name='12-Month Moving Average',
                line=dict(color='#FF6B35', width=3),
                hovertemplate='<b>%{x}</b><br>12-Month Avg: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Yearly aggregation (bottom plot)
        yearly_births = births.copy()
        yearly_births['Year'] = yearly_births['Date'].dt.year
        yearly_summary = yearly_births.groupby('Year')['Births'].sum().reset_index()
        
        fig.add_trace(
            go.Bar(
                x=yearly_summary['Year'],
                y=yearly_summary['Births'],
                name='Annual Total Births',
                marker=dict(color='#005EB8', opacity=0.7),
                hovertemplate='<b>%{x}</b><br>Total Births: %{y:,}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Scotland Birth Rate Time Series Analysis",
                font=dict(size=20, color='#2c3e50'),
                x=0.5
            ),
            height=700,
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12, color='#2c3e50'),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
        fig.update_yaxes(title_text="Number of Births", gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
        fig.update_xaxes(title_text="Year", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
        fig.update_yaxes(title_text="Annual Total", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating comprehensive time series: {e}")
        return None

def create_time_series_chart(births, predictions, date_range, show_predictions):
    """Create main time series visualization"""
    try:
        # Filter births data by date range
        if len(date_range) == 2:
            # Convert date objects to proper datetime for comparison
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            mask = (births['Date'] >= start_date) & (births['Date'] <= end_date)
            filtered_births = births[mask]
        else:
            filtered_births = births
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data with enhanced styling
        fig.add_trace(go.Scatter(
            x=filtered_births['Date'],
            y=filtered_births['Births'],
            mode='lines+markers',
            name='Historical Births',
            line=dict(color='#005EB8', width=3),  # Scotland blue
            marker=dict(size=6, color='#005EB8', line=dict(width=2, color='white')),
            hovertemplate='<b>Date:</b> %{x}<br><b>Births:</b> %{y:,}<extra></extra>'
        ))
        
        # Add predictions if available and requested
        if show_predictions and len(predictions) > 0:
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Births'],
                mode='lines+markers',
                name='Forecasted Births',
                line=dict(color='#FF6B35', dash='dash', width=3),  # Vibrant orange
                marker=dict(size=6, symbol='diamond', color='#FF6B35', line=dict(width=2, color='white')),
                hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:,}<extra></extra>'
            ))
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text="Scotland Birth Rate Trends and Forecasts",
                font=dict(size=20, color='#2c3e50'),
                x=0.5
            ),
            xaxis_title="Date",
            yaxis_title="Number of Births",
            hovermode='x unified',
            showlegend=True,
            height=500,
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12, color='#2c3e50'),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            xaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating time series chart: {e}")
        return None

def create_seasonal_heatmap(births):
    """Create seasonal pattern heatmap"""
    try:
        births_copy = births.copy()
        births_copy['Year'] = births_copy['Date'].dt.year
        births_copy['Month'] = births_copy['Date'].dt.month
        
        # Create pivot table for heatmap
        seasonal_data = births_copy.pivot_table(
            values='Births', 
            index='Year', 
            columns='Month', 
            aggfunc='mean'
        )
        
        # Create heatmap with enhanced styling
        fig = px.imshow(
            seasonal_data,
            labels=dict(x="Month", y="Year", color="Births"),
            aspect="auto",
            color_continuous_scale=[[0, '#f8f9fa'], [0.3, '#85c1e9'], [0.6, '#3498db'], [1, '#1a5490']]
        )
        
        # Update layout with enhanced styling and proper title
        fig.update_layout(
            title=dict(
                text="Seasonal Birth Patterns by Month and Year",
                font=dict(size=14, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Month",
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                tickfont=dict(color='#2c3e50', size=10)
            ),
            yaxis=dict(
                title="Year",
                tickfont=dict(color='#2c3e50', size=10)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=10, color='#2c3e50'),
            height=350,  # Fixed height to match correlation plot
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating seasonal heatmap: {e}")
        return None

def create_holiday_chart(holidays):
    """Create holiday distribution chart"""
    try:
        if len(holidays) == 0:
            st.info("📊 No holiday data available")
            return None
        
        holidays_copy = holidays.copy()
        holidays_copy['Month'] = holidays_copy['date'].dt.month
        holidays_monthly = holidays_copy.groupby('Month').size().reset_index(name='Holiday_Count')
        
        fig = px.bar(
            holidays_monthly,
            x='Month',
            y='Holiday_Count',
            title="Number of Holidays by Month",
            labels={'Month': 'Month', 'Holiday_Count': 'Number of Holidays'}
        )
        
        # Update x-axis to show month names
        fig.update_layout(
            xaxis=dict(
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating holiday chart: {e}")
        return None

def create_births_by_month_chart(births):
    """Create average births by month chart"""
    try:
        births_copy = births.copy()
        births_copy['Month'] = births_copy['Date'].dt.month
        monthly_avg = births_copy.groupby('Month')['Births'].agg(['mean', 'std']).reset_index()
        
        fig = px.bar(
            monthly_avg,
            x='Month',
            y='mean',
            error_y='std',
            labels={'mean': 'Average Births', 'Month': 'Month'},
            color_discrete_sequence=['#005EB8']
        )
        
        # Update layout with proper formatting
        fig.update_layout(
            title=dict(
                text="Average Births by Month",
                font=dict(size=14, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Month",
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Average Births",
                tickfont=dict(size=10)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=10, color='#2c3e50'),
            height=300,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating births by month chart: {e}")
        return None

def create_births_by_year_chart(births):
    """Create total births by year chart"""
    try:
        births_copy = births.copy()
        births_copy['Year'] = births_copy['Date'].dt.year
        yearly_totals = births_copy.groupby('Year')['Births'].sum().reset_index()
        
        fig = px.line(
            yearly_totals,
            x='Year',
            y='Births',
            markers=True,
            labels={'Births': 'Total Births', 'Year': 'Year'},
            color_discrete_sequence=['#005EB8']
        )
        
        # Update layout with proper formatting
        fig.update_layout(
            title=dict(
                text="Total Births by Year - Long-term Trends",
                font=dict(size=16, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Year",
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title="Total Births",
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=11)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=11, color='#2c3e50'),
            height=400,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Style the line
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=6, line=dict(width=2, color='white'))
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating births by year chart: {e}")
        return None

def create_births_by_quarter_chart(births):
    """Create average births by quarter chart"""
    try:
        births_copy = births.copy()
        births_copy['Quarter'] = births_copy['Date'].dt.quarter
        quarterly_avg = births_copy.groupby('Quarter')['Births'].mean().reset_index()
        
        fig = px.bar(
            quarterly_avg,
            x='Quarter',
            y='Births',
            labels={'Births': 'Average Births', 'Quarter': 'Quarter'},
            color_discrete_sequence=['#FF6B35']
        )
        
        # Update layout with proper formatting
        fig.update_layout(
            title=dict(
                text="Average Births by Quarter",
                font=dict(size=14, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Quarter",
                tickvals=[1, 2, 3, 4],
                ticktext=['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)'],
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Average Births",
                tickfont=dict(size=10)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=10, color='#2c3e50'),
            height=300,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating births by quarter chart: {e}")
        return None

def calculate_birth_statistics(births):
    """Calculate comprehensive birth statistics"""
    try:
        births_copy = births.copy()
        births_copy['Year'] = births_copy['Date'].dt.year
        
        # Overall statistics
        stats = {
            'total_births': births['Births'].sum(),
            'avg_monthly_births': births['Births'].mean(),
            'max_monthly_births': births['Births'].max(),
            'min_monthly_births': births['Births'].min(),
            'std_monthly_births': births['Births'].std()
        }
        
        # Yearly statistics
        yearly_data = births_copy.groupby('Year')['Births'].agg(['sum', 'mean', 'max', 'min']).reset_index()
        stats['avg_yearly_births'] = yearly_data['sum'].mean()
        stats['max_yearly_births'] = yearly_data['sum'].max()
        stats['min_yearly_births'] = yearly_data['sum'].min()
        stats['max_year'] = yearly_data.loc[yearly_data['sum'].idxmax(), 'Year']
        stats['min_year'] = yearly_data.loc[yearly_data['sum'].idxmin(), 'Year']
        
        return stats
    except Exception as e:
        st.error(f"❌ Error calculating statistics: {e}")
        return {}



def apply_scenario_filters(births, unemployment, unemployment_rate, use_unemployment_filter, 
                          selected_months, selected_years):
    """Apply all scenario filters to birth data"""
    try:
        filtered_births = births.copy()
        filter_info = []
        
        # Add date components for filtering
        filtered_births['Year'] = filtered_births['Date'].dt.year
        filtered_births['Month'] = filtered_births['Date'].dt.month
        
        # Apply year filter
        if selected_years and len(selected_years) == 2:
            min_year, max_year = selected_years
            filtered_births = filtered_births[
                (filtered_births['Year'] >= min_year) & 
                (filtered_births['Year'] <= max_year)
            ]
            if min_year != max_year:
                filter_info.append(f"Years: {min_year}-{max_year}")
            else:
                filter_info.append(f"Year: {min_year}")
        
        # Apply month filter
        if selected_months and len(selected_months) < 12:
            filtered_births = filtered_births[filtered_births['Month'].isin(selected_months)]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            selected_month_names = [month_names[m-1] for m in selected_months]
            filter_info.append(f"Months: {', '.join(selected_month_names)}")
        
        # Apply unemployment filter
        if use_unemployment_filter and len(unemployment) > 0:
            unemployment_dict = dict(zip(unemployment['Year'], unemployment['Rate']))
            filtered_births['Unemployment_Rate'] = filtered_births['Year'].map(unemployment_dict)
            
            # Filter by unemployment rate within tolerance
            tolerance = 0.5
            min_rate = unemployment_rate - tolerance
            max_rate = unemployment_rate + tolerance
            
            filtered_births = filtered_births[
                (filtered_births['Unemployment_Rate'] >= min_rate) & 
                (filtered_births['Unemployment_Rate'] <= max_rate)
            ]
            filter_info.append(f"Unemployment: {min_rate:.1f}%-{max_rate:.1f}%")
        
        # Clean up temporary columns
        cols_to_drop = ['Year', 'Month', 'Unemployment_Rate']
        for col in cols_to_drop:
            if col in filtered_births.columns:
                filtered_births = filtered_births.drop(col, axis=1)
        
        return filtered_births, filter_info
        
    except Exception as e:
        st.error(f"Error applying filters: {e}")
        return births, []

def apply_custom_css():
    """Apply custom CSS styling for better visual appeal"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Info boxes styling */
    .stAlert > div {
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs > div > div > div > div {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 10px 10px 0 0;
        border: 1px solid #dee2e6;
    }
    
    /* Custom section headers */
    .section-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Chart container styling */
    .stPlotlyChart {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Filter status styling */
    .filter-status {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Scotland flag colors accent */
    .scotland-accent {
        border-left: 5px solid #005EB8;
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    # Apply custom styling
    apply_custom_css()
    
    # Load data
    births, unemployment, holidays, predictions = load_data()
    
    if births is None:
        st.error("❌ Failed to load birth data. Please run data fetcher first!")
        return
    
    # Load model
    model = load_model()
    
    # Dashboard header with custom styling
    st.markdown("""
    <div class="dashboard-header">
        <h1>🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland Birth Rate Forecasting Dashboard</h1>
        <p style="font-size: 1.2em; margin: 0;">Real-time demographic insights and machine learning forecasts for Scotland</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Date range selector
    min_date = births['Date'].min().date()
    max_date = births['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Display options
    st.sidebar.subheader("📈 Display Options")
    show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
    show_trends = st.sidebar.checkbox("Show Trend Lines", value=True)
    
    # Scenario simulator
    st.sidebar.subheader("🎯 Scenario Simulator")
    
    # Unemployment filter
    unemployment_rate = st.sidebar.slider(
        "Unemployment Rate (%)", 
        min_value=2.0, 
        max_value=10.0, 
        value=4.5, 
        step=0.1,
        help="Filter data to show periods with similar unemployment rates (±0.5%)"
    )
    
    use_unemployment_filter = st.sidebar.checkbox(
        "Apply Unemployment Filter", 
        value=False,
        help="Show only data from periods with unemployment rates within ±0.5% of the selected value"
    )
    
    # Month filter
    available_months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    selected_months = st.sidebar.multiselect(
        "Filter by Months",
        options=available_months,
        format_func=lambda x: month_names[x-1],
        default=available_months,
        help="Select specific months to include in analysis"
    )
    
    # Year filter
    selected_years = None
    if len(births) > 0:
        min_year = births['Date'].dt.year.min()
        max_year = births['Date'].dt.year.max()
        selected_years = st.sidebar.slider(
            "Year Range",
            min_value=int(min_year),
            max_value=int(max_year),
            value=(int(min_year), int(max_year)),
            help="Select year range for analysis"
        )
    else:
        selected_years = (2000, 2024)  # Default range
    
    # Model status
    st.sidebar.subheader("🤖 Model Status")
    if model is not None:
        st.sidebar.success("✅ Trained model loaded")
        st.sidebar.info(f"Model type: XGBoost Regressor")
    else:
        st.sidebar.warning("⚠️ No trained model found")
        st.sidebar.info("Train a model to enable predictions")
    
    # Data Management Section
    st.sidebar.subheader("📥 Data Management")
    
    # Current data info
    if len(births) > 0:
        min_year = births['Date'].dt.year.min()
        max_year = births['Date'].dt.year.max()
        st.sidebar.info(f"Current data: {min_year}-{max_year} ({len(births)} records)")
    
    # Fetch Latest Data button
    if st.sidebar.button("🔄 Fetch Latest Data", help="Update dataset with 2024-2025 data"):
        with st.sidebar.spinner("Fetching latest data..."):
            success, record_count = fetch_latest_data()
            
            if success:
                st.sidebar.success(f"✅ Updated! Added {record_count} records")
                st.sidebar.info("Data now includes 2024-2025 estimates")
                st.rerun()  # Refresh the app to show new data
            else:
                st.sidebar.error("❌ Failed to fetch latest data")
    
    # Quick train model button 
    if st.sidebar.button("🚀 Train Model with New Data", help="Train model using the updated dataset"):
        with st.sidebar.spinner("Training model..."):
            import subprocess
            try:
                result = subprocess.run(['python', 'train_model.py'], 
                                       capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    st.sidebar.success("✅ Model trained successfully!")
                    st.rerun()  # Refresh to load new model
                else:
                    st.sidebar.error(f"❌ Training failed: {result.stderr}")
            except Exception as e:
                st.sidebar.error(f"❌ Training error: {e}")
    
    # Apply all scenario filters
    births_for_analysis, filter_info = apply_scenario_filters(
        births, unemployment, unemployment_rate, use_unemployment_filter, 
        selected_months, selected_years
    )
    
    # Display filter status
    if filter_info:
        st.sidebar.info(f"📊 Active filters: {' | '.join(filter_info)}")
        st.sidebar.info(f"📈 Showing {len(births_for_analysis)} of {len(births)} records")
        # Custom styled filter status in main area
        st.markdown(f"""
        <div class="filter-status">
            <strong>🎯 Scenario Analysis Active</strong><br>
            {' | '.join(filter_info)}<br>
            <em>Showing {len(births_for_analysis)} of {len(births)} records</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.info(f"No filters applied. Showing all {len(births)} records.")


    # Main dashboard content
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Main Dashboard", "🔍 Detailed Analysis", "⚙️ Model & Data"])
    
    with tab1:
        st.subheader("Scotland Birth Rate Trends and Forecasts")
        if len(date_range) == 2:
            fig = create_time_series_chart(births, predictions, date_range, show_predictions)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select a valid date range.")
        
        # Display key statistics
        stats = calculate_birth_statistics(births_for_analysis)
        st.markdown("<div class='section-header'>Key Performance Indicators (KPIs)</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        if stats:
            col1.metric("Total Births (in filtered data)", f"{stats.get('total_births', 0):,}")
            col2.metric("Average Monthly Births", f"{stats.get('avg_monthly_births', 0):,.0f}")
            col3.metric("Max Monthly Births", f"{stats.get('max_monthly_births', 0):,}")
            col4.metric("Min Monthly Births", f"{stats.get('min_monthly_births', 0):,}")

    with tab2:
        st.subheader("Detailed Trend Analysis")
        
        # Row 1: Seasonal Heatmap and Unemployment Correlation
        col_seasonal, col_unemployment = st.columns(2)
        with col_seasonal:
            fig_heatmap = create_seasonal_heatmap(births_for_analysis)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        with col_unemployment:
            fig_corr = create_unemployment_correlation(births_for_analysis, unemployment)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)

        # Row 2: Monthly and Quarterly Averages
        col_monthly, col_quarterly = st.columns(2)
        with col_monthly:
            fig_month = create_births_by_month_chart(births_for_analysis)
            if fig_month:
                st.plotly_chart(fig_month, use_container_width=True)
        with col_quarterly:
            fig_quarter = create_births_by_quarter_chart(births_for_analysis)
            if fig_quarter:
                st.plotly_chart(fig_quarter, use_container_width=True)
                
        # Row 3: Long-term yearly trends
        fig_yearly = create_births_by_year_chart(births)
        if fig_yearly:
            st.plotly_chart(fig_yearly, use_container_width=True)

    with tab3:
        st.subheader("Model and Data Management")
        
        # Display model retraining status
        if ModelRetrainingPipeline:
            st.markdown("### Model Retraining Pipeline Status")
            retraining_status = get_retraining_status()
            st.json(retraining_status)
            if st.button("Manually Run Retraining Pipeline"):
                with st.spinner("Starting retraining pipeline..."):
                    result = run_retraining_pipeline(force=True)
                    if 'error' in result:
                        st.error(f"Failed to start pipeline: {result['error']}")
                    else:
                        st.success("Retraining pipeline started successfully!")
                        st.json(result)
        else:
            st.warning("⚠️ Model retraining module is not available.")
        
        st.markdown("### Raw Data View")
        st.markdown("This view provides a direct look at the underlying datasets.")
        
        st.markdown("#### Raw Births Data")
        st.dataframe(births, use_container_width=True)

        st.markdown("#### Raw Unemployment Data")
        st.dataframe(unemployment, use_container_width=True)
        
        st.markdown("#### Raw Holiday Data")
        st.dataframe(holidays, use_container_width=True)


if __name__ == "__main__":
    main()