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

# --- SHAP import for feature importance ---
try:
    import shap
except ImportError:
    shap = None

try:
    from model_retraining import ModelRetrainingPipeline, get_retraining_status, run_retraining_pipeline
except ImportError:
    ModelRetrainingPipeline = None
    get_retraining_status = lambda: {"error": "Retraining module not available"}
    run_retraining_pipeline = lambda force=False: {"error": "Retraining module not available"}

try:
    from data_fetcher import fetch_latest_data
except ImportError:
    fetch_latest_data = lambda: (False, 0)

@st.cache_data
def load_data():
    """Load all datasets with comprehensive error handling"""
    try:
        births = pd.read_csv('births.csv')
        births['Date'] = pd.to_datetime(births['Date'])
        try:
            unemployment = pd.read_csv('unemployment.csv')
            unemployment['Year'] = pd.to_numeric(unemployment['Year'], errors='coerce').astype('Int64')
            unemployment = unemployment.dropna(subset=['Year', 'Rate'])
        except FileNotFoundError:
            st.warning("⚠️ Unemployment data not found")
            unemployment = pd.DataFrame(columns=['Year', 'Rate'])
        try:
            holidays = pd.read_csv('holidays.csv')
            holidays['date'] = pd.to_datetime(holidays['date'], errors='coerce')
            holidays = holidays.dropna(subset=['date'])
        except FileNotFoundError:
            st.warning("⚠️ Holiday data not found")
            holidays = pd.DataFrame(columns=['date', 'title'])
        try:
            predictions = pd.read_csv('predictions.csv')
            predictions['Date'] = pd.to_datetime(predictions['Date'])
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
    try:
        if len(unemployment) == 0:
            st.info("📊 No unemployment data available for correlation analysis")
            return None
        births_yearly = births.copy()
        births_yearly['Year'] = births_yearly['Date'].dt.year
        births_yearly = births_yearly.groupby('Year')['Births'].sum().reset_index()
        births_yearly.columns = ['Year', 'Total_Births']
        births_yearly['Year'] = births_yearly['Year'].astype('Int64')
        unemployment['Year'] = unemployment['Year'].astype('Int64')
        merged = pd.merge(births_yearly, unemployment, on='Year', how='inner')
        if len(merged) == 0:
            st.info("📊 No overlapping years found between births and unemployment data")
            return None
        fig = px.scatter(
            merged,
            x='Rate',
            y='Total_Births',
            trendline='ols',
            hover_data=['Year'],
            labels={'Rate': 'Unemployment Rate (%)', 'Total_Births': 'Total Births per Year'}
        )
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
            height=350,
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
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Monthly Birth Counts Over Time', 'Yearly Birth Trends'],
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
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
        fig.update_xaxes(title_text="Date", gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
        fig.update_yaxes(title_text="Number of Births", gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
        fig.update_xaxes(title_text="Year", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
        fig.update_yaxes(title_text="Annual Total", gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
        return fig
    except Exception as e:
        st.error(f"Error creating comprehensive time series: {e}")
        return None

def create_time_series_chart(births, predictions, date_range, show_predictions):
    try:
        if len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            mask = (births['Date'] >= start_date) & (births['Date'] <= end_date)
            filtered_births = births[mask]
        else:
            filtered_births = births
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_births['Date'],
            y=filtered_births['Births'],
            mode='lines+markers',
            name='Historical Births',
            line=dict(color='#005EB8', width=3),
            marker=dict(size=6, color='#005EB8', line=dict(width=2, color='white')),
            hovertemplate='<b>Date:</b> %{x}<br><b>Births:</b> %{y:,}<extra></extra>'
        ))
        if show_predictions and len(predictions) > 0:
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Births'],
                mode='lines+markers',
                name='Forecasted Births',
                line=dict(color='#FF6B35', dash='dash', width=3),
                marker=dict(size=6, symbol='diamond', color='#FF6B35', line=dict(width=2, color='white')),
                hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:,}<extra></extra>'
            ))
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
    try:
        births_copy = births.copy()
        births_copy['Year'] = births_copy['Date'].dt.year
        births_copy['Month'] = births_copy['Date'].dt.month
        seasonal_data = births_copy.pivot_table(
            values='Births',
            index='Year',
            columns='Month',
            aggfunc='mean'
        )
        fig = px.imshow(
            seasonal_data,
            labels=dict(x="Month", y="Year", color="Births"),
            aspect="auto",
            color_continuous_scale=[[0, '#f8f9fa'], [0.3, '#85c1e9'], [0.6, '#3498db'], [1, '#1a5490']]
        )
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
            height=350,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        return fig
    except Exception as e:
        st.error(f"❌ Error creating seasonal heatmap: {e}")
        return None

def create_holiday_chart(holidays):
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
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=6, line=dict(width=2, color='white'))
        )
        return fig
    except Exception as e:
        st.error(f"❌ Error creating births by year chart: {e}")
        return None

def create_births_by_quarter_chart(births):
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
    try:
        births_copy = births.copy()
        births_copy['Year'] = births_copy['Date'].dt.year
        stats = {
            'total_births': births['Births'].sum(),
            'avg_monthly_births': births['Births'].mean(),
            'max_monthly_births': births['Births'].max(),
            'min_monthly_births': births['Births'].min(),
            'std_monthly_births': births['Births'].std()
        }
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

def apply_scenario_filters(births, unemployment, unemployment_rate, use_unemployment_filter, selected_months, selected_years):
    try:
        filtered_births = births.copy()
        filter_info = []
        filtered_births['Year'] = filtered_births['Date'].dt.year
        filtered_births['Month'] = filtered_births['Date'].dt.month
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
        if selected_months and len(selected_months) < 12:
            filtered_births = filtered_births[filtered_births['Month'].isin(selected_months)]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            selected_month_names = [month_names[m-1] for m in selected_months]
            filter_info.append(f"Months: {', '.join(selected_month_names)}")
        if use_unemployment_filter and len(unemployment) > 0:
            unemployment_dict = dict(zip(unemployment['Year'], unemployment['Rate']))
            filtered_births['Unemployment_Rate'] = filtered_births['Year'].map(unemployment_dict)
            tolerance = 0.5
            min_rate = unemployment_rate - tolerance
            max_rate = unemployment_rate + tolerance
            filtered_births = filtered_births[
                (filtered_births['Unemployment_Rate'] >= min_rate) & 
                (filtered_births['Unemployment_Rate'] <= max_rate)
            ]
            filter_info.append(f"Unemployment: {min_rate:.1f}%-{max_rate:.1f}%")
        cols_to_drop = ['Year', 'Month', 'Unemployment_Rate']
        for col in cols_to_drop:
            if col in filtered_births.columns:
                filtered_births = filtered_births.drop(col, axis=1)
        return filtered_births, filter_info
    except Exception as e:
        st.error(f"Error applying filters: {e}")
        return births, []

def apply_custom_css():
    st.markdown("""
    <style>
    .main > div { padding-top: 2rem; }
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .css-1d391kg { background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%); }
    .stAlert > div {
        border-radius: 10px; border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stTabs > div > div > div > div {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 10px 10px 0 0; border: 1px solid #dee2e6;
    }
    .section-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;
        text-align: center; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 1rem; margin: 1rem 0;
    }
    .filter-status {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .scotland-accent {
        border-left: 5px solid #005EB8;
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        padding: 1rem; border-radius: 0 10px 10px 0; margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def plot_shap_feature_importance(model, feature_df):
    """Plot SHAP feature importance for the trained model."""
    if shap is None:
        st.info("SHAP is not installed. Run: pip install shap")
        return
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df)
        st.markdown('<div class="section-header">🧬 Feature Importance (SHAP)</div>', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, feature_df, plot_type="bar", show=False, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error creating SHAP feature importance plot: {e}")

def main():
    apply_custom_css()
    births, unemployment, holidays, predictions = load_data()
    if births is None:
        st.error("❌ Failed to load birth data. Please run data fetcher first!")
        return
    model = load_model()
    st.markdown("""
    <div class="dashboard-header">
        <h1>🏴 Scotland Birth Rate Forecasting Dashboard</h1>
        <p style="font-size: 1.2em; margin: 0;">Real-time demographic insights and machine learning forecasts for Scotland</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.header("📊 Dashboard Controls")
    min_date = births['Date'].min().date()
    max_date = births['Date'].max().date()
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    st.sidebar.subheader("📈 Display Options")
    show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
    show_trends = st.sidebar.checkbox("Show Trend Lines", value=True)
    st.sidebar.subheader("🎯 Scenario Simulator")
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
        selected_years = (2000, 2024)
    st.sidebar.subheader("🤖 Model Status")
    if model is not None:
        st.sidebar.success("✅ Trained model loaded")
        st.sidebar.info(f"Model type: XGBoost Regressor")
    else:
        st.sidebar.warning("⚠️ No trained model found")
        st.sidebar.info("Train a model to enable predictions")
    st.sidebar.subheader("📥 Data Management")
    if len(births) > 0:
        min_year = births['Date'].dt.year.min()
        max_year = births['Date'].dt.year.max()
        st.sidebar.info(f"Current data: {min_year}-{max_year} ({len(births)} records)")
    if st.sidebar.button("🔄 Fetch Latest Data", help="Update dataset with 2024-2025 data"):
        with st.spinner("Fetching latest data..."):
            success, record_count = fetch_latest_data()
            if success:
                st.sidebar.success(f"✅ Updated! Added {record_count} records")
                st.sidebar.info("Data now includes 2024-2025 estimates")
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to fetch latest data")
    if st.sidebar.button("🚀 Train Model with New Data", help="Train model using the updated dataset"):
        with st.spinner("Training model..."):
          with st.sidebar:
           st.write("Training model...")
        import subprocess
        try:
                result = subprocess.run(['python', 'train_model.py'],
                                       capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    st.sidebar.success("✅ Model trained successfully!")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Training failed: {result.stderr}")
        except Exception as e:
                st.sidebar.error(f"❌ Training error: {e}")
    births_for_analysis, filter_info = apply_scenario_filters(
        births, unemployment, unemployment_rate, use_unemployment_filter,
        selected_months, selected_years
    )
    if filter_info:
        st.sidebar.info(f"📊 Active filters: {' | '.join(filter_info)}")
        st.sidebar.info(f"📈 Showing {len(births_for_analysis)} of {len(births)} records")
        st.markdown(f"""
        <div class="filter-status">
            <strong>🎯 Scenario Analysis Active</strong><br>
            {' | '.join(filter_info)}<br>
            <em>Showing {len(births_for_analysis)} of {len(births)} records</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.info(f"📈 Showing all {len(births)} records")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if len(births_for_analysis) > 0:
            latest_births = births_for_analysis['Births'].iloc[-1]
            prev_births = births_for_analysis['Births'].iloc[-2] if len(births_for_analysis) > 1 else latest_births
            st.metric(
                label="Latest Monthly Births",
                value=f"{latest_births:,}",
                delta=f"{latest_births - prev_births:+.0f}"
            )
        else:
            st.metric(label="Latest Monthly Births", value="N/A", delta="No data in filter")
    with col2:
        if len(births_for_analysis) > 0:
            avg_births = births_for_analysis['Births'].mean()
            latest_births = births_for_analysis['Births'].iloc[-1] if len(births_for_analysis) > 0 else avg_births
            st.metric(
                label="Average Monthly Births",
                value=f"{avg_births:,.0f}",
                delta=f"{((latest_births/avg_births-1)*100):+.1f}%" if avg_births > 0 else "0%"
            )
        else:
            st.metric(label="Average Monthly Births", value="N/A", delta="No data in filter")
    with col3:
        if len(births_for_analysis) > 0:
            total_births = births_for_analysis['Births'].sum()
            st.metric(
                label="Total Births (Filtered)",
                value=f"{total_births:,}",
                delta=f"{len(births_for_analysis)} months"
            )
        else:
            st.metric(label="Total Births (Filtered)", value="0", delta="No data in filter")
    with col4:
        if len(unemployment) > 0:
            latest_unemployment = unemployment['Rate'].iloc[-1]
            st.metric(
                label="Target Unemployment",
                value=f"{unemployment_rate:.1f}%",
                delta=f"{unemployment_rate - latest_unemployment:+.1f}% vs latest"
            )
        else:
            st.metric(
                label="Target Unemployment",
                value=f"{unemployment_rate:.1f}%",
                delta="No historical data"
            )
    st.markdown('<div class="section-header">📈 Complete Time Series Analysis</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    comprehensive_ts_fig = create_comprehensive_time_series(births_for_analysis)
    if comprehensive_ts_fig:
        st.plotly_chart(comprehensive_ts_fig, use_container_width=True)
    else:
        st.warning("Unable to create comprehensive time series chart")
    st.markdown('<div class="section-header">🔍 Detailed Analysis & Forecasts</div>', unsafe_allow_html=True)
    time_series_fig = create_time_series_chart(births_for_analysis, predictions, date_range, show_predictions)
    if time_series_fig:
        st.plotly_chart(time_series_fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="scotland-accent"><strong>📊 Seasonal Patterns</strong></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        seasonal_fig = create_seasonal_heatmap(births_for_analysis)
        if seasonal_fig:
            st.plotly_chart(seasonal_fig, use_container_width=True)
        else:
            st.warning("Unable to create seasonal heatmap")
    with col2:
        st.markdown('<div class="scotland-accent"><strong>💼 Economic Impact</strong></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        unemployment_fig = create_unemployment_correlation(births_for_analysis, unemployment)
        if unemployment_fig:
            st.plotly_chart(unemployment_fig, use_container_width=True)
        else:
            st.warning("Unable to create unemployment correlation chart")
    st.subheader("🎉 Holiday Impact Analysis")
    holiday_fig = create_holiday_chart(holidays)
    if holiday_fig:
        st.plotly_chart(holiday_fig, use_container_width=True)
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    birth_stats = calculate_birth_statistics(births_for_analysis)
    if birth_stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Average Monthly Births",
                value=f"{birth_stats['avg_monthly_births']:,.0f}",
                delta=f"±{birth_stats['std_monthly_births']:,.0f}"
            )
        with col2:
            st.metric(
                label="Max Monthly Births",
                value=f"{birth_stats['max_monthly_births']:,}",
                delta="Peak month"
            )
        with col3:
            st.metric(
                label="Min Monthly Births",
                value=f"{birth_stats['min_monthly_births']:,}",
                delta="Lowest month"
            )
        with col4:
            st.metric(
                label="Average Yearly Births",
                value=f"{birth_stats['avg_yearly_births']:,.0f}",
                delta=f"Peak: {birth_stats['max_year']}"
            )
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        births_month_fig = create_births_by_month_chart(births_for_analysis)
        if births_month_fig:
            st.plotly_chart(births_month_fig, use_container_width=True)
        else:
            st.warning("Unable to create monthly births chart")
    with col2:
        births_quarter_fig = create_births_by_quarter_chart(births_for_analysis)
        if births_quarter_fig:
            st.plotly_chart(births_quarter_fig, use_container_width=True)
        else:
            st.warning("Unable to create quarterly births chart")
    st.markdown("<br>", unsafe_allow_html=True)
    births_year_fig = create_births_by_year_chart(births_for_analysis)
    if births_year_fig:
        st.plotly_chart(births_year_fig, use_container_width=True)
    else:
        st.warning("Unable to create yearly births chart")
    st.markdown('<div class="section-header">🔮 Predictions Analysis</div>', unsafe_allow_html=True)
    if len(predictions) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'Predicted_Births' in predictions.columns:
                avg_prediction = predictions['Predicted_Births'].mean()
                st.metric(
                    label="Average Predicted Births",
                    value=f"{avg_prediction:,.0f}",
                    delta=f"{avg_prediction - births['Births'].mean():+.0f} vs historical"
                )
            else:
                st.metric(label="Average Predicted Births", value="N/A", delta="No data")
        with col2:
            if 'Predicted_Births' in predictions.columns and len(predictions) > 0:
                max_prediction = predictions['Predicted_Births'].max()
                max_idx = predictions['Predicted_Births'].idxmax()
                max_date = predictions.loc[max_idx, 'Date']
                st.metric(
                    label="Peak Predicted Month",
                    value=f"{max_prediction:,}",
                    delta=f"{max_date.strftime('%Y-%m')}"
                )
            else:
                st.metric(label="Peak Predicted Month", value="N/A", delta="No data")
        with col3:
            if 'Predicted_Births' in predictions.columns and len(predictions) > 0:
                min_prediction = predictions['Predicted_Births'].min()
                min_idx = predictions['Predicted_Births'].idxmin()
                min_date = predictions.loc[min_idx, 'Date']
                st.metric(
                    label="Lowest Predicted Month",
                    value=f"{min_prediction:,}",
                    delta=f"{min_date.strftime('%Y-%m')}"
                )
            else:
                st.metric(label="Lowest Predicted Month", value="N/A", delta="No data")
    else:
        st.info("No predictions available. Train a model first to see prediction analysis.")
    st.markdown('<div class="section-header">📋 Raw Data Explorer</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Births", "💼 Unemployment", "🎉 Holidays", "🔮 Predictions"])
    with tab1:
        st.dataframe(births_for_analysis, use_container_width=True)
        if filter_info:
            st.caption(f"📈 Filtered records: {len(births_for_analysis)} of {len(births)} ({' | '.join(filter_info)})")
        else:
            st.caption(f"📈 Total records: {len(births_for_analysis)}")
    with tab2:
        if len(unemployment) > 0:
            st.dataframe(unemployment, use_container_width=True)
            st.caption(f"📊 Total records: {len(unemployment)}")
        else:
            st.info("No unemployment data available")
    with tab3:
        if len(holidays) > 0:
            st.dataframe(holidays, use_container_width=True)
            st.caption(f"🎉 Total records: {len(holidays)}")
        else:
            st.info("No holiday data available")
    with tab4:
        if len(predictions) > 0:
            st.dataframe(predictions, use_container_width=True)
            st.caption(f"🔮 Total predictions: {len(predictions)}")
        else:
            st.info("No predictions available. Train a model first.")
    st.markdown('<div class="section-header">💾 Data Export</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        csv = births_for_analysis.to_csv(index=False)
        filename_suffix = "_filtered" if filter_info else ""
        st.download_button(
            label="📊 Download Births Data",
            data=csv,
            file_name=f"scotland_births{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col2:
        if len(unemployment) > 0:
            unemp_csv = unemployment.to_csv(index=False)
            st.download_button(
                label="💼 Download Unemployment Data",
                data=unemp_csv,
                file_name=f"unemployment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    with col3:
        if len(holidays) > 0:
            holiday_csv = holidays.to_csv(index=False)
            st.download_button(
                label="🎉 Download Holiday Data",
                data=holiday_csv,
                file_name=f"holidays_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    with col4:
        if len(predictions) > 0:
            pred_csv = predictions.to_csv(index=False)
            st.download_button(
                label="🔮 Download Predictions",
                data=pred_csv,
                file_name=f"birth_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    st.markdown('<div class="section-header">🔄 Automated Model Retraining</div>', unsafe_allow_html=True)
    try:
        status = get_retraining_status()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if status.get('current_model_age_days') is not None:
                age_days = status['current_model_age_days']
                st.metric(
                    label="Model Age",
                    value=f"{age_days} days",
                    delta=f"{'Fresh' if age_days <= 7 else 'Aging' if age_days <= 30 else 'Outdated'}"
                )
            else:
                st.metric(label="Model Age", value="Unknown", delta="No training history")
        with col2:
            total_retrainings = status.get('total_retrainings', 0)
            st.metric(
                label="Total Retrainings",
                value=str(total_retrainings),
                delta=f"Automatic pipeline"
            )
        with col3:
            if status.get('next_scheduled_check'):
                next_check = datetime.fromisoformat(status['next_scheduled_check'])
                days_until = (next_check - datetime.now()).days
                st.metric(
                    label="Next Check",
                    value=f"{max(0, days_until)} days",
                    delta=next_check.strftime("%Y-%m-%d")
                )
            else:
                st.metric(label="Next Check", value="Not scheduled", delta="Configure pipeline")
        with col4:
            col_run, col_status = st.columns(2)
            with col_run:
                if st.button("🔄 Run Now", type="primary", use_container_width=True):
                    with st.spinner("Running retraining pipeline..."):
                        try:
                            result = run_retraining_pipeline(force=True)
                            if result.get('retraining_performed'):
                                st.success("Model retrained successfully!")
                                st.rerun()
                            else:
                                st.info("No retraining needed")
                        except Exception as e:
                            st.error(f"Retraining failed: {str(e)}")
            with col_status:
                if st.button("📊 View Status", use_container_width=True):
                    st.session_state.show_retraining_details = not st.session_state.get('show_retraining_details', False)
        if st.session_state.get('show_retraining_details', False):
            st.markdown("#### 📋 Detailed Retraining Status")
            try:
                pipeline = ModelRetrainingPipeline()
                config = pipeline.config
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔧 Configuration**")
                    st.json({
                        "Retraining Frequency": f"{config.get('retraining_frequency_days', 30)} days",
                        "Performance Threshold": f"{config.get('performance_threshold', 0.15):.1%}",
                        "Min Improvement": f"{config.get('min_improvement_threshold', 0.05):.1%}",
                        "Auto Deploy": config.get('auto_deploy', True)
                    })
                with col2:
                    st.markdown("**📈 Last Retraining**")
                    if status.get('last_retraining'):
                        last_training = status['last_retraining']
                        st.json({
                            "Timestamp": last_training.get('timestamp', 'Unknown'),
                            "Duration": f"{last_training.get('duration_seconds', 0):.2f}s",
                            "Status": last_training.get('deployment_status', 'Unknown'),
                            "Success": last_training.get('success', False)
                        })
                    else:
                        st.info("No retraining history available")
                if status.get('total_retrainings', 0) > 0:
                    st.markdown("**📚 Retraining History**")
                    history = pipeline._load_history()
                    history_data = []
                    for entry in history[-10:]:
                        history_data.append({
                            'Date': entry.get('timestamp', 'Unknown')[:10],
                            'Success': '✅' if entry.get('success') else '❌',
                            'Duration': f"{entry.get('duration_seconds', 0):.1f}s",
                            'Status': entry.get('deployment_status', 'Unknown'),
                            'Trigger': ', '.join(entry.get('trigger_reasons', [])[:2])
                        })
                    if history_data:
                        history_df = pd.DataFrame(history_data)
                        st.dataframe(history_df, use_container_width=True)
                st.markdown("**🔍 Current Status Check**")
                check_results = pipeline.should_retrain()
                if check_results['should_retrain']:
                    st.warning(f"⚠️ Retraining recommended: {', '.join(check_results['reasons'])}")
                else:
                    st.success("✅ Model is up to date")
                freshness = check_results.get('data_freshness', {})
                if freshness.get('has_fresh_data'):
                    st.info(f"📊 Data: {freshness.get('record_count', 0)} records, latest: {freshness.get('latest_date', 'Unknown')}")
                else:
                    st.warning("📊 Data may be outdated")
            except Exception as e:
                st.error(f"Error loading pipeline details: {str(e)}")
    except Exception as e:
        st.error(f"Error loading retraining status: {str(e)}")
        st.info("The automated retraining pipeline will be available once the system is properly configured.")

    # --- SHAP Feature Importance Section ---
    if model is not None:
        # Display the saved SHAP summary plot image instead of generating it live
        import os
        shap_img_path = "shap_summary_plot.png"
        if os.path.exists(shap_img_path):
            st.markdown('<div class="section-header">🧬 Feature Importance (SHAP)</div>', unsafe_allow_html=True)
            st.image(shap_img_path, caption="SHAP Feature Importance", use_column_width=True)
        else:
            st.info("SHAP summary plot image not found. Please retrain the model to generate it.")

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2c3e50, #34495e);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h3 style="margin-bottom: 1rem; color: #ecf0f1;">🏴 Scotland Birth Rate Forecasting System</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-bottom: 1rem;">
            <div style="margin: 0.5rem;">
                <strong>📊 Data Sources</strong><br>
                <small>NRS Scotland • World Bank API • UK Government</small>
            </div>
            <div style="margin: 0.5rem;">
                <strong>🤖 Machine Learning</strong><br>
                <small>XGBoost Regressor with Time-Series Validation</small>
            </div>
            <div style="margin: 0.5rem;">
                <strong>🎯 Features</strong><br>
                <small>Real-time Analysis • Scenario Simulation • Interactive Forecasting</small>
            </div>
        </div>
        <hr style="border: 1px solid #7f8c8d; margin: 1rem 0;">
        <small style="color: #bdc3c7;">
            Built with Streamlit • Powered by Authentic Government Data • 
            Designed for Demographic Research & Policy Planning
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
