import streamlit as st
import os
import subprocess
import sys
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="Scotland Birth Rate Forecast Dashboard",
    page_icon="👶",
    layout="wide"
)

def run_data_fetcher():
    """Run the data fetcher script"""
    try:
        import data_fetcher
        data_fetcher.main()
        return True
    except Exception as e:
        st.error(f"Error running data fetcher: {e}")
        return False

def run_model_training():
    """Run the model training script"""
    try:
        import train_model
        train_model.main()
        return True
    except Exception as e:
        st.error(f"Error training model: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = ['births.csv', 'unemployment.csv', 'holidays.csv']
    existing_files = []
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    return existing_files, missing_files

def main():
    st.title("🏴󐁧󐁢󐁳󐁣󐁴󐁿 Scotland Birth Rate Forecasting System")
    st.markdown("Demographic insights and forecasting for Scotland using machine learning")
    
    # Sidebar for system controls
    st.sidebar.header("🔧 System Controls")
    
    # Check data status
    existing_files, missing_files = check_data_files()
    
    st.sidebar.subheader("📊 Data Status")
    if existing_files:
        st.sidebar.success(f"Available: {', '.join(existing_files)}")
    if missing_files:
        st.sidebar.warning(f"Missing: {', '.join(missing_files)}")
    
    # Data fetching section
    st.sidebar.subheader("📥 Data Management")
    
    if st.sidebar.button("🔄 Fetch Latest Data", help="Download fresh data from government APIs"):
        with st.spinner("Fetching data from APIs..."):
            success = run_data_fetcher()
            if success:
                st.sidebar.success("✅ Data fetched successfully!")
                st.rerun()
            else:
                st.sidebar.error("❌ Data fetching failed")
    
    # Model training section
    model_exists = os.path.exists('birth_model.pkl')
    st.sidebar.subheader("🤖 Model Training")
    
    if model_exists:
        st.sidebar.success("✅ Trained model available")
    else:
        st.sidebar.warning("⚠️ No trained model found")
    
    if st.sidebar.button("🚀 Train Model", help="Train XGBoost model on current data"):
        if not existing_files:
            st.sidebar.error("❌ No data files available. Fetch data first.")
        else:
            with st.spinner("Training XGBoost model..."):
                success = run_model_training()
                if success:
                    st.sidebar.success("✅ Model trained successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Model training failed")
    
    # Main content area
    if not existing_files:
        st.warning("⚠️ No data files found. Please fetch data first using the sidebar controls.")
        st.info("""
        **Getting Started:**
        1. Click '🔄 Fetch Latest Data' in the sidebar to download data from government APIs
        2. Once data is available, click '🚀 Train Model' to create the forecasting model
        3. The dashboard will automatically load with interactive visualizations
        """)
        
        # Show system information
        st.subheader("📋 System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(f"**Python Version:** {sys.version.split()[0]}")
        
        with col2:
            st.info(f"**Working Directory:** {os.getcwd()}")
            st.info(f"**Available Files:** {len(os.listdir('.'))} files")
        
    else:
        # Load the main dashboard
        try:
            from dashboard import main as dashboard_main
            dashboard_main()
        except Exception as e:
            st.error(f"❌ Error loading dashboard: {e}")
            st.info("Please ensure all required files are present and try refreshing the page.")

if __name__ == "__main__":
    main()
