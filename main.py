import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io

from data_processor import load_and_process_retail_data, process_uploaded_file
from attribution_models import (
    first_touch_attribution, 
    last_touch_attribution, 
    linear_attribution,
    time_decay_attribution,
    position_based_attribution,
    markov_chain_attribution,
    shapley_value_attribution,
    calculate_attribution_comparison
)

# Page configuration
st.set_page_config(
    page_title="Customer Journey Attribution Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Main title and description
    st.title("🚀 Customer Journey Attribution Analysis")
    st.markdown("""
    This platform analyzes real-world e-commerce data to understand customer journeys and 
    attribute conversions across multiple marketing touchpoints using various attribution models.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload option
    st.sidebar.subheader("📁 Data Source")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose your data source:",
        ["Upload your own dataset", "Use sample dataset for testing"],
        help="Upload real transaction data or test with synthetic data"
    )
    
    uploaded_file = None
    if data_source == "Upload your own dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your e-commerce dataset",
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with transaction data (CustomerID, InvoiceDate, etc.)"
        )
    else:
        st.sidebar.info("Using realistic sample dataset with 300 customers and 1 year of transaction history")
    
    # Attribution model selection
    st.sidebar.subheader("🎯 Attribution Model")
    attribution_model = st.sidebar.selectbox(
        "Select Attribution Model",
        ["First-Touch", "Last-Touch", "Linear", "Time-Decay", "Position-Based", "Markov Chain", "Shapley Value", "Compare All Models"],
        help="Choose how to attribute conversions to marketing touchpoints"
    )
    
    # Initialize model-specific parameters with defaults
    decay_rate = 0.5
    first_touch_weight = 0.4
    last_touch_weight = 0.4
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        lookback_days = st.slider(
            "Journey Lookback Period (days)",
            min_value=1,
            max_value=30,
            value=14,
            help="How many days before conversion to look for touchpoints"
        )
        
        min_touchpoints = st.slider(
            "Minimum Touchpoints per Journey",
            min_value=1,
            max_value=5,
            value=2,
            help="Minimum number of touchpoints to include in analysis"
        )
        
        # Model-specific parameters - only show when relevant
        if attribution_model == "Time-Decay":
            decay_rate = st.slider(
                "Time Decay Rate",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                help="Higher values give more weight to recent touchpoints"
            )
            
        if attribution_model == "Position-Based":
            first_touch_weight = st.slider(
                "First Touch Weight",
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.1,
                help="Weight given to the first touchpoint"
            )
            last_touch_weight = st.slider(
                "Last Touch Weight", 
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.1,
                help="Weight given to the last touchpoint"
            )
        if attribution_model == "Compare All Models":
            decay_rate = st.slider(
                "Time Decay Rate",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                help="Higher values give more weight to recent touchpoints"
            )
            first_touch_weight = st.slider(
                "First Touch Weight",
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.1,
                help="Weight given to the first touchpoint"
            )
            last_touch_weight = st.slider(
                "Last Touch Weight", 
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.1,
                help="Weight given to the last touchpoint"
            )
            
            
            
    
    # Process button
    process_data = st.sidebar.button("🔄 Process Data & Analyze", type="primary")
    
    # Main content area
    if process_data or 'journey_data' in st.session_state:
        
        # Load and process data
        if process_data:
            with st.spinner("Loading and processing data..."):
                try:
                    if data_source == "Upload your own dataset" and uploaded_file is not None:
                        journey_data = process_uploaded_file(uploaded_file, lookback_days, min_touchpoints)
                    elif data_source == "Use sample dataset for testing":
                        # Generate sample data for testing
                        journey_data = load_and_process_retail_data(
                            lookback_days=lookback_days,
                            min_touchpoints=min_touchpoints
                        )
                    else:
                        st.error("Please upload a dataset to proceed with the analysis.")
                        return
                    
                    if journey_data.empty:
                        st.error("❌ No data could be processed. Please check your file format or upload a valid dataset.")
                        return
                    
                    st.session_state['journey_data'] = journey_data
                    st.success(f"✅ Successfully processed {len(journey_data)} touchpoints for analysis!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
                    return
        
        # Use cached data if available
        journey_data = st.session_state.get('journey_data', pd.DataFrame())
        
        if journey_data.empty:
            st.warning("⚠️ No data available. Please upload a dataset or ensure the default dataset is accessible.")
            return
        
        # Display analysis results
        display_analysis_results(journey_data, attribution_model, decay_rate, first_touch_weight, last_touch_weight)
    
    else:
        # Initial state - show instructions
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen with instructions"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 🎯 Welcome to Customer Journey Attribution Analysis
        
        This platform helps you understand how different marketing channels contribute to conversions 
        by analyzing real e-commerce transaction data.
        
        #### 📋 How to Get Started:
        1. **Choose your data source** - Upload your own transaction data or test with sample data
        2. **Select an attribution model** to analyze how touchpoints contribute to conversions
        3. **Configure settings** for journey analysis (lookback period, minimum touchpoints)
        4. **Click "Process Data & Analyze"** to see insights
        
        #### 📊 What You'll Get:
        - **Customer journey insights** with inferred marketing touchpoints
        - **Attribution analysis** showing channel performance
        - **Interactive visualizations** of conversion paths
        - **Actionable recommendations** for marketing optimization
        
        #### 🧠 Attribution Models Available:
        - **First-Touch & Last-Touch**: Simple rule-based attribution
        - **Linear**: Equal credit distribution across touchpoints
        - **Time-Decay**: More weight to recent touchpoints
        - **Position-Based**: U-shaped attribution emphasizing first and last touches
        - **Markov Chain**: Advanced probabilistic model using transition matrices
        - **Shapley Value**: Game theory approach with logistic regression
        
        #### 📁 Data Options:
        - **Sample Dataset**: Realistic synthetic e-commerce data with 300 customers for testing
        - **Your Own Data**: Upload CSV or Excel files with CustomerID and InvoiceDate columns
        
        #### 🚀 Ready to Start?
        Choose your data source in the sidebar and start analyzing customer journeys!
        """)

def display_analysis_results(journey_data, attribution_model, decay_rate, first_touch_weight, last_touch_weight):
    """Display the complete analysis results"""
    
    # Data overview section
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_touchpoints = len(journey_data)
        st.metric("Total Touchpoints", f"{total_touchpoints:,}")
    
    with col2:
        unique_customers = journey_data['customer_id'].nunique()
        st.metric("Unique Customers", f"{unique_customers:,}")
    
    with col3:
        total_conversions = len(journey_data[journey_data['conversion'] == True])
        st.metric("Total Conversions", f"{total_conversions:,}")
    
    with col4:
        avg_journey_length = journey_data.groupby('customer_id').size().mean()
        st.metric("Avg Journey Length", f"{avg_journey_length:.1f}")
    
    # Sample data preview
    with st.expander("🔍 View Sample Journey Data"):
        st.dataframe(
            journey_data.head(20),
            use_container_width=True
        )
    
    # Channel analysis
    st.header("📊 Channel Performance Analysis")
    
    # Channel frequency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Touchpoint Frequency by Channel")
        channel_counts = journey_data['channel'].value_counts()
        
        fig_freq = px.bar(
            x=channel_counts.index,
            y=channel_counts.values,
            title="Number of Touchpoints by Channel",
            labels={'x': 'Marketing Channel', 'y': 'Number of Touchpoints'}
        )
        fig_freq.update_layout(showlegend=False)
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col2:
        st.subheader("Conversion Rate by Channel")
        conversion_rates = journey_data.groupby('channel')['conversion'].agg(['count', 'sum']).reset_index()
        conversion_rates['conversion_rate'] = (conversion_rates['sum'] / conversion_rates['count'] * 100).round(2)
        
        fig_conv = px.bar(
            conversion_rates,
            x='channel',
            y='conversion_rate',
            title="Conversion Rate by Channel (%)",
            labels={'channel': 'Marketing Channel', 'conversion_rate': 'Conversion Rate (%)'}
        )
        st.plotly_chart(fig_conv, use_container_width=True)
    
    # Attribution analysis
    st.header("🎯 Attribution Analysis")
    
    if attribution_model == "Compare All Models":
        display_attribution_comparison(journey_data, decay_rate, first_touch_weight, last_touch_weight)
    else:
        display_single_attribution_model(journey_data, attribution_model, decay_rate, first_touch_weight, last_touch_weight)
    
    # Journey path analysis
    display_journey_path_analysis(journey_data)
    
    # Insights and recommendations
    display_insights_and_recommendations(journey_data, attribution_model)

def display_single_attribution_model(journey_data, model_name, decay_rate=0.5, first_touch_weight=0.4, last_touch_weight=0.4):
    """Display results for a single attribution model"""
    
    # Calculate attribution
    if model_name == "First-Touch":
        attribution_results = first_touch_attribution(journey_data)
    elif model_name == "Last-Touch":
        attribution_results = last_touch_attribution(journey_data)
    elif model_name == "Linear":
        attribution_results = linear_attribution(journey_data)
    elif model_name == "Time-Decay":
        attribution_results = time_decay_attribution(journey_data, decay_rate=decay_rate)
    elif model_name == "Position-Based":
        attribution_results = position_based_attribution(journey_data, first_touch_weight=first_touch_weight, last_touch_weight=last_touch_weight)
    elif model_name == "Markov Chain":
        attribution_results = markov_chain_attribution(journey_data)
    elif model_name == "Shapley Value":
        attribution_results = shapley_value_attribution(journey_data)
    else:
        attribution_results = first_touch_attribution(journey_data)  # fallback
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{model_name} Attribution Results")
        
        # Create attribution chart
        fig = px.bar(
            attribution_results,
            x='channel',
            y='attributed_conversions',
            title=f"{model_name} Attribution: Conversions by Channel",
            labels={'channel': 'Marketing Channel', 'attributed_conversions': 'Attributed Conversions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Attribution Details")
        
        # Display attribution table
        attribution_display = attribution_results.copy()
        attribution_display['attributed_conversions'] = attribution_display['attributed_conversions'].round(2)
        attribution_display['attribution_percentage'] = (
            attribution_display['attributed_conversions'] / 
            attribution_display['attributed_conversions'].sum() * 100
        ).round(1)
        
        st.dataframe(
            attribution_display[['channel', 'attributed_conversions', 'attribution_percentage']].rename(columns={
                'channel': 'Channel',
                'attributed_conversions': 'Attributed Conversions',
                'attribution_percentage': 'Attribution %'
            }),
            use_container_width=True,
            hide_index=True
        )

def display_attribution_comparison(journey_data, decay_rate=0.5, first_touch_weight=0.4, last_touch_weight=0.4):
    """Display comparison of all attribution models"""
    
    st.subheader("Attribution Model Comparison")
    
    comparison_results = calculate_attribution_comparison(journey_data, decay_rate=decay_rate, first_touch_weight=first_touch_weight, last_touch_weight=last_touch_weight)
    
    # Create comparison visualization
    fig = go.Figure()
    
    models = comparison_results['model'].unique()
    channels = comparison_results['channel'].unique()
    
    for model in models:
        model_data = comparison_results[comparison_results['model'] == model]
        fig.add_trace(go.Bar(
            name=model,
            x=model_data['channel'],
            y=model_data['attributed_conversions']
        ))
    
    fig.update_layout(
        title="Attribution Model Comparison: Conversions by Channel",
        xaxis_title="Marketing Channel",
        yaxis_title="Attributed Conversions",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display comparison table
    pivot_table = comparison_results.pivot(index='channel', columns='model', values='attributed_conversions').round(2)
    st.dataframe(pivot_table, use_container_width=True)

def display_journey_path_analysis(journey_data):
    """Display customer journey path analysis"""
    
    st.header("🛤️ Customer Journey Path Analysis")
    
    # Calculate most common journey paths
    journey_paths = journey_data.groupby('customer_id')['channel'].apply(
        lambda x: ' → '.join(x.tolist())
    ).value_counts().head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Journey Paths")
        
        path_df = pd.DataFrame({
            'Journey Path': journey_paths.index,
            'Frequency': journey_paths.values
        })
        
        fig = px.bar(
            path_df,
            y='Journey Path',
            x='Frequency',
            orientation='h',
            title="Most Common Customer Journey Paths"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Journey Length Distribution")
        
        journey_lengths = journey_data.groupby('customer_id').size()
        
        fig = px.histogram(
            x=journey_lengths.values,
            nbins=10,
            title="Distribution of Journey Lengths",
            labels={'x': 'Number of Touchpoints', 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_insights_and_recommendations(journey_data, attribution_model):
    """Display actionable insights and recommendations"""
    
    st.header("💡 Insights & Recommendations")
    
    # Calculate key metrics for insights
    channel_performance = journey_data.groupby('channel').agg({
        'conversion': ['count', 'sum'],
        'cost': 'sum'
    }).round(2)
    
    channel_performance.columns = ['touchpoints', 'conversions', 'total_cost']
    channel_performance['conversion_rate'] = (
        channel_performance['conversions'] / channel_performance['touchpoints'] * 100
    ).round(2)
    
    channel_performance['cost_per_conversion'] = (
        channel_performance['total_cost'] / channel_performance['conversions']
    ).round(2)
    
    # Generate insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Insights")
        
        # Top performing channel
        best_conversion_channel = channel_performance['conversion_rate'].idxmax()
        best_conversion_rate = channel_performance.loc[best_conversion_channel, 'conversion_rate']
        
        st.success(f"**Highest Converting Channel:** {best_conversion_channel} ({best_conversion_rate}% conversion rate)")
        
        # Most cost-effective channel
        if not channel_performance['cost_per_conversion'].isna().all():
            most_efficient_channel = channel_performance['cost_per_conversion'].idxmin()
            efficiency_cost = channel_performance.loc[most_efficient_channel, 'cost_per_conversion']
            st.info(f"**Most Cost-Effective:** {most_efficient_channel} (${efficiency_cost:.2f} per conversion)")
        
        # Journey insights
        avg_journey_length = journey_data.groupby('customer_id').size().mean()
        st.metric("Average Customer Journey Length", f"{avg_journey_length:.1f} touchpoints")
        
        # Conversion timing
        conversion_customers = journey_data[journey_data['conversion'] == True]['customer_id'].unique()
        if len(conversion_customers) > 0:
            st.success(f"**Conversion Rate:** {len(conversion_customers) / journey_data['customer_id'].nunique() * 100:.1f}% of customers converted")
    
    with col2:
        st.subheader("📋 Recommendations")
        
        recommendations = []
        
        # Performance-based recommendations
        if best_conversion_rate > 20:
            recommendations.append(f"🚀 **Scale up {best_conversion_channel}** - it shows strong conversion performance ({best_conversion_rate}%)")
        
        # Cost efficiency recommendations
        if not channel_performance['cost_per_conversion'].isna().all():
            high_cost_channels = channel_performance[
                channel_performance['cost_per_conversion'] > channel_performance['cost_per_conversion'].median()
            ].index.tolist()
            
            if high_cost_channels:
                recommendations.append(f"💰 **Optimize budget allocation** - Review spend on {', '.join(high_cost_channels)} for cost efficiency")
        
        # Journey optimization
        if avg_journey_length > 4:
            recommendations.append("⚡ **Simplify customer journeys** - Consider reducing touchpoints to accelerate conversions")
        elif avg_journey_length < 2:
            recommendations.append("🎯 **Enhance awareness campaigns** - Consider adding more touchpoints to build brand familiarity")
        
        # Attribution-specific recommendations
        if attribution_model == "First-Touch":
            recommendations.append("🎬 **Focus on awareness channels** - First-touch attribution highlights channels that initiate customer journeys")
        elif attribution_model == "Last-Touch":
            recommendations.append("🎯 **Optimize conversion channels** - Last-touch attribution shows channels that close deals")
        elif attribution_model == "Linear":
            recommendations.append("⚖️ **Balanced approach** - Linear attribution provides equal credit across the customer journey")
        
        for i, rec in enumerate(recommendations[:5], 1):
            st.markdown(f"{i}. {rec}")
        
        if len(recommendations) == 0:
            st.info("📊 Upload more data or adjust settings to generate specific recommendations")

if __name__ == "__main__":
    main()
