"""
NovaMart Marketing Analytics Dashboard - Starter Template
=========================================================
Masters of AI in Business - Data Visualization Assignment

This starter template provides the basic structure for your Streamlit dashboard.
Complete the TODO sections to build your visualizations.

To run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="NovaMart Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING (with caching)
# =============================================================================
@st.cache_data
def load_data():
    """Load all datasets"""
    data = {}
    
    # Update these paths to match your data location
    data_path = "data/"  # Change this to your data folder path
    
    try:
        data['campaigns'] = pd.read_csv(f"{data_path}campaign_performance.csv", parse_dates=['date'])
        data['customers'] = pd.read_csv(f"{data_path}customer_data.csv")
        data['products'] = pd.read_csv(f"{data_path}product_sales.csv")
        data['leads'] = pd.read_csv(f"{data_path}lead_scoring_results.csv")
        data['feature_importance'] = pd.read_csv(f"{data_path}feature_importance.csv")
        data['learning_curve'] = pd.read_csv(f"{data_path}learning_curve.csv")
        data['geographic'] = pd.read_csv(f"{data_path}geographic_data.csv")
        data['attribution'] = pd.read_csv(f"{data_path}channel_attribution.csv")
        data['funnel'] = pd.read_csv(f"{data_path}funnel_data.csv")
        data['journey'] = pd.read_csv(f"{data_path}customer_journey.csv")
        data['correlation'] = pd.read_csv(f"{data_path}correlation_matrix.csv", index_col=0)
        
        return data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please ensure all CSV files are in the 'data/' folder")
        return None

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
def sidebar():
    """Create sidebar navigation"""
    st.sidebar.title("📊 NovaMart Analytics")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "🏠 Executive Overview",
            "📈 Campaign Analytics",
            "👥 Customer Insights",
            "📦 Product Performance",
            "🗺️ Geographic Analysis",
            "🎯 Attribution & Funnel",
            "🤖 ML Model Evaluation"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Masters of AI in Business\nData Visualization Assignment")
    
    return page

# =============================================================================
# PAGE: EXECUTIVE OVERVIEW
# =============================================================================
def page_executive_overview(data):
    """Executive Overview Dashboard"""
    st.title("🏠 Executive Overview")
    st.markdown("Key performance metrics at a glance")
    
    campaigns = data['campaigns']
    customers = data['customers']
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = campaigns['revenue'].sum()
        st.metric("Total Revenue", f"₹{total_revenue/1e7:.2f} Cr")
    
    with col2:
        total_conversions = campaigns['conversions'].sum()
        st.metric("Total Conversions", f"{total_conversions:,}")
    
    with col3:
        avg_roas = campaigns[campaigns['roas'] > 0]['roas'].mean()
        st.metric("Avg ROAS", f"{avg_roas:.2f}x")
    
    with col4:
        total_customers = len(customers)
        st.metric("Total Customers", f"{total_customers:,}")
    
    st.markdown("---")
    
    # TODO: Add Line Chart - Revenue Trend
    st.subheader("📈 Revenue Trend Over Time")
    
    # Aggregate by month
    monthly_revenue = campaigns.groupby(pd.Grouper(key='date', freq='M'))['revenue'].sum().reset_index()
    
    fig = px.line(
        monthly_revenue, 
        x='date', 
        y='revenue',
        title='Monthly Revenue Trend',
        labels={'date': 'Month', 'revenue': 'Revenue (₹)'}
    )
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # TODO: Add Bar Chart - Channel Performance
    st.subheader("📊 Revenue by Channel")
    
    channel_revenue = campaigns.groupby('channel')['revenue'].sum().sort_values(ascending=True).reset_index()
    
    fig = px.bar(
        channel_revenue,
        x='revenue',
        y='channel',
        orientation='h',
        title='Total Revenue by Marketing Channel',
        labels={'revenue': 'Revenue (₹)', 'channel': 'Channel'}
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: CAMPAIGN ANALYTICS
# =============================================================================
def page_campaign_analytics(data):
    """Campaign Analytics Page"""
    st.title("📈 Campaign Analytics")
    
    campaigns = data['campaigns']
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_channels = st.multiselect(
            "Select Channels",
            options=campaigns['channel'].unique(),
            default=campaigns['channel'].unique()
        )
    with col2:
        selected_regions = st.multiselect(
            "Select Regions",
            options=campaigns['region'].unique(),
            default=campaigns['region'].unique()
        )
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(campaigns['date'].min(), campaigns['date'].max()),
            min_value=campaigns['date'].min(),
            max_value=campaigns['date'].max()
        )
    
    # Filter data
    filtered = campaigns[
        (campaigns['channel'].isin(selected_channels)) &
        (campaigns['region'].isin(selected_regions)) &
        (campaigns['date'] >= pd.to_datetime(date_range[0])) &
        (campaigns['date'] <= pd.to_datetime(date_range[1]))
    ]
    
    st.markdown("---")
    
    # TODO: Grouped Bar Chart - Regional Performance by Quarter
    st.subheader("📊 Regional Performance by Quarter")
    
    regional_quarterly = filtered.groupby(['region', 'quarter'])['revenue'].sum().reset_index()
    
    fig = px.bar(
        regional_quarterly,
        x='quarter',
        y='revenue',
        color='region',
        barmode='group',
        title='Revenue by Region and Quarter',
        labels={'revenue': 'Revenue (₹)', 'quarter': 'Quarter', 'region': 'Region'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # TODO: Stacked Area Chart - Channel Contribution Over Time
    st.subheader("📈 Channel Contribution Over Time")
    
    channel_time = filtered.groupby([pd.Grouper(key='date', freq='W'), 'channel'])['conversions'].sum().reset_index()
    
    fig = px.area(
        channel_time,
        x='date',
        y='conversions',
        color='channel',
        title='Weekly Conversions by Channel (Stacked)',
        labels={'date': 'Week', 'conversions': 'Conversions', 'channel': 'Channel'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # TODO: Calendar Heatmap
    # Hint: Use plotly's heatmap with day of week vs week number

# =============================================================================
# PAGE: CUSTOMER INSIGHTS
# =============================================================================
def page_customer_insights(data):
    """Customer Insights Page"""
    st.title("👥 Customer Insights")
    
    customers = data['customers']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # TODO: Histogram - Age Distribution
        st.subheader("📊 Age Distribution")
        
        bin_size = st.slider("Bin Size", min_value=2, max_value=10, value=5)
        
        fig = px.histogram(
            customers,
            x='age',
            nbins=int((customers['age'].max() - customers['age'].min()) / bin_size),
            title='Customer Age Distribution',
            labels={'age': 'Age', 'count': 'Count'},
            color_discrete_sequence=['#3366cc']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # TODO: Box Plot - LTV by Segment
        st.subheader("📦 Lifetime Value by Segment")
        
        fig = px.box(
            customers,
            x='customer_segment',
            y='lifetime_value',
            color='customer_segment',
            title='LTV Distribution by Customer Segment',
            labels={'customer_segment': 'Segment', 'lifetime_value': 'Lifetime Value (₹)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # TODO: Scatter Plot - Income vs LTV
    st.subheader("🔵 Income vs Lifetime Value")
    
    fig = px.scatter(
        customers,
        x='income',
        y='lifetime_value',
        color='customer_segment',
        size='total_purchases',
        hover_data=['age', 'tenure_months', 'satisfaction_score'],
        title='Income vs Lifetime Value by Segment',
        labels={'income': 'Income (₹)', 'lifetime_value': 'Lifetime Value (₹)'}
    )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)
    
    # TODO: Violin Plot - Satisfaction by NPS Category

# =============================================================================
# PAGE: PRODUCT PERFORMANCE
# =============================================================================
def page_product_performance(data):
    """Product Performance Page"""
    st.title("📦 Product Performance")
    
    products = data['products']
    
    # TODO: Treemap - Product Hierarchy
    st.subheader("🌳 Product Sales Hierarchy")
    
    fig = px.treemap(
        products,
        path=['category', 'subcategory', 'product_name'],
        values='sales',
        color='profit_margin',
        color_continuous_scale='RdYlGn',
        title='Product Sales Treemap (Size: Sales, Color: Profit Margin %)'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # TODO: Sunburst Chart
    st.subheader("☀️ Sales by Category & Region")
    
    category_region = products.groupby(['category', 'region'])['sales'].sum().reset_index()
    
    fig = px.sunburst(
        category_region,
        path=['category', 'region'],
        values='sales',
        title='Sales Distribution: Category → Region'
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: GEOGRAPHIC ANALYSIS
# =============================================================================
def page_geographic_analysis(data):
    """Geographic Analysis Page"""
    st.title("🗺️ Geographic Analysis")
    
    geo = data['geographic']
    
    # Metric selector
    metric = st.selectbox(
        "Select Metric to Display",
        ['total_revenue', 'total_customers', 'market_penetration', 'yoy_growth', 'customer_satisfaction']
    )
    
    # TODO: Bubble Map
    st.subheader("📍 State-wise Performance")
    
    fig = px.scatter_geo(
        geo,
        lat='latitude',
        lon='longitude',
        size=metric,
        color='region',
        hover_name='state',
        hover_data=['total_revenue', 'total_customers', 'store_count'],
        title=f'State Performance - {metric.replace("_", " ").title()}',
        scope='asia'
    )
    fig.update_geos(
        center=dict(lat=20.5937, lon=78.9629),
        projection_scale=4,
        showland=True,
        landcolor='lightgray'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📊 State-wise Metrics Table")
    st.dataframe(
        geo[['state', 'region', 'total_customers', 'total_revenue', 'market_penetration', 'yoy_growth', 'customer_satisfaction']].sort_values('total_revenue', ascending=False),
        use_container_width=True
    )

# =============================================================================
# PAGE: ATTRIBUTION & FUNNEL
# =============================================================================
def page_attribution_funnel(data):
    """Attribution & Funnel Page"""
    st.title("🎯 Attribution & Funnel Analysis")
    
    attribution = data['attribution']
    funnel = data['funnel']
    correlation = data['correlation']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # TODO: Donut Chart - Attribution Model
        st.subheader("🍩 Channel Attribution")
        
        model = st.selectbox(
            "Select Attribution Model",
            ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']
        )
        
        fig = px.pie(
            attribution,
            values=model,
            names='channel',
            hole=0.4,
            title=f'Channel Attribution - {model.replace("_", " ").title()}'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # TODO: Funnel Chart
        st.subheader("🔻 Conversion Funnel")
        
        fig = go.Figure(go.Funnel(
            y=funnel['stage'],
            x=funnel['visitors'],
            textinfo="value+percent initial",
            marker=dict(color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3'])
        ))
        fig.update_layout(title='Marketing Funnel')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # TODO: Correlation Heatmap
    st.subheader("🔥 Metric Correlation Heatmap")
    
    fig = px.imshow(
        correlation,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Correlation Between Marketing Metrics'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: ML MODEL EVALUATION
# =============================================================================
def page_ml_evaluation(data):
    """ML Model Evaluation Page"""
    st.title("🤖 ML Model Evaluation")
    st.markdown("Lead Scoring Model Performance Analysis")
    
    leads = data['leads']
    feature_imp = data['feature_importance']
    learning = data['learning_curve']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # TODO: Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        
        y_true = leads['actual_converted']
        y_pred = (leads['predicted_probability'] >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Converted', 'Converted'],
            y=['Not Converted', 'Converted'],
            text_auto=True,
            color_continuous_scale='Blues',
            title=f'Confusion Matrix (Threshold: {threshold})'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        st.write(f"**Accuracy:** {accuracy:.3f} | **Precision:** {precision:.3f} | **Recall:** {recall:.3f}")
    
    with col2:
        # TODO: ROC Curve
        st.subheader("📈 ROC Curve")
        
        fpr, tpr, thresholds = roc_curve(leads['actual_converted'], leads['predicted_probability'])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # TODO: Feature Importance
        st.subheader("🎯 Feature Importance")
        
        feature_imp_sorted = feature_imp.sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_imp_sorted,
            x='importance',
            y='feature',
            orientation='h',
            error_x='importance_std',
            title='Feature Importance (with std dev)',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # TODO: Learning Curve
        st.subheader("📚 Learning Curve")
        
        fig = go.Figure()
        
        # Training score
        fig.add_trace(go.Scatter(
            x=learning['training_size'],
            y=learning['train_score'],
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue')
        ))
        
        # Validation score
        fig.add_trace(go.Scatter(
            x=learning['training_size'],
            y=learning['validation_score'],
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            yaxis=dict(range=[0.4, 1.0])
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application"""
    
    # Load data
    data = load_data()
    
    if data is None:
        st.stop()
    
    # Sidebar navigation
    page = sidebar()
    
    # Route to pages
    if page == "🏠 Executive Overview":
        page_executive_overview(data)
    elif page == "📈 Campaign Analytics":
        page_campaign_analytics(data)
    elif page == "👥 Customer Insights":
        page_customer_insights(data)
    elif page == "📦 Product Performance":
        page_product_performance(data)
    elif page == "🗺️ Geographic Analysis":
        page_geographic_analysis(data)
    elif page == "🎯 Attribution & Funnel":
        page_attribution_funnel(data)
    elif page == "🤖 ML Model Evaluation":
        page_ml_evaluation(data)

if __name__ == "__main__":
    main()
