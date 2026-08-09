import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import io
import os

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(page_title="NovaMart Dashboard", layout="wide", page_icon="🛒")

# ==========================================
# Data Loading & Caching (UPDATED PATHS)
# ==========================================
@st.cache_data
def load_data():
    try:
        # Pointing pandas to look inside the "data" folder
        data_dir = "data"
        
        campaign_df = pd.read_csv(os.path.join(data_dir, "campaign_performance.csv"))
        # Ensure date format for time series
        if 'date' in campaign_df.columns:
            campaign_df['date'] = pd.to_datetime(campaign_df['date'])
        
        return {
            "campaign": campaign_df,
            "customer": pd.read_csv(os.path.join(data_dir, "customer_data.csv")),
            "product": pd.read_csv(os.path.join(data_dir, "product_sales.csv")),
            "lead_scoring": pd.read_csv(os.path.join(data_dir, "lead_scoring_results.csv")),
            "feature_imp": pd.read_csv(os.path.join(data_dir, "feature_importance.csv")),
            "learning_curve": pd.read_csv(os.path.join(data_dir, "learning_curve.csv")),
            "geo": pd.read_csv(os.path.join(data_dir, "geographic_data.csv")),
            "attribution": pd.read_csv(os.path.join(data_dir, "channel_attribution.csv")),
            "funnel": pd.read_csv(os.path.join(data_dir, "funnel_data.csv")),
            "journey": pd.read_csv(os.path.join(data_dir, "customer_journey.csv")),
            "corr": pd.read_csv(os.path.join(data_dir, "correlation_matrix.csv")).set_index("Unnamed: 0")
        }
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None

# ==========================================
# Helper Export Function (Bonus)
# ==========================================
def export_button(df, filename):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Data",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("🛒 NovaMart")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation", 
    ["Executive Overview", "Campaign Analytics", "Customer Insights", 
     "Product Performance", "Geographic Analysis", "Attribution & Funnel", 
     "ML Model Evaluation"]
)
st.sidebar.markdown("---")
st.sidebar.info("Select pages from the navigation menu above to explore NovaMart's performance.")

# Load Data
data = load_data()
if data is None:
    st.stop()

# ==========================================
# Page 1: Executive Overview
# ==========================================
if page == "Executive Overview":
    st.title("Executive Overview")
    st.markdown("Top-level KPIs and overarching performance metrics.")
    
    df_camp = data["campaign"]
    df_cust = data["customer"]
    
    # KPIs
    total_revenue = df_camp["revenue"].sum()
    total_conversions = df_camp["conversions"].sum()
    avg_roas = df_camp["roas"].mean()
    total_customers = df_cust["customer_id"].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Conversions", f"{total_conversions:,}")
    col3.metric("Average ROAS", f"{avg_roas:.2f}x")
    col4.metric("Total Customers", f"{total_customers:,}")
    
    st.markdown("---")
    
    # Line Chart: Revenue Trend
    st.subheader("Revenue Trend Over Time")
    col_t1, col_t2, col_t3 = st.columns([1,1,2])
    agg_toggle = col_t1.selectbox("Aggregation", ["D", "W", "M"], format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x])
    channel_filter = col_t2.multiselect("Filter Channel", options=df_camp['channel'].unique())
    
    df_trend = df_camp.copy()
    if channel_filter:
        df_trend = df_trend[df_trend['channel'].isin(channel_filter)]
    
    df_trend = df_trend.set_index('date').groupby(pd.Grouper(freq=agg_toggle))['revenue'].sum().reset_index()
    fig_trend = px.line(df_trend, x='date', y='revenue', markers=True, title="Revenue Timeline")
    st.plotly_chart(fig_trend, use_container_width=True)
    export_button(df_trend, "revenue_trend.csv")
    
    st.markdown("---")
    
    # Bar Chart: Channel Performance
    st.subheader("Channel Performance")
    metric_toggle = st.selectbox("Select Metric", ["revenue", "conversions", "roas"])
    
    channel_agg = df_camp.groupby('channel')[metric_toggle].mean() if metric_toggle == 'roas' else df_camp.groupby('channel')[metric_toggle].sum()
    channel_agg = channel_agg.reset_index().sort_values(by=metric_toggle, ascending=False)
    
    fig_bar = px.bar(channel_agg, x='channel', y=metric_toggle, color='channel', title=f"{metric_toggle.capitalize()} by Channel")
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# Page 2: Campaign Analytics
# ==========================================
elif page == "Campaign Analytics":
    st.title("Campaign Analytics")
    df_camp = data["campaign"]
    
    # Animated Feature / Time-based evolution (Bonus)
    st.subheader("Campaign Revenue Over Time (Animated)")
    df_anim = df_camp.groupby(['month', 'region'])['revenue'].sum().reset_index()
    # Ensure correct month order for animation
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df_anim['month'] = pd.Categorical(df_anim['month'], categories=month_order, ordered=True)
    df_anim = df_anim.sort_values('month')
    fig_anim = px.bar(df_anim, x="region", y="revenue", color="region",
                      animation_frame="month", animation_group="region", range_y=[0, df_anim['revenue'].max()*1.1])
    st.plotly_chart(fig_anim, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue across Regions (Quarterly)")
        year_sel = st.selectbox("Select Year", df_camp['year'].unique())
        df_reg = df_camp[df_camp['year'] == year_sel].groupby(['quarter', 'region'])['revenue'].sum().reset_index()
        fig_reg = px.bar(df_reg, x='quarter', y='revenue', color='region', barmode='group')
        st.plotly_chart(fig_reg, use_container_width=True)
        
    with col2:
        st.subheader("Campaign Type Contribution to Spend")
        stack_mode = st.radio("Stack Mode", ["Absolute", "100% Stacked"], horizontal=True)
        norm = 'percent' if stack_mode == "100% Stacked" else None
        df_type = df_camp.groupby(['month', 'campaign_type'])['spend'].sum().reset_index()
        fig_type = px.histogram(df_type, x='month', y='spend', color='campaign_type', barnorm=norm)
        st.plotly_chart(fig_type, use_container_width=True)

    # Area Chart
    st.subheader("Cumulative Conversions by Channel")
    reg_filter = st.selectbox("Region Filter", ["All"] + list(df_camp['region'].unique()))
    df_area = df_camp if reg_filter == "All" else df_camp[df_camp['region'] == reg_filter]
    df_area = df_area.groupby(['date', 'channel'])['conversions'].sum().reset_index()
    df_area['cum_conversions'] = df_area.groupby('channel')['conversions'].cumsum()
    fig_area = px.area(df_area, x='date', y='cum_conversions', color='channel')
    st.plotly_chart(fig_area, use_container_width=True)

# ==========================================
# Page 3: Customer Insights
# ==========================================
elif page == "Customer Insights":
    st.title("Customer Insights")
    df_cust = data["customer"]
    df_camp = data["campaign"]
    
    st.subheader("Customer Demographics & Segments")
    tab1, tab2, tab3 = st.tabs(["Age Distribution", "LTV by Segment", "Segmentation Hierarchy"])
    
    with tab1:
        c1, c2 = st.columns([1,3])
        bins = c1.slider("Bin Size", 5, 20, 10)
        seg = c1.selectbox("Filter Segment", ["All"] + list(df_cust['customer_segment'].unique()))
        
        df_age = df_cust if seg == "All" else df_cust[df_cust['customer_segment'] == seg]
        fig_age = px.histogram(df_age, x='age', nbins=bins, title="Age Distribution")
        c2.plotly_chart(fig_age, use_container_width=True)
        
    with tab2:
        overlay = st.checkbox("Overlay Individual Data Points")
        fig_box = px.box(df_cust, x="customer_segment", y="lifetime_value", color="customer_segment", 
                         points="all" if overlay else "outliers", title="LTV by Segment")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab3:
        fig_sun = px.sunburst(df_cust, path=['region', 'city_tier', 'customer_segment'], 
                              title="Hierarchy sized by Customer Count")
        st.plotly_chart(fig_sun, use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Satisfaction Score")
        split_chan = st.checkbox("Split by Acquisition Channel")
        y_val = "acquisition_channel" if split_chan else None
        fig_vio = px.violin(df_cust, x="nps_category", y="satisfaction_score", color="nps_category", box=True)
        st.plotly_chart(fig_vio, use_container_width=True)
        
    with col2:
        st.subheader("Income vs LTV")
        trend = st.checkbox("Show Trend Line")
        fig_scat = px.scatter(df_cust, x="income", y="lifetime_value", color="customer_segment", 
                              trendline="ols" if trend else None, hover_data=['age', 'city_tier'])
        st.plotly_chart(fig_scat, use_container_width=True)

    st.subheader("CTR vs. Conversion Rate Matrix")
    # Aggregate camp data to channel level
    df_bub = df_camp.groupby("channel").agg({"ctr":"mean", "conversion_rate":"mean", "spend":"sum"}).reset_index()
    fig_bub = px.scatter(df_bub, x="ctr", y="conversion_rate", size="spend", color="channel", 
                         hover_name="channel", size_max=60, title="Bubble size = Total Spend")
    st.plotly_chart(fig_bub, use_container_width=True)

# ==========================================
# Page 4: Product Performance
# ==========================================
elif page == "Product Performance":
    st.title("Product Performance")
    df_prod = data["product"]
    
    st.subheader("Product Hierarchy")
    st.markdown("Explore category, subcategory, and individual product sales. (Sized by Sales, Colored by Profit Margin)")
    
    fig_tree = px.treemap(df_prod, path=[px.Constant("All Products"), 'category', 'subcategory', 'product_name'], 
                          values='sales', color='profit_margin', color_continuous_scale='RdYlGn',
                          hover_data=['units_sold', 'avg_rating'])
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.dataframe(df_prod)
    export_button(df_prod, "product_data.csv")

# ==========================================
# Page 5: Geographic Analysis
# ==========================================
elif page == "Geographic Analysis":
    st.title("Geographic Analysis")
    df_geo = data["geo"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("State-level Analysis")
        metric_geo = st.selectbox("Select Metric", ["total_revenue", "total_customers", "market_penetration", "yoy_growth"])
        
        # We simulate choropleth using scatter_geo for simplicity since Folium needs strict geojson mapping for India states.
        # Bubble size reflects metric. 
        fig_map = px.scatter_geo(df_geo, 
                                 lat='latitude', lon='longitude', 
                                 size=metric_geo, color='region',
                                 hover_name='state', 
                                 scope='asia', center={'lat':20.5937, 'lon':78.9629},
                                 title=f"India: {metric_geo.replace('_', ' ').capitalize()}")
        fig_map.update_geos(fitbounds="locations")
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col2:
        st.subheader("Store Distribution")
        fig_bubble = px.scatter_mapbox(df_geo, lat="latitude", lon="longitude", hover_name="state",
                                       size="store_count", color="customer_satisfaction",
                                       color_continuous_scale=px.colors.cyclical.IceFire, 
                                       size_max=25, zoom=3, mapbox_style="carto-positron")
        st.plotly_chart(fig_bubble, use_container_width=True)

# ==========================================
# Page 6: Attribution & Funnel
# ==========================================
elif page == "Attribution & Funnel":
    st.title("Attribution & Marketing Funnel")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Marketing Funnel")
        df_funnel = data["funnel"]
        fig_fun = go.Figure(go.Funnel(
            y=df_funnel['stage'],
            x=df_funnel['visitors'],
            textinfo="value+percent initial"
        ))
        st.plotly_chart(fig_fun, use_container_width=True)
        
    with col2:
        st.subheader("Channel Attribution Models")
        df_attr = data["attribution"]
        attr_model = st.selectbox("Attribution Model", ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based'])
        fig_donut = px.pie(df_attr, values=attr_model, names='channel', hole=0.4)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Metric Correlation Matrix")
    df_corr = data["corr"]
    fig_corr = px.imshow(df_corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Customer Journeys (Sankey Diagram - Bonus)")
    df_journey = data["journey"]
    
    # Create simple node list and links for Sankey
    # Touchpoint 1 -> 2, 2 -> 3, 3 -> 4
    nodes = list(set(df_journey['touchpoint_1']).union(set(df_journey['touchpoint_2'])).union(
                 set(df_journey['touchpoint_3'])).union(set(df_journey['touchpoint_4'])))
    node_dict = {node: i for i, node in enumerate(nodes)}
    
    source = []
    target = []
    value = []
    
    for _, row in df_journey.iterrows():
        source.extend([node_dict[row['touchpoint_1']], node_dict[row['touchpoint_2']], node_dict[row['touchpoint_3']]])
        target.extend([node_dict[row['touchpoint_2']], node_dict[row['touchpoint_3']], node_dict[row['touchpoint_4']]])
        value.extend([row['customer_count']]*3)
        
    fig_sankey = go.Figure(data=[go.Sankey(
        node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5), label = nodes),
        link = dict(source = source, target = target, value = value)
    )])
    st.plotly_chart(fig_sankey, use_container_width=True)

# ==========================================
# Page 7: ML Model Evaluation
# ==========================================
elif page == "ML Model Evaluation":
    st.title("Lead Scoring ML Model Evaluation")
    
    df_ls = data["lead_scoring"]
    
    st.subheader("Classification Threshold & Confusion Matrix")
    thresh = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Calculate predictions based on threshold
    preds = (df_ls['predicted_probability'] >= thresh).astype(int)
    cm = confusion_matrix(df_ls['actual_converted'], preds)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                           x=['Not Converted (0)', 'Converted (1)'], 
                           y=['Not Converted (0)', 'Converted (1)'], 
                           color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        st.subheader("ROC Curve & Precision-Recall (Bonus)")
        tab_roc, tab_pr = st.tabs(["ROC Curve", "PR Curve"])
        
        with tab_roc:
            fpr, tpr, thresholds = roc_curve(df_ls['actual_converted'], df_ls['predicted_probability'])
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
            fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with tab_pr:
            precision, recall, pr_thresh = precision_recall_curve(df_ls['actual_converted'], df_ls['predicted_probability'])
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
            fig_pr.update_layout(xaxis_title='Recall', yaxis_title='Precision')
            st.plotly_chart(fig_pr, use_container_width=True)
            
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Learning Curve")
        df_lc = data["learning_curve"]
        show_bands = st.checkbox("Show Confidence Bands", value=True)
        
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(x=df_lc['training_size'], y=df_lc['train_score'], mode='lines+markers', name='Train Score'))
        fig_lc.add_trace(go.Scatter(x=df_lc['training_size'], y=df_lc['validation_score'], mode='lines+markers', name='Validation Score'))
        
        if show_bands:
            fig_lc.add_trace(go.Scatter(x=pd.concat([df_lc['training_size'], df_lc['training_size'][::-1]]),
                                        y=pd.concat([df_lc['train_score'] + df_lc['train_score_std'], (df_lc['train_score'] - df_lc['train_score_std'])[::-1]]),
                                        fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            fig_lc.add_trace(go.Scatter(x=pd.concat([df_lc['training_size'], df_lc['training_size'][::-1]]),
                                        y=pd.concat([df_lc['validation_score'] + df_lc['validation_score_std'], (df_lc['validation_score'] - df_lc['validation_score_std'])[::-1]]),
                                        fill='toself', fillcolor='rgba(255,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        st.plotly_chart(fig_lc, use_container_width=True)
        
    with col4:
        st.subheader("Feature Importance")
        df_fi = data["feature_imp"]
        sort_asc = st.checkbox("Sort Ascending", value=True)
        df_fi = df_fi.sort_values(by="importance", ascending=sort_asc)
        
        fig_fi = px.bar(df_fi, x="importance", y="feature", orientation='h', error_x="importance_std", title="Model Features")
        st.plotly_chart(fig_fi, use_container_width=True)
