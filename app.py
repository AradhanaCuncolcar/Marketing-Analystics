import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(page_title="NovaMart Dashboard", layout="wide", page_icon="🛒")

# ==========================================
# Helper Functions
# ==========================================
def format_large_number(num):
    """Formats large numbers into K, M, B for cleaner display."""
    try:
        num = float(num)
        if abs(num) >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif abs(num) >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif abs(num) >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return f"{num:.2f}"
    except (ValueError, TypeError):
        return num

def export_button(df, filename):
    """Provides a unified download button for dataframes."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Data", data=csv, file_name=filename, mime='text/csv')

# ==========================================
# Data Loading & Caching
# ==========================================
@st.cache_data
def load_data():
    try:
        data_dir = "data"
        campaign_df = pd.read_csv(os.path.join(data_dir, "campaign_performance.csv"))
        if 'date' in campaign_df.columns:
            campaign_df['date'] = pd.to_datetime(campaign_df['date'], format='%d/%m/%Y')
        
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
# Modular Visualization Functions
# ==========================================
def plot_revenue_trend(df, agg_toggle):
    df_grouped = df.set_index('date').groupby(pd.Grouper(freq=agg_toggle))['revenue'].sum().reset_index()
    fig = px.line(df_grouped, x='date', y='revenue', markers=True, title="Revenue Timeline")
    return fig, df_grouped

def plot_channel_performance(df, metric):
    channel_agg = df.groupby('channel')[metric].mean() if metric == 'roas' else df.groupby('channel')[metric].sum()
    channel_agg = channel_agg.reset_index().sort_values(by=metric, ascending=True) 
    fig = px.bar(channel_agg, x=metric, y='channel', orientation='h', color='channel', 
                 title=f"{metric.capitalize()} by Channel", text_auto='.2s')
    fig.update_traces(marker_line_color='black', marker_line_width=1, textposition='outside')
    return fig

def plot_campaign_revenue_animated(df):
    df_anim = df.groupby(['month', 'region'])['revenue'].sum().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df_anim['month'] = pd.Categorical(df_anim['month'], categories=month_order, ordered=True)
    df_anim = df_anim.sort_values('month')
    fig = px.bar(df_anim, x="region", y="revenue", color="region", animation_frame="month", 
                 animation_group="region", range_y=[0, df_anim['revenue'].max()*1.1], text_auto='.2s')
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    return fig

def plot_regional_revenue_qtr(df):
    fig = px.bar(df, x='quarter', y='revenue', color='region', barmode='group', text_auto='.2s')
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    return fig

def plot_campaign_spend_contribution(df, norm):
    fig = px.histogram(df, x='month', y='spend', color='campaign_type', barnorm=norm, text_auto='.2s')
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    return fig

def plot_cumulative_conversions(df):
    df['cum_conversions'] = df.groupby('channel')['conversions'].cumsum()
    fig = px.area(df, x='date', y='cum_conversions', color='channel')
    return fig

def plot_calendar_heatmap(df, year, metric):
    df_yr = df[df['year'] == year].copy()
    if df_yr.empty: return go.Figure(layout=dict(title="No data for selected year"))
    
    daily = df_yr.groupby('date')[metric].sum().reset_index()
    full_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')
    
    # Reindex safely
    daily = daily.set_index('date').reindex(full_dates).fillna(0).reset_index()
    daily.rename(columns={'index': 'date'}, inplace=True)
    
    # Extract week and weekday attributes
    daily['week'] = daily['date'].dt.isocalendar().week.astype(int)
    daily['weekday'] = daily['date'].dt.weekday
    
    # Use pivot_table to safely aggregate any overlapping ISO calendar weeks at the start/end of the year
    pivot = daily.pivot_table(index='weekday', columns='week', values=metric, aggfunc='sum')
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig = px.imshow(pivot, labels=dict(x="ISO Week", y="Day", color=metric.capitalize()),
                    y=day_names, title=f"Daily {metric.capitalize()} Heatmap ({year})",
                    color_continuous_scale="Greens", aspect="auto")
    return fig

def plot_age_distribution(df, bins):
    fig = px.histogram(df, x='age', nbins=bins, title="Age Distribution", text_auto=True)
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    return fig

def plot_ltv_boxplot(df, overlay):
    points = "all" if overlay else "outliers"
    fig = px.box(df, x="customer_segment", y="lifetime_value", color="customer_segment", points=points, title="LTV by Segment")
    return fig

def plot_satisfaction_violin(df, split_by_channel):
    color_col = "acquisition_channel" if split_by_channel else "nps_category"
    return px.violin(df, x="nps_category", y="satisfaction_score", color=color_col, box=True)

def plot_sunburst_segments(df):
    fig = px.sunburst(df, path=['region', 'city_tier', 'customer_segment'], title="Hierarchy sized by Customer Count")
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    return fig

def plot_income_ltv_scatter(df, trend):
    trendline_type = "ols" if trend else None
    return px.scatter(df, x="income", y="lifetime_value", color="customer_segment", trendline=trendline_type, hover_data=['age', 'city_tier'])

def plot_ctr_cvr_bubble(df):
    return px.scatter(df, x="ctr", y="conversion_rate", size="spend", color="channel", hover_name="channel", size_max=60, title="Bubble size = Total Spend")

def plot_product_treemap(df):
    fig = px.treemap(df, path=[px.Constant("All Products"), 'category', 'subcategory', 'product_name'], values='sales', color='profit_margin', color_continuous_scale='RdYlGn', hover_data=['units_sold', 'avg_rating'])
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    return fig

def plot_geo_choropleth(df, metric):
    fig = px.scatter_geo(df, lat='latitude', lon='longitude', size=metric, color='region', hover_name='state', scope='asia', center={'lat':20.5937, 'lon':78.9629}, title=f"India: {metric.replace('_', ' ').capitalize()}")
    fig.update_geos(fitbounds="locations")
    return fig

def plot_geo_store_bubbles(df):
    return px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="state", size="store_count", color="customer_satisfaction", color_continuous_scale=px.colors.cyclical.IceFire, size_max=25, zoom=3, mapbox_style="carto-positron")

def plot_funnel(df):
    fig = go.Figure(go.Funnel(y=df['stage'], x=df['visitors'], textinfo="value+percent initial"))
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    return fig

def plot_attribution_donut(df, model, total_conversions):
    fig = px.pie(df, values=model, names='channel', hole=0.5)
    fig.update_traces(textinfo='label+percent', textposition='inside', marker_line_color='black', marker_line_width=1)
    fig.update_layout(annotations=[dict(text=f"Total<br>{total_conversions:,.0f}", x=0.5, y=0.5, font_size=16, showarrow=False)])
    return fig

def plot_correlation_heatmap(df):
    return px.imshow(df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')

def plot_sankey_journey(df):
    nodes = list(set(df['touchpoint_1']).union(set(df['touchpoint_2'])).union(set(df['touchpoint_3'])).union(set(df['touchpoint_4'])))
    node_dict = {node: i for i, node in enumerate(nodes)}
    source, target, value = [], [], []
    for _, row in df.iterrows():
        source.extend([node_dict[row['touchpoint_1']], node_dict[row['touchpoint_2']], node_dict[row['touchpoint_3']]])
        target.extend([node_dict[row['touchpoint_2']], node_dict[row['touchpoint_3']], node_dict[row['touchpoint_4']]])
        value.extend([row['customer_count']]*3)
    return go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes), link=dict(source=source, target=target, value=value))])

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum()
    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1%})" for j in range(2)] for i in range(2)]
    
    fig = px.imshow(cm, text_auto=False, labels=dict(x="Predicted", y="Actual"), x=['Not Converted (0)', 'Converted (1)'], y=['Not Converted (0)', 'Converted (1)'], color_continuous_scale="Blues")
    fig.update_traces(text=text, texttemplate="%{text}")
    return fig

def plot_roc_curve_with_optimal(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    opt_fpr, opt_tpr, opt_thresh = fpr[optimal_idx], tpr[optimal_idx], thresholds[optimal_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[opt_fpr], y=[opt_tpr], mode='markers', marker=dict(color='red', size=12, symbol='star'), name=f'Optimal Threshold ({opt_thresh:.2f})'))
    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curve')
    return fig

def plot_pr_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
    fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
    return fig

def plot_learning_curve(df, show_bands):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['training_size'], y=df['train_score'], mode='lines+markers', name='Train Score'))
    fig.add_trace(go.Scatter(x=df['training_size'], y=df['validation_score'], mode='lines+markers', name='Validation Score'))
    if show_bands:
        fig.add_trace(go.Scatter(x=pd.concat([df['training_size'], df['training_size'][::-1]]), y=pd.concat([df['train_score'] + df['train_score_std'], (df['train_score'] - df['train_score_std'])[::-1]]), fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=pd.concat([df['training_size'], df['training_size'][::-1]]), y=pd.concat([df['validation_score'] + df['validation_score_std'], (df['validation_score'] - df['validation_score_std'])[::-1]]), fill='toself', fillcolor='rgba(255,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
    return fig

def plot_feature_importance(df, sort_asc, show_error):
    df_sorted = df.sort_values(by="importance", ascending=sort_asc)
    err_col = "importance_std" if show_error else None
    fig = px.bar(df_sorted, x="importance", y="feature", orientation='h', error_x=err_col, title="Model Features", text_auto='.2f')
    fig.update_traces(marker_line_color='black', marker_line_width=1, textposition='outside')
    return fig

# ==========================================
# Main App Execution
# ==========================================
st.sidebar.title("🛒 NovaMart")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Executive Overview", "Campaign Analytics", "Customer Insights", "Product Performance", "Geographic Analysis", "Attribution & Funnel", "ML Model Evaluation"])
st.sidebar.markdown("---")

data = load_data()
if data is None:
    st.stop()

# ==========================================
# Page 1: Executive Overview
# ==========================================
if page == "Executive Overview":
    st.title("Executive Overview")
    df_camp = data["campaign"]
    df_cust = data["customer"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True): st.metric("Total Revenue", f"${format_large_number(df_camp['revenue'].sum())}")
    with col2:
        with st.container(border=True): st.metric("Total Conversions", f"{format_large_number(df_camp['conversions'].sum())}")
    with col3:
        with st.container(border=True): st.metric("Average ROAS", f"{df_camp['roas'].mean():.2f}x")
    with col4:
        with st.container(border=True): st.metric("Total Customers", f"{format_large_number(df_cust['customer_id'].nunique())}")
    
    with st.container(border=True):
        st.subheader("Revenue Trend Over Time")
        col_t1, col_t2, col_t3 = st.columns([1,1,2])
        agg_toggle = col_t1.selectbox("Aggregation", ["D", "W", "M"], format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x])
        channel_filter = col_t2.multiselect("Filter Channel", options=df_camp['channel'].unique())
        
        min_date, max_date = df_camp['date'].min(), df_camp['date'].max()
        selected_dates = col_t3.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        df_trend = df_camp.copy()
        if channel_filter:
            df_trend = df_trend[df_trend['channel'].isin(channel_filter)]
            
        if len(selected_dates) == 2:
            start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
            df_trend = df_trend[(df_trend['date'] >= start_date) & (df_trend['date'] <= end_date)]
            
        fig_trend, processed_df = plot_revenue_trend(df_trend, agg_toggle)
        st.plotly_chart(fig_trend, use_container_width=True)
        export_button(processed_df, "revenue_trend.csv")
    
    with st.container(border=True):
        st.subheader("Channel Performance")
        metric_toggle = st.selectbox("Select Metric", ["revenue", "conversions", "roas"])
        st.plotly_chart(plot_channel_performance(df_camp, metric_toggle), use_container_width=True)

# ==========================================
# Page 2: Campaign Analytics
# ==========================================
elif page == "Campaign Analytics":
    st.title("Campaign Analytics")
    df_camp = data["campaign"]
    
    with st.container(border=True):
        st.subheader("Campaign Revenue Over Time (Animated)")
        st.plotly_chart(plot_campaign_revenue_animated(df_camp), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Revenue across Regions (Quarterly)")
            year_sel = st.selectbox("Select Year", df_camp['year'].unique(), key="qtr_year")
            df_reg = df_camp[df_camp['year'] == year_sel].groupby(['quarter', 'region'])['revenue'].sum().reset_index()
            st.plotly_chart(plot_regional_revenue_qtr(df_reg), use_container_width=True)
            
        with st.container(border=True):
            st.subheader("Daily Performance (Calendar Heatmap)")
            h_year = st.selectbox("Select Year", df_camp['year'].unique(), key="heat_year")
            h_metric = st.selectbox("Select Metric", ["revenue", "conversions", "spend"], key="heat_metric")
            st.plotly_chart(plot_calendar_heatmap(df_camp, h_year, h_metric), use_container_width=True)
        
    with col2:
        with st.container(border=True):
            st.subheader("Campaign Type Contribution to Spend")
            stack_mode = st.radio("Stack Mode", ["Absolute", "100% Stacked"], horizontal=True)
            norm = 'percent' if stack_mode == "100% Stacked" else None
            df_type = df_camp.groupby(['month', 'campaign_type'])['spend'].sum().reset_index()
            st.plotly_chart(plot_campaign_spend_contribution(df_type, norm), use_container_width=True)

        with st.container(border=True):
            st.subheader("Cumulative Conversions by Channel")
            reg_filter = st.selectbox("Region Filter", ["All"] + list(df_camp['region'].unique()))
            df_area = df_camp if reg_filter == "All" else df_camp[df_camp['region'] == reg_filter]
            df_area = df_area.groupby(['date', 'channel'])['conversions'].sum().reset_index()
            st.plotly_chart(plot_cumulative_conversions(df_area), use_container_width=True)

# ==========================================
# Page 3: Customer Insights
# ==========================================
elif page == "Customer Insights":
    st.title("Customer Insights")
    df_cust, df_camp = data["customer"], data["campaign"]
    
    with st.container(border=True):
        st.subheader("Customer Demographics & Segments")
        tab1, tab2, tab3 = st.tabs(["Age Distribution", "LTV by Segment", "Segmentation Hierarchy"])
        with tab1:
            c1, c2 = st.columns([1,3])
            bins = c1.slider("Bin Size", 5, 20, 10)
            seg = c1.selectbox("Filter Segment", ["All"] + list(df_cust['customer_segment'].unique()))
            df_age = df_cust if seg == "All" else df_cust[df_cust['customer_segment'] == seg]
            c2.plotly_chart(plot_age_distribution(df_age, bins), use_container_width=True)
        with tab2:
            overlay = st.checkbox("Overlay Individual Data Points")
            st.plotly_chart(plot_ltv_boxplot(df_cust, overlay), use_container_width=True)
        with tab3:
            st.plotly_chart(plot_sunburst_segments(df_cust), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Satisfaction Score")
            split_chan = st.checkbox("Split by Acquisition Channel")
            st.plotly_chart(plot_satisfaction_violin(df_cust, split_chan), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Income vs LTV")
            trend = st.checkbox("Show Trend Line")
            st.plotly_chart(plot_income_ltv_scatter(df_cust, trend), use_container_width=True)

    with st.container(border=True):
        st.subheader("CTR vs. Conversion Rate Matrix")
        df_bub = df_camp.groupby("channel").agg({"ctr":"mean", "conversion_rate":"mean", "spend":"sum"}).reset_index()
        st.plotly_chart(plot_ctr_cvr_bubble(df_bub), use_container_width=True)

# ==========================================
# Page 4: Product Performance
# ==========================================
elif page == "Product Performance":
    st.title("Product Performance")
    df_prod = data["product"]
    
    with st.container(border=True):
        st.subheader("Product Hierarchy")
        st.plotly_chart(plot_product_treemap(df_prod), use_container_width=True)
        st.dataframe(df_prod, use_container_width=True)
        export_button(df_prod, "product_data.csv")

# ==========================================
# Page 5: Geographic Analysis
# ==========================================
elif page == "Geographic Analysis":
    st.title("Geographic Analysis")
    df_geo = data["geo"]
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("State-level Analysis")
            metric_geo = st.selectbox("Select Metric", ["total_revenue", "total_customers", "market_penetration", "yoy_growth"])
            st.plotly_chart(plot_geo_choropleth(df_geo, metric_geo), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Store Distribution")
            st.plotly_chart(plot_geo_store_bubbles(df_geo), use_container_width=True)

# ==========================================
# Page 6: Attribution & Funnel
# ==========================================
elif page == "Attribution & Funnel":
    st.title("Attribution & Marketing Funnel")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Marketing Funnel")
            st.plotly_chart(plot_funnel(data["funnel"]), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Channel Attribution Models")
            attr_model = st.selectbox("Attribution Model", ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based'])
            total_conv = data["campaign"]["conversions"].sum()
            st.plotly_chart(plot_attribution_donut(data["attribution"], attr_model, total_conv), use_container_width=True)
        
    with st.container(border=True):
        st.subheader("Metric Correlation Matrix")
        st.plotly_chart(plot_correlation_heatmap(data["corr"]), use_container_width=True)
    
    with st.container(border=True):
        st.subheader("Customer Journeys (Sankey Diagram - Bonus)")
        st.plotly_chart(plot_sankey_journey(data["journey"]), use_container_width=True)

# ==========================================
# Page 7: ML Model Evaluation
# ==========================================
elif page == "ML Model Evaluation":
    st.title("Lead Scoring ML Model Evaluation")
    df_ls = data["lead_scoring"]
    
    with st.container(border=True):
        st.subheader("Classification Threshold & Confusion Matrix")
        thresh = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        preds = (df_ls['predicted_probability'] >= thresh).astype(int)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_confusion_matrix(df_ls['actual_converted'], preds), use_container_width=True)
        with col2:
            st.markdown("**ROC Curve & Precision-Recall (Bonus)**")
            tab_roc, tab_pr = st.tabs(["ROC Curve", "PR Curve"])
            with tab_roc:
                st.plotly_chart(plot_roc_curve_with_optimal(df_ls['actual_converted'], df_ls['predicted_probability']), use_container_width=True)
            with tab_pr:
                st.plotly_chart(plot_pr_curve(df_ls['actual_converted'], df_ls['predicted_probability']), use_container_width=True)
            
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.subheader("Learning Curve")
            show_bands = st.checkbox("Show Confidence Bands", value=True)
            st.plotly_chart(plot_learning_curve(data["learning_curve"], show_bands), use_container_width=True)
    with col4:
        with st.container(border=True):
            st.subheader("Feature Importance")
            col_fi1, col_fi2 = st.columns(2)
            sort_asc = col_fi1.checkbox("Sort Ascending", value=True)
            show_error = col_fi2.checkbox("Show Error Bars", value=True)
            st.plotly_chart(plot_feature_importance(data["feature_imp"], sort_asc, show_error), use_container_width=True)
