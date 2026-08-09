import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NovaMart Marketing Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load all 11 marketing datasets with caching for optimal performance."""
    data_files = {
        "campaign": "campaign_performance.csv",
        "customer": "customer_data.csv",
        "product": "product_sales.csv",
        "lead_scoring": "lead_scoring_results.csv",
        "feature_importance": "feature_importance.csv",
        "learning_curve": "learning_curve.csv",
        "geographic": "geographic_data.csv",
        "attribution": "channel_attribution.csv",
        "funnel": "funnel_data.csv",
        "journey": "customer_journey.csv",
        "correlation": "correlation_matrix.csv"
    }
    
    datasets = {}
    for key, filename in data_files.items():
        if os.path.exists(filename):
            datasets[key] = pd.read_csv(filename)
            if "date" in datasets[key].columns:
                datasets[key]["date"] = pd.to_datetime(datasets[key]["date"])
        else:
            datasets[key] = None
            
    return datasets

data = load_data()

# Helper download button for data export
def add_export_button(df, filename_prefix):
    if df is not None:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Export {filename_prefix} Data (CSV)",
            data=csv,
            file_name=f"{filename_prefix}_export.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🛒 NovaMart Analytics")
st.sidebar.markdown("---")

navigation_page = st.sidebar.radio(
    "Select Perspective:",
    [
        "Page 1: Executive Overview",
        "Page 2: Campaign Analytics",
        "Page 3: Customer Insights",
        "Page 4: Product Performance",
        "Page 5: Geographic Analysis",
        "Page 6: Attribution & Funnel",
        "Page 7: ML Model Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Masters of AI in Business — Data Visualization Project")


# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
if navigation_page == "Page 1: Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("High-level overview of NovaMart's key marketing KPIs and revenue indicators.")
    st.markdown("---")
    
    campaign_df = data.get("campaign")
    customer_df = data.get("customer")
    
    # Key KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    if campaign_df is not None:
        total_rev = campaign_df["revenue"].sum()
        total_conv = campaign_df["conversions"].sum()
        total_spend = campaign_df["spend"].sum()
        overall_roas = total_rev / total_spend if total_spend > 0 else 0
    else:
        total_rev, total_conv, overall_roas = 0, 0, 0

    total_cust = len(customer_df) if customer_df is not None else 0
    
    col1.metric("Total Revenue", f"₹{total_rev:,.2f}")
    col2.metric("Total Conversions", f"{total_conv:,.0f}")
    col3.metric("Overall ROAS", f"{overall_roas:.2f}x")
    col4.metric("Total Customers", f"{total_cust:,.0f}")
    
    st.markdown("---")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("📈 Revenue Trend Line Chart")
        if campaign_df is not None:
            rev_daily = campaign_df.groupby("date")["revenue"].sum().reset_index()
            fig_line = px.line(
                rev_daily, x="date", y="revenue",
                title="Daily Revenue Trend Across All Channels",
                labels={"revenue": "Revenue (₹)", "date": "Date"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig_line.update_traces(mode="lines+markers", hovertemplate="%{x|%b %d, %Y}<br>Revenue: ₹%{y:,.2f}")
            fig_line.update_layout(xaxis_title="Date", yaxis_title="Revenue (₹)", hovermode="x unified")
            st.plotly_chart(fig_line, use_container_width=True)
            
            with st.expander("💡 View Insight"):
                st.write("Revenue shows significant growth trajectory with pronounced seasonal spikes during major holiday periods.")
        else:
            st.warning("campaign_performance.csv dataset not found.")

    with row1_col2:
        st.subheader("📊 Channel Performance Comparison")
        if campaign_df is not None:
            channel_rev = campaign_df.groupby("channel")["revenue"].sum().reset_index().sort_values(by="revenue", ascending=True)
            fig_bar = px.bar(
                channel_rev, x="revenue", y="channel", orientation="h",
                text="revenue",
                color="channel",
                title="Total Revenue Generated by Channel",
                labels={"revenue": "Total Revenue (₹)", "channel": "Marketing Channel"},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bar.update_traces(texttemplate="₹%{text:.2s}", textposition="outside")
            fig_bar.update_layout(showlegend=False, xaxis_title="Revenue (₹)", yaxis_title="Channel")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            with st.expander("💡 View Insight"):
                st.write("Google Ads and Email drive the highest overall volume, while LinkedIn exhibits higher relative conversion quality.")
        else:
            st.warning("campaign_performance.csv dataset not found.")

    st.markdown("---")
    add_export_button(campaign_df, "Executive_Overview")


# =============================================================================
# PAGE 2: CAMPAIGN ANALYTICS
# =============================================================================
elif navigation_page == "Page 2: Campaign Analytics":
    st.title("🎯 Campaign Analytics")
    st.markdown("Detailed breakdown of temporal performance, channel comparisons, and daily revenue intensity.")
    st.markdown("---")
    
    campaign_df = data.get("campaign")
    
    if campaign_df is not None:
        # Chart 1.1: Bar Chart - Metric Switcher
        st.subheader("1.1 Channel Performance Comparison")
        metric_choice = st.selectbox("Select Metric:", ["revenue", "conversions", "roas"])
        
        if metric_choice == "roas":
            agg_df = campaign_df.groupby("channel").apply(lambda x: x["revenue"].sum() / x["spend"].sum()).reset_index(name="roas")
        else:
            agg_df = campaign_df.groupby("channel")[metric_choice].sum().reset_index()
            
        agg_df = agg_df.sort_values(by=metric_choice, ascending=True)
        
        fig1_1 = px.bar(
            agg_df, x=metric_choice, y="channel", orientation="h",
            text=metric_choice, color="channel",
            title=f"Channel Comparison by {metric_choice.upper()}",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig1_1.update_traces(texttemplate="%{text:.2f}" if metric_choice == "roas" else "%{text:.2s}", textposition="outside")
        fig1_1.update_layout(showlegend=False)
        st.plotly_chart(fig1_1, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 1.2: Grouped Bar Chart
        st.subheader("1.2 Regional Performance by Quarter")
        campaign_df["year"] = campaign_df["date"].dt.year
        campaign_df["quarter"] = "Q" + campaign_df["date"].dt.quarter.astype(str)
        
        available_years = sorted(campaign_df["year"].unique().tolist())
        selected_year = st.selectbox("Select Year:", available_years, index=len(available_years)-1)
        
        filtered_q = campaign_df[campaign_df["year"] == selected_year]
        reg_q = filtered_q.groupby(["quarter", "region"])["revenue"].sum().reset_index()
        
        fig1_2 = px.bar(
            reg_q, x="quarter", y="revenue", color="region", barmode="group",
            text="revenue",
            title=f"Regional Revenue Comparison by Quarter ({selected_year})",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1_2.update_traces(texttemplate="₹%{text:.2s}", textposition="outside")
        st.plotly_chart(fig1_2, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 1.3: Stacked Bar Chart
        st.subheader("1.3 Campaign Type Contribution to Spend")
        stack_mode = st.radio("Display Mode:", ["Absolute Value", "100% Stacked"], horizontal=True)
        
        campaign_df["month_year"] = campaign_df["date"].dt.to_period("M").astype(str)
        spend_camp = campaign_df.groupby(["month_year", "campaign_type"])["spend"].sum().reset_index()
        
        fig1_3 = px.bar(
            spend_camp, x="month_year", y="spend", color="campaign_type",
            barmode="stack",
            groupnorm="fraction" if stack_mode == "100% Stacked" else None,
            title="Monthly Spend Contribution by Campaign Type",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        if stack_mode == "100% Stacked":
            fig1_3.update_layout(yaxis=dict(tickformat=".0%"), yaxis_title="Percentage Share")
        else:
            fig1_3.update_layout(yaxis_title="Total Spend (₹)")
        st.plotly_chart(fig1_3, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 2.1: Line Chart
        st.subheader("2.1 Revenue Trend Over Time")
        col_a, col_b, col_c = st.columns(3)
        
        min_date = campaign_df["date"].min().date()
        max_date = campaign_df["date"].max().date()
        
        with col_a:
            date_range = st.date_input("Date Range:", [min_date, max_date])
        with col_b:
            agg_level = st.selectbox("Aggregation:", ["Daily", "Weekly", "Monthly"])
        with col_c:
            channels = st.multiselect("Channels:", campaign_df["channel"].unique().tolist(), default=campaign_df["channel"].unique().tolist())
            
        start_d, end_d = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
        mask = (campaign_df["date"].dt.date >= start_d) & (campaign_df["date"].dt.date <= end_d) & (campaign_df["channel"].isin(channels))
        filtered_trend = campaign_df[mask].copy()
        
        if agg_level == "Weekly":
            filtered_trend["period"] = filtered_trend["date"].dt.to_period("W").dt.start_time
        elif agg_level == "Monthly":
            filtered_trend["period"] = filtered_trend["date"].dt.to_period("M").dt.start_time
        else:
            filtered_trend["period"] = filtered_trend["date"]
            
        trend_agg = filtered_trend.groupby(["period", "channel"])["revenue"].sum().reset_index()
        fig2_1 = px.line(
            trend_agg, x="period", y="revenue", color="channel",
            title=f"{agg_level} Revenue Trend by Channel",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig2_1, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 2.2: Area Chart
        st.subheader("2.2 Cumulative Conversions Over Time")
        selected_region = st.selectbox("Filter Region:", ["All"] + list(campaign_df["region"].unique()))
        
        area_data = campaign_df if selected_region == "All" else campaign_df[campaign_df["region"] == selected_region]
        area_grouped = area_data.groupby(["date", "channel"])["conversions"].sum().unstack(fill_value=0).cumsum().reset_index()
        area_melted = area_grouped.melt(id_vars=["date"], var_name="channel", value_name="cumulative_conversions")
        
        fig2_2 = px.area(
            area_melted, x="date", y="cumulative_conversions", color="channel",
            title=f"Cumulative Conversions by Channel ({selected_region})",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig2_2, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 4.4: Calendar Heatmap
        st.subheader("4.4 Daily Performance Calendar Heatmap")
        cal_year = st.selectbox("Heatmap Year:", available_years, index=0, key="cal_yr")
        cal_metric = st.selectbox("Heatmap Metric:", ["revenue", "conversions", "spend"])
        
        cal_data = campaign_df[campaign_df["year"] == cal_year].copy()
        cal_data["month"] = cal_data["date"].dt.month_name()
        cal_data["day"] = cal_data["date"].dt.day
        
        pivot_cal = cal_data.pivot_table(index="month", columns="day", values=cal_metric, aggfunc="sum").fillna(0)
        months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        pivot_cal = pivot_cal.reindex([m for m in months_order if m in pivot_cal.index])
        
        fig4_4 = px.imshow(
            pivot_cal,
            labels=dict(x="Day of Month", y="Month", color=cal_metric.capitalize()),
            title=f"Daily Intensity Heatmap for {cal_metric.capitalize()} ({cal_year})",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig4_4, use_container_width=True)

        add_export_button(campaign_df, "Campaign_Analytics")
    else:
        st.warning("campaign_performance.csv dataset missing.")


# =============================================================================
# PAGE 3: CUSTOMER INSIGHTS
# =============================================================================
elif navigation_page == "Page 3: Customer Insights":
    st.title("👥 Customer Insights")
    st.markdown("Explore customer demographic patterns, lifetime value distribution, segmentation, and multi-touchpoint journeys.")
    st.markdown("---")
    
    customer_df = data.get("customer")
    campaign_df = data.get("campaign")
    journey_df = data.get("journey")
    
    if customer_df is not None:
        # Chart 3.1: Histogram
        st.subheader("3.1 Customer Age Distribution")
        col_bin, col_seg = st.columns(2)
        with col_bin:
            bin_size = st.slider("Bin Width:", min_value=1, max_value=20, value=5)
        with col_seg:
            seg_overlay = st.multiselect("Segment Overlay:", customer_df["segment"].unique().tolist(), default=customer_df["segment"].unique().tolist())
            
        cust_filtered = customer_df[customer_df["segment"].isin(seg_overlay)]
        
        fig3_1 = px.histogram(
            cust_filtered, x="age", color="segment",
            nbins=int((cust_filtered["age"].max() - cust_filtered["age"].min()) / bin_size),
            title="Age Distribution Across Customer Segments",
            color_discrete_sequence=px.colors.qualitative.Set1,
            barmode="overlay",
            opacity=0.75
        )
        st.plotly_chart(fig3_1, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 3.2: Box Plot
        st.subheader("3.2 Lifetime Value (LTV) by Customer Segment")
        show_points = st.checkbox("Overlay Individual Data Points", value=False)
        
        fig3_2 = px.box(
            customer_df, x="segment", y="ltv", color="segment",
            points="all" if show_points else "outliers",
            title="LTV Distribution Across Customer Segments",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig3_2.update_traces(boxmean=True)
        st.plotly_chart(fig3_2, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 3.3: Violin Plot
        st.subheader("3.3 Satisfaction Score Distribution by NPS Category")
        if "nps_category" in customer_df.columns:
            fig3_3 = px.violin(
                customer_df, x="nps_category", y="satisfaction_score", color="acquisition_channel" if "acquisition_channel" in customer_df.columns else "segment",
                box=True, points="all",
                title="Satisfaction Score Distribution Split by NPS Category",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig3_3, use_container_width=True)
        else:
            st.info("nps_category column not available in customer_data.csv")
            
        st.markdown("---")
        
        # Chart 4.1: Scatter Plot
        st.subheader("4.1 Income vs. Lifetime Value (LTV)")
        show_trend = st.checkbox("Show Overall Trendline", value=True)
        
        fig4_1 = px.scatter(
            customer_df, x="income", y="ltv", color="segment",
            hover_data=["customer_id"] if "customer_id" in customer_df.columns else None,
            trendline="ols" if show_trend else None,
            title="Income vs. Lifetime Value Correlation",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        st.plotly_chart(fig4_1, use_container_width=True)
        
        st.markdown("---")
        
        # Chart 5.3: Sunburst Chart
        st.subheader("5.3 Customer Segmentation Breakdown")
        path_cols = [c for c in ["region", "city_tier", "segment"] if c in customer_df.columns]
        if len(path_cols) >= 2:
            fig5_3 = px.sunburst(
                customer_df, path=path_cols,
                title="Hierarchy: Region > City Tier > Customer Segment",
                color="segment",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig5_3, use_container_width=True)
            
        st.markdown("---")

    # Chart 4.2: Bubble Chart
    if campaign_df is not None:
        st.subheader("4.2 Channel Performance Matrix (Bubble Chart)")
        camp_agg = campaign_df.groupby("channel").agg(
            total_spend=("spend", "sum"),
            avg_ctr=("ctr", "mean"),
            avg_cvr=("conversions", lambda x: (x.sum() / campaign_df.loc[x.index, "clicks"].sum()) if campaign_df.loc[x.index, "clicks"].sum() > 0 else 0)
        ).reset_index()
        
        fig4_2 = px.scatter(
            camp_agg, x="avg_ctr", y="avg_cvr", size="total_spend", color="channel",
            text="channel",
            title="CTR vs. Conversion Rate (Bubble Size = Total Spend)",
            labels={"avg_ctr": "Average CTR", "avg_cvr": "Conversion Rate (CVR)"},
            size_max=60
        )
        fig4_2.update_traces(textposition="top center")
        st.plotly_chart(fig4_2, use_container_width=True)
        st.markdown("---")

    # BONUS CHALLENGE: SANKEY DIAGRAM
    if journey_df is not None:
        st.subheader("✨ Bonus Challenge: Customer Journey Sankey Diagram")
        
        # Extract unique nodes
        source_nodes = journey_df["source"].tolist() if "source" in journey_df.columns else []
        target_nodes = journey_df["target"].tolist() if "target" in journey_df.columns else []
        values = journey_df["value"].tolist() if "value" in journey_df.columns else []
        
        if source_nodes and target_nodes:
            all_labels = list(set(source_nodes + target_nodes))
            label_map = {label: i for i, label in enumerate(all_labels)}
            
            sources = [label_map[s] for s in source_nodes]
            targets = [label_map[t] for t in target_nodes]
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_labels,
                    color="navy"
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(31, 119, 180, 0.4)"
                )
            )])
            fig_sankey.update_layout(title_text="Multi-Touchpoint Customer Journey Paths", font_size=12)
            st.plotly_chart(fig_sankey, use_container_width=True)

    add_export_button(customer_df, "Customer_Insights")


# =============================================================================
# PAGE 4: PRODUCT PERFORMANCE
# =============================================================================
elif navigation_page == "Page 4: Product Performance":
    st.title("📦 Product Performance")
    st.markdown("Analyze sales volume, margins, and product performance across geographic regions.")
    st.markdown("---")
    
    product_df = data.get("product")
    
    if product_df is not None:
        # Chart 5.2: Treemap
        st.subheader("5.2 Product Sales Hierarchy Treemap")
        path_cols = [c for c in ["category", "subcategory", "product_name"] if c in product_df.columns]
        
        if len(path_cols) >= 2:
            fig5_2 = px.treemap(
                product_df, path=path_cols, values="sales", color="profit_margin",
                color_continuous_scale="RdYlGn",
                title="Product Sales Volume and Profit Margin Treemap"
            )
            fig5_2.update_traces(textinfo="label+value+percent parent")
            st.plotly_chart(fig5_2, use_container_width=True)
            
        st.markdown("---")
        
        # Category Comparison & Regional Performance
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Category Sales Comparison")
            cat_sales = product_df.groupby("category")["sales"].sum().reset_index()
            fig_cat = px.bar(
                cat_sales, x="category", y="sales", text="sales", color="category",
                title="Total Sales by Product Category",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_cat.update_traces(texttemplate="₹%{text:.2s}", textposition="outside")
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with col_p2:
            st.subheader("Regional Product Performance")
            if "region" in product_df.columns:
                reg_prod = product_df.groupby(["region", "category"])["sales"].sum().reset_index()
                fig_reg_prod = px.bar(
                    reg_prod, x="region", y="sales", color="category", barmode="group",
                    text="sales",
                    title="Category Sales Across Regions",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_reg_prod.update_traces(texttemplate="₹%{text:.2s}", textposition="outside")
                st.plotly_chart(fig_reg_prod, use_container_width=True)

        add_export_button(product_df, "Product_Performance")
    else:
        st.warning("product_sales.csv dataset missing.")


# =============================================================================
# PAGE 5: GEOGRAPHIC ANALYSIS
# =============================================================================
elif navigation_page == "Page 5: Geographic Analysis":
    st.title("🗺️ Geographic Analysis")
    st.markdown("Geographic distribution of sales, customer density, market penetration, and store metrics across India.")
    st.markdown("---")
    
    geo_df = data.get("geographic")
    
    if geo_df is not None:
        # Chart 6.1: Map Metric Selector
        st.subheader("6.1 State-wise Geographic Performance")
        map_metric = st.selectbox("Select Map Metric:", ["revenue", "customers", "market_penetration", "yoy_growth"])
        
        if "lat" in geo_df.columns and "lon" in geo_df.columns:
            fig6_1 = px.scatter_geo(
                geo_df, lat="lat", lon="lon", size=map_metric, color=map_metric,
                hover_name="state" if "state" in geo_df.columns else None,
                title=f"India Map - State Distribution by {map_metric.replace('_', ' ').capitalize()}",
                color_continuous_scale="Viridis",
                scope="asia"
            )
            fig6_1.update_geos(fitbounds="locations")
            st.plotly_chart(fig6_1, use_container_width=True)
        else:
            fig6_1 = px.bar(
                geo_df, x="state", y=map_metric, color=map_metric, text=map_metric,
                title=f"State Distribution by {map_metric.replace('_', ' ').capitalize()}",
                color_continuous_scale="Viridis"
            )
            fig6_1.update_traces(texttemplate="%{text:.2s}", textposition="outside")
            st.plotly_chart(fig6_1, use_container_width=True)
            
        st.markdown("---")
        
        # Chart 6.2: Bubble Map
        st.subheader("6.2 Store Performance & Customer Satisfaction")
        if "lat" in geo_df.columns and "lon" in geo_df.columns and "store_count" in geo_df.columns:
            fig6_2 = px.scatter_geo(
                geo_df, lat="lat", lon="lon",
                size="store_count" if "store_count" in geo_df.columns else "revenue",
                color="satisfaction" if "satisfaction" in geo_df.columns else "revenue",
                hover_name="state" if "state" in geo_df.columns else None,
                title="Store Count (Bubble Size) vs. Customer Satisfaction (Color)",
                color_continuous_scale="Plasma",
                scope="asia"
            )
            fig6_2.update_geos(fitbounds="locations")
            st.plotly_chart(fig6_2, use_container_width=True)

        add_export_button(geo_df, "Geographic_Analysis")
    else:
        st.warning("geographic_data.csv dataset missing.")


# =============================================================================
# PAGE 6: ATTRIBUTION & FUNNEL
# =============================================================================
elif navigation_page == "Page 6: Attribution & Funnel":
    st.title("🔀 Attribution & Marketing Funnel")
    st.markdown("Evaluate attribution model distribution, multi-step conversion funnels, and metric correlations.")
    st.markdown("---")
    
    attr_df = data.get("attribution")
    funnel_df = data.get("funnel")
    corr_df = data.get("correlation")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        # Chart 5.1: Donut Chart
        st.subheader("5.1 Attribution Model Comparison")
        if attr_df is not None:
            models = [c for c in attr_df.columns if c != "channel"]
            selected_model = st.selectbox("Select Attribution Model:", models)
            
            fig5_1 = px.pie(
                attr_df, values=selected_model, names="channel", hole=0.4,
                title=f"Channel Credit Under {selected_model.replace('_', ' ').capitalize()}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig5_1.update_traces(textinfo="percent+label")
            st.plotly_chart(fig5_1, use_container_width=True)
        else:
            st.warning("channel_attribution.csv dataset missing.")
            
    with col_f2:
        # Chart 5.4: Funnel Chart
        st.subheader("5.4 Conversion Funnel")
        if funnel_df is not None:
            fig5_4 = px.funnel(
                funnel_df, x="visitors" if "visitors" in funnel_df.columns else funnel_df.columns[1],
                y="stage" if "stage" in funnel_df.columns else funnel_df.columns[0],
                title="Marketing Funnel Progression",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig5_4, use_container_width=True)
        else:
            st.warning("funnel_data.csv dataset missing.")
            
    st.markdown("---")
    
    # Chart 4.3: Heatmap Correlation Matrix
    st.subheader("4.3 Correlation Matrix Heatmap")
    if corr_df is not None:
        corr_clean = corr_df.set_index(corr_df.columns[0]) if not pd.api.types.is_numeric_dtype(corr_df.iloc[:,0]) else corr_df
        
        fig4_3 = px.imshow(
            corr_clean,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap Between Marketing Metrics",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig4_3, use_container_width=True)
        add_export_button(corr_df, "Attribution_And_Funnel")
    else:
        st.warning("correlation_matrix.csv dataset missing.")


# =============================================================================
# PAGE 7: ML MODEL EVALUATION
# =============================================================================
elif navigation_page == "Page 7: ML Model Evaluation":
    st.title("🤖 ML Model Evaluation")
    st.markdown("Diagnostic evaluation of the AI Lead Scoring Model performance and feature importances.")
    st.markdown("---")
    
    lead_df = data.get("lead_scoring")
    feat_df = data.get("feature_importance")
    learn_df = data.get("learning_curve")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        # Chart 7.1: Confusion Matrix
        st.subheader("7.1 Confusion Matrix")
        if lead_df is not None and "actual_converted" in lead_df.columns and "predicted_probability" in lead_df.columns:
            threshold = st.slider("Classification Threshold:", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
            preds = (lead_df["predicted_probability"] >= threshold).astype(int)
            
            cm = confusion_matrix(lead_df["actual_converted"], preds)
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
            
            fig7_1 = px.imshow(
                cm_df, text_auto=True, color_continuous_scale="Blues",
                title=f"Confusion Matrix (Threshold = {threshold:.2f})"
            )
            st.plotly_chart(fig7_1, use_container_width=True)
        else:
            st.warning("lead_scoring_results.csv dataset missing.")
            
    with col_m2:
        # Chart 7.2 & Bonus PR Curve: ROC & Precision-Recall Curves
        st.subheader("7.2 ROC & Precision-Recall Curves")
        if lead_df is not None and "actual_converted" in lead_df.columns and "predicted_probability" in lead_df.columns:
            fpr, tpr, _ = roc_curve(lead_df["actual_converted"], lead_df["predicted_probability"])
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(lead_df["actual_converted"], lead_df["predicted_probability"])
            pr_auc = average_precision_score(lead_df["actual_converted"], lead_df["predicted_probability"])
            
            curve_type = st.radio("Select Curve:", ["ROC Curve", "Precision-Recall Curve"], horizontal=True)
            
            if curve_type == "ROC Curve":
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc:.2f})", mode="lines", line=dict(color="darkorange", width=2)))
                fig_curve.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline", mode="lines", line=dict(dash="dash", color="navy")))
                fig_curve.update_layout(title="Receiver Operating Characteristic (ROC)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            else:
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(x=recall, y=precision, name=f"PR (AP = {pr_auc:.2f})", mode="lines", line=dict(color="green", width=2)))
                fig_curve.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                
            st.plotly_chart(fig_curve, use_container_width=True)
        else:
            st.warning("lead_scoring_results.csv dataset missing.")
            
    st.markdown("---")
    
    col_m3, col_m4 = st.columns(2)
    
    with col_m3:
        # Chart 7.3: Learning Curve
        st.subheader("7.3 Learning Curve")
        if learn_df is not None:
            fig7_3 = go.Figure()
            
            show_ci = st.checkbox("Show Confidence Bands", value=True)
            
            train_col = "train_score" if "train_score" in learn_df.columns else learn_df.columns[1]
            val_col = "val_score" if "val_score" in learn_df.columns else learn_df.columns[2]
            size_col = "train_size" if "train_size" in learn_df.columns else learn_df.columns[0]
            
            fig7_3.add_trace(go.Scatter(x=learn_df[size_col], y=learn_df[train_col], name="Training Score", mode="lines+markers", line=dict(color="blue")))
            fig7_3.add_trace(go.Scatter(x=learn_df[size_col], y=learn_df[val_col], name="Validation Score", mode="lines+markers", line=dict(color="red")))
            
            if show_ci and "train_std" in learn_df.columns and "val_std" in learn_df.columns:
                fig7_3.add_trace(go.Scatter(
                    x=pd.concat([learn_df[size_col], learn_df[size_col][::-1]]),
                    y=pd.concat([learn_df[train_col] + learn_df["train_std"], (learn_df[train_col] - learn_df["train_std"])[::-1]]),
                    fill="toself", fillcolor="rgba(0,0,255,0.1)", line=dict(color="rgba(255,255,255,0)"), name="Train Std"
                ))
                
            fig7_3.update_layout(title="Model Learning Curve Diagnostics", xaxis_title="Training Set Size", yaxis_title="Score")
            st.plotly_chart(fig7_3, use_container_width=True)
        else:
            st.warning("learning_curve.csv dataset missing.")
            
    with col_m4:
        # Chart 7.4: Feature Importance
        st.subheader("7.4 Feature Importance")
        if feat_df is not None:
            sort_order = st.radio("Sort Importance:", ["Descending", "Ascending"], horizontal=True)
            feat_sorted = feat_df.sort_values(by="importance", ascending=(sort_order == "Ascending"))
            
            fig7_4 = px.bar(
                feat_sorted, x="importance", y="feature", orientation="h",
                text="importance",
                error_x="std" if "std" in feat_sorted.columns else None,
                title="Lead Scoring Feature Importances",
                color="importance",
                color_continuous_scale="Viridis"
            )
            fig7_4.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig7_4, use_container_width=True)
            add_export_button(feat_df, "ML_Evaluation")
        else:
            st.warning("feature_importance.csv dataset missing.")
