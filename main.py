import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="E-commerce Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #1a1f2e 0%, #242938 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2d3348;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
    }
    .stMetric {
        background: transparent;
    }
    .stMetric label {
        color: #8b95a8 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        background: linear-gradient(90deg, #00d4ff 0%, #00a3cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #2d3348;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #e0e6ed !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
    }
    .caption {
        color: #8b95a8 !important;
        font-size: 1rem !important;
    }
    .stDownloadButton button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.3) !important;
    }
    .success-message {
        background: linear-gradient(135deg, #1a4d2e 0%, #1f5f3b 100%);
        border-left: 4px solid #4ade80;
        padding: 1rem;
        border-radius: 8px;
        color: #ffffff;
        margin: 1rem 0;
    }
    .info-message {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        color: #ffffff;
        margin: 1rem 0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141824 0%, #1a1f2e 100%);
    }
    .uploadedFile {
        background-color: #1a1f2e !important;
        border: 1px solid #2d3348 !important;
    }
    .stDataFrame {
        background-color: #1a1f2e;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("E-commerce Analytics Platform")
st.markdown('<p class="caption">Advanced Business Intelligence & Performance Metrics Dashboard</p>', unsafe_allow_html=True)

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.sidebar.file_uploader(
    "Upload Orders Dataset",
    type=["csv", "xlsx"],
    help="Upload CSV or Excel file containing order data"
)

if uploaded_file is None:
    st.markdown('<div class="info-message">Please upload a dataset from the sidebar to begin analysis.</div>', unsafe_allow_html=True)
    st.stop()

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# =====================================================
# SECTION 1 — DATA CLEANING & FEATURE ENGINEERING
# =====================================================
with st.spinner("Processing and cleaning data..."):
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")

    df["Discount"] = (
        df["DiscountText"]
        .astype(str)
        .str.replace("%", "")
        .str.strip()
    )
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce") / 100
    df["Discount"].fillna(0, inplace=True)

    df["Quantity_numeric"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Quantity_Valid"] = df["Quantity_numeric"].notna()

    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df["TotalSales"] = (
        df["Quantity_numeric"] *
        df["UnitPrice"] *
        (1 - df["Discount"])
    )

    df["DeliveryDays"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days
    df.loc[df["DeliveryDays"] < 0, "DeliveryDays"] = np.nan

    df["OrderYear"] = df["OrderDate"].dt.year
    df["OrderMonth"] = df["OrderDate"].dt.month

    df["MonthName"] = df["OrderDate"].dt.strftime("%B")

    df["MissingDelivery"] = df["DeliveryDate"].isna()

    df["UrgentOrder"] = df["Notes"].astype(str).str.contains("urgent", case=False)

    def segment(x):
        if x > 5000:
            return "High Value"
        elif x > 2000:
            return "Medium Value"
        else:
            return "Low Value"

    df["CustomerSegment"] = df["TotalSales"].apply(segment)

    median = df["TotalSales"].median()
    std = df["TotalSales"].std()
    df["Outlier"] = abs(df["TotalSales"] - median) > std

st.markdown('<div class="success-message">Data processing complete. All metrics calculated successfully.</div>', unsafe_allow_html=True)

# =====================================================
# EXECUTIVE DASHBOARD — KEY METRICS
# =====================================================
st.header("Executive Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    total_revenue = df['TotalSales'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    total_orders = len(df)
    st.metric("Total Orders", f"{total_orders:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    avg_order_value = df['TotalSales'].mean()
    st.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    valid_delivery = df["DeliveryDays"].dropna()
    avg_delivery = valid_delivery.mean()
    st.metric("Avg Delivery Days", f"{avg_delivery:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    unique_customers = df['CustomerName'].nunique()
    st.metric("Unique Customers", f"{unique_customers:,}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    median_order = df['TotalSales'].median()
    st.metric("Median Order Value", f"${median_order:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    delivered_orders = (df["OrderStatus"] == "Delivered").sum()
    st.metric("Delivered Orders", f"{delivered_orders:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    pending_orders = (df["OrderStatus"] == "Pending").sum()
    st.metric("Pending Orders", f"{pending_orders:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    returned_orders = (df["OrderStatus"] == "Returned").sum()
    st.metric("Returned Orders", f"{returned_orders:,}")
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# REVENUE ANALYTICS
# =====================================================
st.header("Revenue Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Region")
    rev_region = df.groupby("Region")["TotalSales"].sum().reset_index()
    fig = go.Figure(data=[
        go.Bar(
            x=rev_region["Region"],
            y=rev_region["TotalSales"],
            marker=dict(
                color=rev_region["TotalSales"],
                colorscale="Teal",
                showscale=False
            ),
            text=rev_region["TotalSales"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside"
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue by Product Category")
    rev_category = df.groupby("ProductCategory")["TotalSales"].sum().reset_index()
    fig = go.Figure(data=[
        go.Pie(
            labels=rev_category["ProductCategory"],
            values=rev_category["TotalSales"],
            hole=0.4,
            marker=dict(colors=["#00d4ff", "#00a3cc", "#007a99", "#005266"]),
            textinfo="label+percent",
            textfont=dict(size=14, color="#ffffff")
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Revenue Trend")
    monthly_sales = (
        df.groupby(["OrderYear", "OrderMonth", "MonthName"])["TotalSales"]
        .sum()
        .reset_index()
        .sort_values(["OrderYear", "OrderMonth"])
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sales["MonthName"],
        y=monthly_sales["TotalSales"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=10, color="#00d4ff", line=dict(color="#ffffff", width=2)),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.1)"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        xaxis=dict(showgrid=True, gridcolor="#2d3348"),
        yaxis=dict(showgrid=True, gridcolor="#2d3348")
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Average Order Value by Category")
    avg_cat = df.groupby("ProductCategory")["TotalSales"].mean().reset_index()
    fig = go.Figure(data=[
        go.Bar(
            x=avg_cat["ProductCategory"],
            y=avg_cat["TotalSales"],
            marker=dict(
                color=["#00d4ff", "#00a3cc", "#007a99"],
                line=dict(color="#ffffff", width=1.5)
            ),
            text=avg_cat["TotalSales"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside"
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# OPERATIONS & DELIVERY ANALYTICS
# =====================================================
st.header("Operations & Delivery Analytics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    missing_delivery = df["MissingDelivery"].sum()
    st.metric("Missing Delivery Dates", f"{missing_delivery:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    urgent_orders = df["UrgentOrder"].sum()
    st.metric("Urgent Orders", f"{urgent_orders:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    avg_qty_east = df[df["Region"] == "East"]["Quantity_numeric"].mean()
    st.metric("Avg Quantity (East)", f"{avg_qty_east:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Delivery Days Distribution")
    delivery_data = df[df["DeliveryDays"].notna()]
    fig = go.Figure(data=[
        go.Histogram(
            x=delivery_data["DeliveryDays"],
            nbinsx=20,
            marker=dict(color="#00d4ff", line=dict(color="#ffffff", width=1)),
            opacity=0.8
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        xaxis=dict(title="Delivery Days", showgrid=True, gridcolor="#2d3348"),
        yaxis=dict(title="Frequency", showgrid=True, gridcolor="#2d3348")
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Order Status Distribution")
    status_counts = df["OrderStatus"].value_counts().reset_index()
    status_counts.columns = ["OrderStatus", "Count"]
    fig = go.Figure(data=[
        go.Pie(
            labels=status_counts["OrderStatus"],
            values=status_counts["Count"],
            hole=0.4,
            marker=dict(colors=["#4ade80", "#fbbf24", "#f87171", "#8b95a8"]),
            textinfo="label+percent",
            textfont=dict(size=14, color="#ffffff")
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# CUSTOMER ANALYTICS
# =====================================================
st.header("Customer Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Segmentation")
    seg_counts = df["CustomerSegment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig = go.Figure(data=[
        go.Bar(
            x=seg_counts["Segment"],
            y=seg_counts["Count"],
            marker=dict(
                color=["#4ade80", "#fbbf24", "#f87171"],
                line=dict(color="#ffffff", width=1.5)
            ),
            text=seg_counts["Count"],
            textposition="outside"
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue by Customer Segment")
    seg_rev = df.groupby("CustomerSegment")["TotalSales"].sum().reset_index()
    fig = go.Figure(data=[
        go.Pie(
            labels=seg_rev["CustomerSegment"],
            values=seg_rev["TotalSales"],
            hole=0.4,
            marker=dict(colors=["#4ade80", "#fbbf24", "#f87171"]),
            textinfo="label+value",
            textfont=dict(size=14, color="#ffffff")
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 10 Customers by Revenue")
top_customers = (
    df.groupby("CustomerName")["TotalSales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
fig = go.Figure(data=[
    go.Bar(
        y=top_customers["CustomerName"],
        x=top_customers["TotalSales"],
        orientation="h",
        marker=dict(
            color=top_customers["TotalSales"],
            colorscale="Teal",
            showscale=False,
            line=dict(color="#ffffff", width=1)
        ),
        text=top_customers["TotalSales"].apply(lambda x: f"${x:,.0f}"),
        textposition="outside"
    )
])
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1a1f2e",
    font=dict(color="#ffffff"),
    height=500,
    margin=dict(t=30, b=0, l=0, r=0),
    xaxis=dict(showgrid=True, gridcolor="#2d3348"),
    yaxis=dict(showgrid=False)
)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ADVANCED ANALYTICS
# =====================================================
st.header("Advanced Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Heatmap: Region vs Category")
    heatmap_data = (
        df.pivot_table(
            index="Region",
            columns="ProductCategory",
            values="TotalSales",
            aggfunc="sum"
        )
    )
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale="Teal",
        text=heatmap_data.values,
        texttemplate="%{text:,.0f}",
        textfont={"size": 12},
        colorbar=dict(title="Revenue")
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Sales vs Delivery Efficiency")
    scatter_data = df[df["DeliveryDays"].notna() & df["TotalSales"].notna()]
    fig = px.scatter(
        scatter_data,
        x="DeliveryDays",
        y="TotalSales",
        color="OrderStatus",
        color_discrete_map={
            "Delivered": "#4ade80",
            "Pending": "#fbbf24",
            "Returned": "#f87171"
        },
        opacity=0.6
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        xaxis=dict(showgrid=True, gridcolor="#2d3348"),
        yaxis=dict(showgrid=True, gridcolor="#2d3348")
    )
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Discount Impact Analysis")
    discount_data = df[df["Discount"] > 0]
    fig = px.scatter(
        discount_data,
        x="Discount",
        y="TotalSales",
        color="CustomerSegment",
        color_discrete_map={
            "High Value": "#4ade80",
            "Medium Value": "#fbbf24",
            "Low Value": "#f87171"
        },
        opacity=0.6
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        xaxis=dict(showgrid=True, gridcolor="#2d3348"),
        yaxis=dict(showgrid=True, gridcolor="#2d3348")
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Order Value Distribution")
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df["TotalSales"],
        marker=dict(color="#00d4ff"),
        boxmean="sd",
        name="Total Sales"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#ffffff"),
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="#2d3348")
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# REGIONAL INSIGHTS
# =====================================================
st.header("Regional Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    electronics_east = df[
        (df["ProductCategory"] == "Electronics") &
        (df["Region"] == "East")
    ]["TotalSales"].sum()
    st.metric("Electronics Revenue (East)", f"${electronics_east:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    west_delivered = ((df["OrderStatus"] == "Delivered") & (df["Region"] == "West")).sum()
    st.metric("Delivered Orders (West)", f"{west_delivered:,}")
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# DATA QUALITY & OUTLIERS
# =====================================================
st.header("Data Quality & Outliers")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    outlier_count = df["Outlier"].sum()
    st.metric("Outlier Orders Detected", f"{outlier_count:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    invalid_qty = (~df["Quantity_Valid"]).sum()
    st.metric("Invalid Quantity Records", f"{invalid_qty:,}")
    st.markdown('</div>', unsafe_allow_html=True)

if outlier_count > 0:
    st.subheader("Outlier Orders Details")
    outlier_df = df[df["Outlier"]][["OrderID", "CustomerName", "Region", "ProductCategory", "TotalSales"]].sort_values("TotalSales", ascending=False)
    st.dataframe(outlier_df, use_container_width=True, height=300)

# =====================================================
# TOP PERFORMERS
# =====================================================
st.header("Top Performers")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Orders by Revenue")
    top_orders = df.sort_values("TotalSales", ascending=False).head(10)[
        ["OrderID", "CustomerName", "Region", "ProductCategory", "TotalSales", "OrderDate"]
    ]
    st.dataframe(top_orders, use_container_width=True, height=400)

with col2:
    st.subheader("Product Categories")
    categories = df["ProductCategory"].unique()
    st.write(f"Total Categories: {len(categories)}")
    for cat in categories:
        cat_revenue = df[df["ProductCategory"] == cat]["TotalSales"].sum()
        cat_orders = len(df[df["ProductCategory"] == cat])
        st.markdown(f"""
        <div class="metric-container">
            <strong style="color: #00d4ff; font-size: 1.1rem;">{cat}</strong><br>
            <span style="color: #8b95a8;">Revenue: ${cat_revenue:,.2f}</span><br>
            <span style="color: #8b95a8;">Orders: {cat_orders:,}</span>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# DOWNLOAD SECTION
# =====================================================
st.header("Export Data")

st.markdown("""
<div class="info-message">
    Download the complete dataset with all calculated metrics, cleaned data, and derived features.
    This export includes all 24+ calculated fields for further analysis.
</div>
""", unsafe_allow_html=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Corrected CSV with All Metrics",
    data=csv,
    file_name=f"ecommerce_analytics_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #8b95a8; font-size: 0.875rem;">E-commerce Analytics Platform | Advanced Business Intelligence Dashboard</p>', unsafe_allow_html=True)
