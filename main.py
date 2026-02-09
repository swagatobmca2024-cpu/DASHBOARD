import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Advanced E-Commerce Data Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (HTML + CSS)
# -------------------------------------------------
st.markdown("""
<style>
body {background-color:#f4f6fa;}
.kpi-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    text-align: center;
}
.kpi-title {font-size: 14px; color: #6b7280;}
.kpi-value {font-size: 28px; font-weight: bold; color: #111827;}
hr {margin-top: 30px; margin-bottom: 30px;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("Ecommerce_Data_Analytics_Cleaner_Dataset.xlsx")

df = load_data()

# -------------------------------------------------
# DATA CLEANING & FEATURE ENGINEERING
# -------------------------------------------------

# Discount cleaning
df["Discount"] = (
    df["DiscountText"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
    .replace("nan", "0")
    .astype(float) / 100
)

# Quantity validation
df["Quantity_Num"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["InvalidQuantity"] = df["Quantity_Num"].isna()

# Dates
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")

# Delivery metrics
df["DeliveryDays"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days
df["MissingDelivery"] = df["DeliveryDate"].isna()

# Sales
df["TotalSales"] = (
    df["Quantity_Num"] *
    df["UnitPrice"] *
    (1 - df["Discount"])
)

# Date analytics
df["OrderYear"] = df["OrderDate"].dt.year
df["OrderMonth"] = df["OrderDate"].dt.month
df["MonthName"] = df["OrderDate"].dt.month_name()

# Urgent flag
df["UrgentOrder"] = df["Notes"].str.contains("urgent", case=False, na=False)

# Customer segmentation
def segment(val):
    if val > 5000:
        return "High Value"
    elif val > 2000:
        return "Medium Value"
    else:
        return "Low Value"

df["CustomerSegment"] = df["TotalSales"].apply(segment)

# Delivered dataset
delivered_df = df[df["OrderStatus"] == "Delivered"]

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------
total_revenue = df["TotalSales"].sum()
avg_order_value = df["TotalSales"].mean()
avg_delivery_days = delivered_df["DeliveryDays"].mean()
delivered_pct = (len(delivered_df) / len(df)) * 100

# -------------------------------------------------
# KPI DISPLAY
# -------------------------------------------------
st.title("📊 Advanced E-Commerce Analytics Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Revenue</div><div class='kpi-value'>₹{total_revenue:,.0f}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Avg Order Value</div><div class='kpi-value'>₹{avg_order_value:,.0f}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Avg Delivery Days</div><div class='kpi-value'>{avg_delivery_days:.1f}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Delivered %</div><div class='kpi-value'>{delivered_pct:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------------------------
# BUSINESS ANALYTICS VISUALS
# -------------------------------------------------
c1, c2 = st.columns(2)

region_rev = df.groupby("Region")["TotalSales"].sum().reset_index()
fig1 = px.bar(region_rev, x="Region", y="TotalSales", title="Revenue by Region")
c1.plotly_chart(fig1, use_container_width=True)

monthly_rev = df.groupby("MonthName")["TotalSales"].sum().reindex(
    ["January", "February", "March"]
).reset_index()
fig2 = px.line(monthly_rev, x="MonthName", y="TotalSales", markers=True,
               title="Monthly Revenue Trend")
c2.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# CATEGORY PERFORMANCE
# -------------------------------------------------
cat_avg = df.groupby("ProductCategory")["TotalSales"].mean().reset_index()
fig3 = px.bar(cat_avg, x="ProductCategory", y="TotalSales",
              title="Average Order Value by Category")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# SEGMENTATION
# -------------------------------------------------
seg_count = df["CustomerSegment"].value_counts().reset_index()
fig4 = px.pie(seg_count, names="index", values="CustomerSegment",
              title="Customer Value Segmentation")
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------
# DELIVERY PERFORMANCE (MATPLOTLIB)
# -------------------------------------------------
st.subheader("📦 Delivery Time Distribution")
fig, ax = plt.subplots()
ax.hist(delivered_df["DeliveryDays"].dropna(), bins=10)
ax.set_xlabel("Delivery Days")
ax.set_ylabel("Orders")
st.pyplot(fig)

# -------------------------------------------------
# STATISTICAL ANALYSIS
# -------------------------------------------------
mean_sales = df["TotalSales"].mean()
median_sales = df["TotalSales"].median()
std_sales = df["TotalSales"].std()

df["Outlier"] = abs(df["TotalSales"] - median_sales) > std_sales

st.markdown("### 📐 Statistical Summary")
st.write({
    "Mean Order Value": round(mean_sales, 2),
    "Median Order Value": round(median_sales, 2),
    "Standard Deviation": round(std_sales, 2),
    "Outlier Orders": int(df["Outlier"].sum())
})

# -------------------------------------------------
# DATA QUALITY REPORT
# -------------------------------------------------
st.markdown("### 🧹 Data Quality Issues")
dq1, dq2, dq3 = st.columns(3)
dq1.metric("Invalid Quantity Rows", df["InvalidQuantity"].sum())
dq2.metric("Missing Delivery Dates", df["MissingDelivery"].sum())
dq3.metric("Urgent Orders", df["UrgentOrder"].sum())

# -------------------------------------------------
# DOWNLOAD CLEAN REPORT
# -------------------------------------------------
st.markdown("### ⬇ Download Clean Analytics Report")

def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_df(df)

st.download_button(
    label="Download Full Analytics CSV",
    data=csv,
    file_name="Ecommerce_Analytics_Report.csv",
    mime="text/csv"
)
