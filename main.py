import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="E-commerce Data Analytics", layout="wide")
st.title("📊 E-commerce Data Analytics Dashboard")
st.caption("Junior Data Analyst | Management Reporting & Insights")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Orders Dataset (CSV / Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload a dataset to begin analysis.")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# =====================================================
# SECTION 1 — DATA CLEANING & FEATURE ENGINEERING
# =====================================================
st.header("🧹 Section 1: Data Cleaning & Preparation")

# Dates
df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")

# Task 5 — Discount cleaning
df["Discount"] = (
    df["DiscountText"]
    .astype(str)
    .str.replace("%", "")
    .str.strip()
)
df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce") / 100
df["Discount"].fillna(0, inplace=True)

# Task 6 — Quantity validation
df["Quantity_numeric"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Quantity_Valid"] = df["Quantity_numeric"].notna()

# Unit Price numeric
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

# Task 1 — Total Sales
df["TotalSales"] = (
    df["Quantity_numeric"] *
    df["UnitPrice"] *
    (1 - df["Discount"])
)

# Task 2 & 19 — Delivery Days
df["DeliveryDays"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days
df.loc[df["DeliveryDays"] < 0, "DeliveryDays"] = np.nan  # data quality fix

# Task 3 & 4 — Order Year & Month
df["OrderYear"] = df["OrderDate"].dt.year
df["OrderMonth"] = df["OrderDate"].dt.month

# Task 20 — Month Name
df["MonthName"] = df["OrderDate"].dt.strftime("%B")

# Task 7 — Missing Delivery Date
df["MissingDelivery"] = df["DeliveryDate"].isna()

# Task 8 — Urgent Orders
df["UrgentOrder"] = df["Notes"].astype(str).str.contains("urgent", case=False)

# Task 15 — Customer Segmentation
def segment(x):
    if x > 5000:
        return "High Value"
    elif x > 2000:
        return "Medium Value"
    else:
        return "Low Value"

df["CustomerSegment"] = df["TotalSales"].apply(segment)

st.success("Data cleaned and calculated successfully.")

# =====================================================
# SECTION 2 — EXECUTIVE METRICS
# =====================================================
st.header("📌 Executive Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"{df['TotalSales'].sum():,.2f}")
col2.metric("Average Order Value", f"{df['TotalSales'].mean():,.2f}")
col3.metric("Median Order Value", f"{df['TotalSales'].median():,.2f}")
col4.metric("Std Dev of Sales", f"{df['TotalSales'].std():,.2f}")

# =====================================================
# SECTION 3 — BUSINESS PERFORMANCE
# =====================================================
st.header("📈 Business Performance Analysis")

# Task 9 — Revenue by Region
rev_region = df.groupby("Region")["TotalSales"].sum().reset_index()
st.subheader("Total Revenue by Region")
st.plotly_chart(px.bar(rev_region, x="Region", y="TotalSales"), use_container_width=True)

# Task 10 — Electronics in East
electronics_east = df[
    (df["ProductCategory"] == "Electronics") &
    (df["Region"] == "East")
]["TotalSales"].sum()

st.metric("Electronics Revenue (East)", f"{electronics_east:,.2f}")

# Task 11 & 12 — Delivered Orders
st.metric("Delivered Orders (Total)", (df["OrderStatus"] == "Delivered").sum())
st.metric("Delivered Orders (West)", ((df["OrderStatus"] == "Delivered") & (df["Region"] == "West")).sum())

# Task 13 — Avg Order Value by Category
avg_cat = df.groupby("ProductCategory")["TotalSales"].mean().reset_index()
st.subheader("Average Order Value by Category")
st.plotly_chart(px.bar(avg_cat, x="ProductCategory", y="TotalSales"), use_container_width=True)

# Task 14 — Avg Quantity in East
avg_qty_east = df[df["Region"] == "East"]["Quantity_numeric"].mean()
st.metric("Avg Quantity Ordered (East)", f"{avg_qty_east:.2f}")

# =====================================================
# SECTION 4 — OPERATIONS & DELIVERY
# =====================================================
st.header("🚚 Operations & Delivery Efficiency")

valid_delivery = df["DeliveryDays"].dropna()
st.metric("Average Delivery Days", f"{valid_delivery.mean():.2f}")
st.metric("Orders with Missing Delivery Date", df["MissingDelivery"].sum())

# =====================================================
# SECTION 5 — CUSTOMER SEGMENTATION
# =====================================================
st.header("👥 Customer Segmentation")

seg_counts = df["CustomerSegment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]

st.plotly_chart(px.pie(seg_counts, names="Segment", values="Count"), use_container_width=True)

# Task 17 — Unique Product Categories
st.subheader("Unique Product Categories")
st.write(df["ProductCategory"].unique())

# Task 18 — Top Orders
st.subheader("Top Orders by Total Sales")
st.dataframe(df.sort_values("TotalSales", ascending=False).head(10))

# =====================================================
# SECTION 6 — STATISTICAL ANALYSIS & OUTLIERS
# =====================================================
st.header("📐 Statistical Analysis")

median = df["TotalSales"].median()
std = df["TotalSales"].std()

df["Outlier"] = abs(df["TotalSales"] - median) > std

st.metric("Outlier Orders Count", df["Outlier"].sum())
st.dataframe(df[df["Outlier"]][["OrderID", "CustomerName", "TotalSales"]])

# =====================================================
# DOWNLOAD CLEANED DATA
# =====================================================
st.header("⬇️ Download Cleaned & Calculated Dataset")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV with All 24 Calculations",
    csv,
    "ecommerce_analytics_cleaned.csv",
    "text/csv"
)
