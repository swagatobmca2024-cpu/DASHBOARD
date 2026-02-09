import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="E-commerce Data Analytics Dashboard", layout="wide")

st.title("📊 E-commerce Advanced Data Analytics Dashboard")
st.caption("Upload raw order data → Clean → Analyze → Visualize")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV or Excel file to begin analysis.")
    st.stop()

# ===============================
# LOAD DATA
# ===============================
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.success("File uploaded successfully!")

# ===============================
# DATA CLEANING & FEATURE ENGINEERING
# ===============================

# ---- Date parsing
df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")

# ---- Discount cleaning (Task 5)
df["Discount"] = (
    df["DiscountText"]
    .astype(str)
    .str.replace("%", "")
    .str.strip()
)

df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce") / 100
df["Discount"].fillna(0, inplace=True)

# ---- Quantity cleaning & validation (Task 6)
df["Quantity_numeric"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Quantity_Valid"] = df["Quantity_numeric"].notna()

# ---- Unit price numeric
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

# ---- Total Sales (Task 1)
df["TotalSales"] = (
    df["Quantity_numeric"] *
    df["UnitPrice"] *
    (1 - df["Discount"])
)

# ---- Delivery Days (Task 2 & 19)
df["DeliveryDays"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days

# ---- Order Year & Month (Task 3 & 4)
df["OrderYear"] = df["OrderDate"].dt.year
df["OrderMonth"] = df["OrderDate"].dt.month

# ---- Month Name (Task 20)
df["MonthName"] = df["OrderDate"].dt.strftime("%B")

# ---- Missing delivery flag (Task 7)
df["MissingDelivery"] = df["DeliveryDate"].isna()

# ---- Urgent order flag (Task 8)
df["UrgentOrder"] = df["Notes"].astype(str).str.contains("urgent", case=False)

# ---- Customer Value Segmentation (Task 15)
def segment(x):
    if x > 5000:
        return "High Value"
    elif x > 2000:
        return "Medium Value"
    else:
        return "Low Value"

df["CustomerSegment"] = df["TotalSales"].apply(segment)

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("🔍 Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].dropna().unique(),
    default=df["Region"].dropna().unique()
)

category_filter = st.sidebar.multiselect(
    "Select Product Category",
    options=df["ProductCategory"].dropna().unique(),
    default=df["ProductCategory"].dropna().unique()
)

status_filter = st.sidebar.multiselect(
    "Select Order Status",
    options=df["OrderStatus"].dropna().unique(),
    default=df["OrderStatus"].dropna().unique()
)

filtered_df = df[
    (df["Region"].isin(region_filter)) &
    (df["ProductCategory"].isin(category_filter)) &
    (df["OrderStatus"].isin(status_filter))
]

# ===============================
# EXECUTIVE KPIs
# ===============================
st.subheader("📌 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Revenue", f"{filtered_df['TotalSales'].sum():,.2f}")
col2.metric("📦 Avg Order Value", f"{filtered_df['TotalSales'].mean():,.2f}")
col3.metric("🚚 Avg Delivery Days", f"{filtered_df['DeliveryDays'].mean():.1f}")
col4.metric("⚠️ Invalid Quantity Rows", f"{(~filtered_df['Quantity_Valid']).sum()}")

# ===============================
# SECTION 3 — BUSINESS ANALYTICS
# ===============================
st.subheader("📈 Sales & Business Performance")

# Task 9 — Revenue by Region
fig_region = px.bar(
    filtered_df.groupby("Region", as_index=False)["TotalSales"].sum(),
    x="Region", y="TotalSales",
    title="Total Revenue by Region"
)
st.plotly_chart(fig_region, use_container_width=True)

# Task 13 — Avg Order Value by Category
fig_cat = px.bar(
    filtered_df.groupby("ProductCategory", as_index=False)["TotalSales"].mean(),
    x="ProductCategory", y="TotalSales",
    title="Average Order Value by Category"
)
st.plotly_chart(fig_cat, use_container_width=True)

# ===============================
# SECTION — OPERATIONS ANALYTICS
# ===============================
st.subheader("🚚 Delivery & Operations")

fig_delivery = px.histogram(
    filtered_df,
    x="DeliveryDays",
    title="Delivery Duration Distribution"
)
st.plotly_chart(fig_delivery, use_container_width=True)

# ===============================
# SECTION — CUSTOMER SEGMENTATION
# ===============================
st.subheader("👥 Customer Segmentation")

fig_segment = px.pie(
    filtered_df,
    names="CustomerSegment",
    title="Customer Value Segments"
)
st.plotly_chart(fig_segment, use_container_width=True)

# ===============================
# SECTION — STATISTICAL ANALYSIS
# ===============================
st.subheader("📐 Statistical Insights")

mean_val = filtered_df["TotalSales"].mean()
median_val = filtered_df["TotalSales"].median()
std_val = filtered_df["TotalSales"].std()

col1, col2, col3 = st.columns(3)
col1.metric("Mean Sales", f"{mean_val:.2f}")
col2.metric("Median Sales", f"{median_val:.2f}")
col3.metric("Std Deviation", f"{std_val:.2f}")

# Task 24 — Outliers
filtered_df["Outlier"] = abs(filtered_df["TotalSales"] - median_val) > std_val

st.write("🚨 Orders with Strong Deviation")
st.dataframe(filtered_df[filtered_df["Outlier"]][
    ["OrderID", "CustomerName", "TotalSales"]
])

# ===============================
# DATA QUALITY SECTION
# ===============================
st.subheader("🧪 Data Quality Report")

st.write("❌ Invalid Quantity Rows")
st.dataframe(df[~df["Quantity_Valid"]])

st.write("📭 Missing Delivery Dates")
st.dataframe(df[df["MissingDelivery"]])
