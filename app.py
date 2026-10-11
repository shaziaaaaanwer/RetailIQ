from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io

st.set_page_config(
    page_title="RetailIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

[data-testid="stMetric"] {
    background: #f4f5fb;
    border: 1px solid #e3e5ef;
    padding: 14px;
    border-radius: 12px;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #222222 !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RetailIQ")
st.caption("E-commerce Customer Intelligence & RFM Analytics")
st.write(
    "Explore retail performance, identify customer groups, and export "
    "customer-level analysis from transaction data."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload Online Retail.xlsx",
        type=["xlsx", "xls"],
        help="Use the original UCI Online Retail Excel dataset.",
    )
    st.divider()
    st.caption("Pipeline: cleaning → RFM features → scaling → K-Means")

if uploaded is None:
    st.info("Upload **Online Retail.xlsx** in the sidebar to start.")
    st.subheader("What RetailIQ provides")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Business analytics**\n\nRevenue, orders, products, and monthly trends.")
    c2.markdown("**Customer segmentation**\n\nRFM features and K-Means clustering.")
    c3.markdown("**Exports**\n\nDownload cleaned transactions and customer segments.")
    st.stop()


@st.cache_data(show_spinner="Reading and cleaning transactions...")
def load_and_clean(file_bytes):
    data = pd.read_excel(io.BytesIO(file_bytes))
    required = {
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_count = len(data)
    data = data.drop_duplicates()
    data = data.dropna(subset=["CustomerID"]).copy()
    data = data[
        ~data["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
    ].copy()
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)].copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"]).copy()
    data["CustomerID"] = data["CustomerID"].astype(float).astype("int64").astype(str)
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    return data, raw_count


@st.cache_data(show_spinner="Calculating RFM features and customer segments...")
def build_rfm_and_clusters(clean_data):
    analysis_date = clean_data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = (
        clean_data.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (analysis_date - x.max().normalize()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    if len(rfm) < 4:
        raise ValueError("At least four unique customers are needed for segmentation.")

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    candidate_scores = {}
    candidate_models = {}
    for k in range(2, min(5, len(rfm) - 1) + 1):
        candidate = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = candidate.fit_predict(scaled)
        if len(set(labels)) > 1:
            candidate_scores[k] = silhouette_score(scaled, labels)
            candidate_models[k] = candidate

    if not candidate_scores:
        raise ValueError("Could not evaluate a valid clustering configuration.")

    best_k = max(candidate_scores, key=candidate_scores.get)
    model = candidate_models[best_k]
    rfm["Cluster"] = model.labels_

    scores_df = pd.DataFrame({
        "Clusters (k)": list(candidate_scores.keys()),
        "Silhouette score": list(candidate_scores.values()),
    }).sort_values("Clusters (k)")

    return rfm, best_k, candidate_scores, scores_df, analysis_date


try:
    df, raw_rows = load_and_clean(uploaded.getvalue())
    if df.empty:
        st.error("No valid transactions remain after cleaning.")
        st.stop()
    rfm, best_k, scores, score_df, analysis_date = build_rfm_and_clusters(df)
except Exception as exc:
    st.error(f"Could not process this file: {exc}")
    st.stop()

with st.sidebar:
    st.success(f"{len(df):,} cleaned transaction rows")
    all_clusters = sorted(rfm["Cluster"].unique().tolist())
    chosen_clusters = st.multiselect(
        "Customer segments",
        options=all_clusters,
        default=all_clusters,
        format_func=lambda cluster: f"Segment {cluster + 1}",
    )

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()
with st.sidebar:
    date_range = st.date_input(
        "Transaction date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    date_start, date_end = date_range
else:
    date_start, date_end = min_date, max_date

chosen_customer_ids = rfm.loc[rfm["Cluster"].isin(chosen_clusters), "CustomerID"]
filtered_tx = df[
    df["CustomerID"].isin(chosen_customer_ids)
    & (df["InvoiceDate"].dt.date >= date_start)
    & (df["InvoiceDate"].dt.date <= date_end)
].copy()
filtered_rfm = rfm[rfm["Cluster"].isin(chosen_clusters)].copy()

total_revenue = filtered_tx["Revenue"].sum()
order_count = filtered_tx["InvoiceNo"].nunique()
customer_count = filtered_tx["CustomerID"].nunique()
avg_order_value = total_revenue / order_count if order_count else 0

tab_overview, tab_segments, tab_customers, tab_exports = st.tabs(
    ["Overview", "Segmentation", "Customer Explorer", "Data & Export"]
)

with tab_overview:
    st.subheader("Business Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", f"£{total_revenue:,.2f}")
    k2.metric("Orders", f"{order_count:,}")
    k3.metric("Customers", f"{customer_count:,}")
    k4.metric("Average order value", f"£{avg_order_value:,.2f}")

    left, right = st.columns(2)
    with left:
        if not filtered_tx.empty:
            monthly = (
                filtered_tx.set_index("InvoiceDate")
                .resample("MS")["Revenue"].sum().reset_index()
            )
            fig = px.line(
                monthly, x="InvoiceDate", y="Revenue", markers=True,
                title="Monthly Revenue Trend",
                labels={"InvoiceDate": "Month", "Revenue": "Revenue (£)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transactions match the selected filters.")

    with right:
        top_products = (
            filtered_tx.dropna(subset=["Description"])
            .groupby("Description")["Revenue"].sum()
            .nlargest(10).sort_values().reset_index()
        )
        if not top_products.empty:
            fig = px.bar(
                top_products, x="Revenue", y="Description", orientation="h",
                title="Top 10 Products by Revenue",
                labels={"Revenue": "Revenue (£)", "Description": "Product"},
            )
            st.plotly_chart(fig, use_container_width=True)

    country = (
        filtered_tx.groupby("Country")["Revenue"].sum()
        .nlargest(10).reset_index()
    )
    if not country.empty:
        fig = px.bar(country, x="Country", y="Revenue", title="Top 10 Countries by Revenue")
        st.plotly_chart(fig, use_container_width=True)

with tab_segments:
    st.subheader("Customer Segmentation")
    st.write(
        f"Selected **k = {best_k}** because it had the highest silhouette score "
        f"among the tested cluster counts ({min(scores)}–{max(scores)})."
    )
    st.metric("Best silhouette score", f"{scores[best_k]:.3f}")

    profiles = (
        rfm.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency_Days=("Recency", "mean"),
            Avg_Invoices=("Frequency", "mean"),
            Avg_Spend_GBP=("Monetary", "mean"),
            Total_Spend_GBP=("Monetary", "sum"),
        )
        .reset_index()
    )
    profiles["Customer Share (%)"] = (profiles["Customers"] / len(rfm) * 100).round(2)
    st.dataframe(profiles.round(2), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            filtered_rfm, x="Frequency", y="Monetary", color="Cluster",
            hover_data=["CustomerID", "Recency"],
            title="Purchase Frequency vs Total Spend",
            labels={"Frequency": "Distinct invoices", "Monetary": "Total spend (£)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            profiles, x="Cluster", y="Total_Spend_GBP", color="Cluster",
            title="Total Spend by Customer Segment",
            labels={"Total_Spend_GBP": "Total spend (£)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(
        px.bar(score_df, x="Clusters (k)", y="Silhouette score",
               title="Silhouette Score by Number of Clusters"),
        use_container_width=True,
    )
    st.caption(
        "Cluster IDs are arbitrary. Interpret them from their RFM profiles; "
        "silhouette score is not classification accuracy."
    )

with tab_customers:
    st.subheader("Customer Explorer")
    if filtered_rfm.empty:
        st.info("Choose at least one segment in the sidebar.")
    else:
        st.dataframe(
            filtered_rfm.sort_values("Monetary", ascending=False).round(2),
            use_container_width=True,
            hide_index=True,
        )
        customer_id = st.selectbox(
            "Inspect a customer",
            filtered_rfm["CustomerID"].tolist(),
        )
        customer_transactions = df[df["CustomerID"] == customer_id].sort_values(
            "InvoiceDate", ascending=False
        )
        st.write(f"Transactions for customer **{customer_id}**")
        st.dataframe(customer_transactions, use_container_width=True, hide_index=True)

with tab_exports:
    st.subheader("Data Quality & Downloads")
    st.write(f"Original rows: **{raw_rows:,}**")
    st.write(f"Cleaned rows: **{len(df):,}**")
    st.write(f"Rows removed: **{raw_rows - len(df):,}**")
    st.write(f"Analysis date for recency: **{analysis_date.date()}**")
    st.download_button(
        "Download customer segments CSV",
        data=rfm.to_csv(index=False).encode("utf-8"),
        file_name="retailiq_customer_segments.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download segment profiles CSV",
        data=profiles.to_csv(index=False).encode("utf-8"),
        file_name="retailiq_segment_profiles.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cleaned transactions CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="retailiq_clean_transactions.csv",
        mime="text/csv",
    )

st.divider()
st.caption("RetailIQ • Educational analytics prototype • Results depend on the uploaded dataset and cleaning rules.")
