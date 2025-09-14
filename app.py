

# import pandas as pd
# import streamlit as st
# import altair as alt

# # -----------------------------
# # Config
# # -----------------------------
# st.set_page_config(
#     page_title="Marketing Intelligence Dashboard",
#     layout="wide"
# )

# # Updated REQUIRED to match your actual column names
# REQUIRED = {
#     "facebook": ["date","tactic","state","campaign","impression","clicks","spend","attributed revenue"],
#     "google":   ["date","tactic","state","campaign","impression","clicks","spend","attributed revenue"],
#     "tiktok":   ["date","tactic","state","campaign","impression","clicks","spend","attributed revenue"],
#     "business": ["date","# of orders","# of new orders","new customers","total revenue","gross profit","COGS"]
# }

# def check_schema(df, required_cols, name):
#     """Check for missing columns and provide detailed feedback."""
#     missing = [c for c in required_cols if c not in df.columns]
#     present = [c for c in required_cols if c in df.columns]
    
#     if missing:
#         st.warning(f" {name} dataset missing columns: {missing}")
#         st.info(f" {name} dataset has columns: {present}")
#     else:
#         st.success(f" {name} dataset has all required columns")
    
#     return df

# def safe_divide(numerator, denominator, default=0):
#     """Safely divide two series, handling zeros and NaNs."""
#     return numerator / denominator.replace(0, 1).fillna(1)

# # -----------------------------
# # Load data with error handling
# # -----------------------------
# @st.cache_data
# def load_data():
#     try:
#         facebook = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/Facebook.csv")
#         google   = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/Google.csv")
#         tiktok   = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/TikTok.csv")
#         business = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/Business.csv")
        
#         st.success(" All CSV files loaded successfully")
#     except FileNotFoundError as e:
#         st.error(f" File not found: {e}")
#         st.stop()
#     except Exception as e:
#         st.error(f" Error loading files: {e}")
#         st.stop()

#     # Normalize column names to match expected format
#     # For marketing datasets: rename 'attributed revenue' to 'attributed_revenue'
#     for df in [facebook, google, tiktok]:
#         if 'attributed revenue' in df.columns:
#             df.rename(columns={'attributed revenue': 'attributed_revenue'}, inplace=True)
    
#     # For business dataset: normalize column names
#     if '# of orders' in business.columns:
#         business.rename(columns={'# of orders': 'orders'}, inplace=True)
#     if '# of new orders' in business.columns:
#         business.rename(columns={'# of new orders': 'new_orders'}, inplace=True)
#     if 'total revenue' in business.columns:
#         business.rename(columns={'total revenue': 'total_revenue'}, inplace=True)
#     if 'gross profit' in business.columns:
#         business.rename(columns={'gross profit': 'gross_profit'}, inplace=True)

#     # Updated requirements after normalization
#     REQUIRED_NORMALIZED = {
#         "facebook": ["date","tactic","state","campaign","impression","clicks","spend","attributed_revenue"],
#         "google":   ["date","tactic","state","campaign","impression","clicks","spend","attributed_revenue"],
#         "tiktok":   ["date","tactic","state","campaign","impression","clicks","spend","attributed_revenue"],
#         "business": ["date","orders","new_orders","new customers","total_revenue","gross_profit","COGS"]
#     }

#     # Check schemas with normalized names
#     facebook = check_schema(facebook, REQUIRED_NORMALIZED["facebook"], "Facebook")
#     google   = check_schema(google, REQUIRED_NORMALIZED["google"], "Google")
#     tiktok   = check_schema(tiktok, REQUIRED_NORMALIZED["tiktok"], "TikTok")
#     business = check_schema(business, REQUIRED_NORMALIZED["business"], "Business")

#     # Convert dates
#     for df, name in [(facebook, "Facebook"), (google, "Google"), (tiktok, "TikTok"), (business, "Business")]:
#         if "date" in df.columns:
#             try:
#                 df["date"] = pd.to_datetime(df["date"], errors="coerce")
#                 invalid_dates = df["date"].isna().sum()
#                 if invalid_dates > 0:
#                     st.warning(f" {name}: {invalid_dates} invalid dates converted to NaT")
#             except Exception as e:
#                 st.error(f" Error converting dates in {name}: {e}")

#     return facebook, google, tiktok, business

# # -----------------------------
# # Data Processing Functions
# # -----------------------------
# def create_marketing_dataset(facebook, google, tiktok):
#     """Combine marketing datasets with error handling."""
#     facebook["channel"] = "Facebook"
#     google["channel"] = "Google"
#     tiktok["channel"] = "TikTok"
    
#     # Only concatenate if dataframes have required columns
#     dfs_to_concat = []
#     for df, name in [(facebook, "Facebook"), (google, "Google"), (tiktok, "TikTok")]:
#         if "date" in df.columns and "campaign" in df.columns:
#             dfs_to_concat.append(df)
#         else:
#             st.warning(f" Skipping {name} - missing essential columns (date, campaign)")
    
#     if not dfs_to_concat:
#         st.error(" No marketing datasets have the minimum required columns")
#         return pd.DataFrame()
    
#     return pd.concat(dfs_to_concat, ignore_index=True)

# def aggregate_daily_data(marketing, business):
#     """Aggregate marketing and business data by date."""
#     daily_data = pd.DataFrame()
    
#     # Aggregate marketing data if available
#     if not marketing.empty and "date" in marketing.columns:
#         marketing_cols = ["impression", "clicks", "spend", "attributed_revenue"]
#         available_marketing_cols = {col: "sum" for col in marketing_cols if col in marketing.columns}
        
#         if available_marketing_cols:
#             daily_marketing = marketing.groupby("date", as_index=False).agg(available_marketing_cols)
#             daily_data = daily_marketing.copy()
    
#     # Merge with business data if available
#     if not business.empty and "date" in business.columns:
#         business_cols = ["orders", "new_orders", "new customers", "total_revenue", "gross_profit", "COGS"]
#         available_business_cols = [col for col in business_cols if col in business.columns]
        
#         if available_business_cols:
#             business_subset = business[["date"] + available_business_cols].copy()
            
#             if daily_data.empty:
#                 daily_data = business_subset
#             else:
#                 daily_data = pd.merge(daily_data, business_subset, on="date", how="outer")
    
#     # Fill NaN values with 0
#     daily_data = daily_data.fillna(0)
    
#     return daily_data

# def calculate_kpis(df):
#     """Calculate KPIs with available data."""
#     # Rolling averages
#     if "spend" in df.columns:
#         df["spend_ma7"] = df["spend"].rolling(7, min_periods=1).mean()
#     if "total_revenue" in df.columns:
#         df["revenue_ma7"] = df["total_revenue"].rolling(7, min_periods=1).mean()

#     # Performance metrics
#     if "clicks" in df.columns and "impression" in df.columns:
#         df["CTR"] = safe_divide(df["clicks"], df["impression"])
    
#     if "orders" in df.columns and "clicks" in df.columns:
#         df["Conversion Rate"] = safe_divide(df["orders"], df["clicks"])
    
#     if "spend" in df.columns and "new customers" in df.columns:
#         df["CAC"] = safe_divide(df["spend"], df["new customers"])
    
#     if "attributed_revenue" in df.columns and "spend" in df.columns:
#         df["ROI"] = safe_divide(df["attributed_revenue"], df["spend"])
    
#     if "gross_profit" in df.columns and "total_revenue" in df.columns:
#         df["Gross Margin %"] = safe_divide(df["gross_profit"], df["total_revenue"]) * 100

#     return df

# # -----------------------------
# # Main App
# # -----------------------------
# st.title(" Marketing Intelligence Dashboard")
# st.markdown("This dashboard connects marketing campaign activity with business outcomes.")

# # Load and process data
# with st.spinner("Loading data..."):
#     facebook, google, tiktok, business = load_data()

# st.markdown("---")

# # Create datasets
# marketing = create_marketing_dataset(facebook, google, tiktok)
# daily = aggregate_daily_data(marketing, business)

# if daily.empty:
#     st.error(" No data available to display. Please check your CSV files and column names.")
#     st.stop()

# # Calculate KPIs
# daily = calculate_kpis(daily)

# # Date filter
# if "date" in daily.columns and len(daily) > 0:
#     min_date = daily["date"].min()
#     max_date = daily["date"].max()
    
#     if pd.notna(min_date) and pd.notna(max_date):
#         start_date, end_date = st.date_input(
#             " Select Date Range", 
#             [min_date.date(), max_date.date()],
#             min_value=min_date.date(),
#             max_value=max_date.date()
#         )
        
#         mask = (daily["date"] >= pd.to_datetime(start_date)) & (daily["date"] <= pd.to_datetime(end_date))
#         df = daily[mask].copy()
#     else:
#         st.warning(" Invalid date range in data")
#         df = daily.copy()
# else:
#     st.warning(" No date column found or data is empty")
#     df = daily.copy()

# # -----------------------------
# # KPI Cards
# # -----------------------------
# st.subheader("Key Performance Indicators")

# if df.empty:
#     st.warning(" No data available for selected date range")
# else:
#     cols = st.columns(5)

#     def kpi_card(col, title, column_name, fmt="${:,.0f}", fallback="No Data"):
#         if column_name in df.columns and not df[column_name].isna().all():
#             value = df[column_name].sum() if "%" not in title else df[column_name].mean()
#             if pd.notna(value) and value != 0:
#                 col.metric(title, fmt.format(value))
#             else:
#                 col.metric(title, "0")
#         else:
#             col.metric(title, fallback)

#     kpi_card(cols[0], " Total Spend", "spend")
#     kpi_card(cols[1], " Attributed Revenue", "attributed_revenue")
#     kpi_card(cols[2], " New Customers", "new customers", fmt="{:,.0f}")
#     kpi_card(cols[3], " Avg ROI", "ROI", fmt="{:.2f}")
#     kpi_card(cols[4], " Gross Margin %", "Gross Margin %", fmt="{:.2f}%")

# st.markdown("---")

# # -----------------------------
# # Charts
# # -----------------------------
# st.subheader(" Trends & Insights")

# if df.empty or "date" not in df.columns:
#     st.info(" No data available for visualization")
# else:
#     tab1, tab2, tab3 = st.tabs(["💰 Spend vs Revenue", "🎯 Engagement Metrics", "👥 CAC Analysis"])

#     with tab1:
#         if "spend_ma7" in df.columns and "revenue_ma7" in df.columns:
#             st.subheader("Spend vs Revenue (7-day Moving Average)")
            
#             # Create base chart
#             base = alt.Chart(df).add_selection(
#                 alt.selection_interval()
#             )
            
#             spend_line = base.mark_line(color="orange", strokeWidth=3).encode(
#                 x=alt.X("date:T", title="Date"),
#                 y=alt.Y("spend_ma7:Q", title="7-Day MA Spend", scale=alt.Scale(zero=False)),
#                 tooltip=["date:T", alt.Tooltip("spend_ma7:Q", format="$,.0f", title="Spend MA")]
#             )
            
#             revenue_line = base.mark_line(color="green", strokeWidth=3).encode(
#                 x=alt.X("date:T", title="Date"),
#                 y=alt.Y("revenue_ma7:Q", title="7-Day MA Revenue", scale=alt.Scale(zero=False)),
#                 tooltip=["date:T", alt.Tooltip("revenue_ma7:Q", format="$,.0f", title="Revenue MA")]
#             )
            
#             # Layer charts with different y-axes
#             chart = alt.layer(spend_line, revenue_line).resolve_scale(y='independent')
#             st.altair_chart(chart, use_container_width=True)
#         else:
#             st.info(" Spend or Revenue data not available for trend analysis")

#     with tab2:
#         available_metrics = []
#         if "CTR" in df.columns:
#             available_metrics.append("CTR")
#         if "Conversion Rate" in df.columns:
#             available_metrics.append("Conversion Rate")
            
#         if available_metrics:
#             st.subheader("Click-Through Rate & Conversion Rate")
            
#             # Melt data for multiple metrics
#             melted = df[["date"] + available_metrics].melt(
#                 id_vars=["date"], 
#                 value_vars=available_metrics,
#                 var_name="Metric", 
#                 value_name="Value"
#             )
            
#             chart = alt.Chart(melted).mark_line(strokeWidth=2).encode(
#                 x=alt.X("date:T", title="Date"),
#                 y=alt.Y("Value:Q", title="Rate"),
#                 color=alt.Color("Metric:N", scale=alt.Scale(range=["#1f77b4", "#ff7f0e"])),
#                 tooltip=["date:T", "Metric:N", alt.Tooltip("Value:Q", format=".4f")]
#             ).interactive()
            
#             st.altair_chart(chart, use_container_width=True)
#         else:
#             st.info(" CTR or Conversion Rate data not available")

#     with tab3:
#         if "CAC" in df.columns and not df["CAC"].isna().all():
#             st.subheader("Customer Acquisition Cost Trend")
            
#             chart = alt.Chart(df).mark_line(color="red", strokeWidth=3).encode(
#                 x=alt.X("date:T", title="Date"),
#                 y=alt.Y("CAC:Q", title="Customer Acquisition Cost", scale=alt.Scale(zero=False)),
#                 tooltip=["date:T", alt.Tooltip("CAC:Q", format="$,.2f", title="CAC")]
#             ).interactive()
            
#             st.altair_chart(chart, use_container_width=True)
            
#             # Show average CAC
#             avg_cac = df["CAC"].mean()
#             if pd.notna(avg_cac):
#                 st.metric("Average CAC", f"${avg_cac:.2f}")
#         else:
#             st.info(" CAC data not available")

# st.markdown("---")

# # -----------------------------
# # Campaign Drilldown
# # -----------------------------
# st.subheader("🔍 Campaign Drilldown")

# if not marketing.empty and "channel" in marketing.columns:
#     channel = st.selectbox("Select Channel", ["All"] + sorted(marketing["channel"].unique().tolist()))
#     filtered = marketing if channel == "All" else marketing[marketing["channel"] == channel]
    
#     if "campaign" in filtered.columns:
#         campaigns = st.multiselect("Select Campaigns", sorted(filtered["campaign"].unique()))
#         if campaigns:
#             filtered = filtered[filtered["campaign"].isin(campaigns)]
    
#     if not filtered.empty:
#         # Display summary stats
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             if "spend" in filtered.columns:
#                 st.metric("Total Spend", f"${filtered['spend'].sum():,.2f}")
        
#         with col2:
#             if "clicks" in filtered.columns:
#                 st.metric("Total Clicks", f"{filtered['clicks'].sum():,}")
        
#         with col3:
#             if "impression" in filtered.columns:
#                 st.metric("Total Impressions", f"{filtered['impression'].sum():,}")
        
#         # Display filtered data
#         display_cols = [col for col in filtered.columns if col in ["date", "channel", "campaign", "state", "spend", "clicks", "impression", "attributed_revenue"]]
#         st.dataframe(filtered[display_cols].sort_values("date", ascending=False), use_container_width=True)
        
#         # Download button
#         csv = filtered.to_csv(index=False)
#         st.download_button(
#             label=" Download Campaign Data",
#             data=csv,
#             file_name="campaign_data.csv",
#             mime="text/csv"
#         )
#     else:
#         st.info(" No campaign data matches the selected filters")
# else:
#     st.info(" No marketing data available for campaign analysis")

# st.markdown("---")

# # -----------------------------
# # State-level Performance
# # -----------------------------
# st.subheader(" State-level Performance")

# if not marketing.empty and "state" in marketing.columns and "spend" in marketing.columns:
#     state_agg_dict = {"spend": "sum"}
#     if "attributed_revenue" in marketing.columns:
#         state_agg_dict["attributed_revenue"] = "sum"
#     if "clicks" in marketing.columns:
#         state_agg_dict["clicks"] = "sum"
    
#     state_agg = marketing.groupby("state", as_index=False).agg(state_agg_dict)
    
#     if not state_agg.empty:
#         # Sort by spend
#         state_agg = state_agg.sort_values("spend", ascending=False)
        
#         # Create bar chart
#         chart = alt.Chart(state_agg).mark_bar(color="steelblue").encode(
#             x=alt.X("state:N", title="State", sort="-y"),
#             y=alt.Y("spend:Q", title="Total Spend"),
#             tooltip=[
#                 "state:N", 
#                 alt.Tooltip("spend:Q", format="$,.0f", title="Spend")
#             ] + ([alt.Tooltip("attributed_revenue:Q", format="$,.0f", title="Revenue")] if "attributed_revenue" in state_agg.columns else [])
#         ).interactive()
        
#         st.altair_chart(chart, use_container_width=True)
        
#         # Display top performing states
#         st.subheader(" Top 5 States by Spend")
#         top_states = state_agg.head().copy()
#         st.dataframe(top_states, use_container_width=True, hide_index=True)
#     else:
#         st.info(" No state-level data available")
# else:
#     st.info(" State or spend data not available for geographic analysis")

# # -----------------------------
# # Data Summary
# # -----------------------------
# st.markdown("---")
# st.subheader(" Data Summary")

# col1, col2 = st.columns(2)

# with col1:
#     st.write("**Marketing Data:**")
#     if not marketing.empty:
#         st.write(f"- Total records: {len(marketing):,}")
#         st.write(f"- Date range: {marketing['date'].min().date()} to {marketing['date'].max().date()}")
#         st.write(f"- Channels: {', '.join(marketing['channel'].unique())}")
#     else:
#         st.write("- No marketing data available")

# with col2:
#     st.write("**Business Data:**")
#     if not business.empty:
#         st.write(f"- Total records: {len(business):,}")
#         if "date" in business.columns:
#             st.write(f"- Date range: {business['date'].min().date()} to {business['date'].max().date()}")
#         available_business_cols = [col for col in ["orders", "new_orders", "new customers", "total_revenue", "gross_profit", "COGS"] if col in business.columns]
#         st.write(f"- Available metrics: {', '.join(available_business_cols) if available_business_cols else 'None'}")
#     else:
#         st.write("- No business data available")

import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated REQUIRED to match actual column names
REQUIRED = {
    "facebook": ["date","tactic","state","campaign","impression","clicks","spend","attributed revenue"],
    "google":   ["date","tactic","state","campaign","impression","clicks","spend","attributed revenue"],
    "tiktok":   ["date","tactic","state","campaign","impression","clicks","spend","attributed revenue"],
    "business": ["date","# of orders","# of new orders","new customers","total revenue","gross profit","COGS"]
}

def check_schema(df, required_cols, name):
    """Check for missing columns and provide detailed feedback."""
    missing = [c for c in required_cols if c not in df.columns]
    present = [c for c in required_cols if c in df.columns]
    
    if missing:
        return f"{name}: Missing columns - {missing}"
    else:
        return f"{name}: Schema validated"

def safe_divide(numerator, denominator, default=0):
    """Safely divide two series or scalars, handling zeros and NaNs."""
    # Handle scalar values
    if np.isscalar(denominator):
        if denominator == 0 or pd.isna(denominator):
            return np.nan
        return numerator / denominator
    
    # Handle Series/DataFrame
    return numerator / denominator.replace(0, np.nan).fillna(np.nan)

# -----------------------------
# Load data with error handling
# -----------------------------
@st.cache_data
def load_data():
    try:
        facebook = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/Facebook.csv")
        google   = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/Google.csv")
        tiktok   = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/TikTok.csv")
        business = pd.read_csv("C:/Users/Lenovo/Downloads/Marketing_Documentation/Data/Business.csv")
        
        load_status = "All CSV files loaded successfully"
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    # Normalize column names to match expected format
    for df in [facebook, google, tiktok]:
        if 'attributed revenue' in df.columns:
            df.rename(columns={'attributed revenue': 'attributed_revenue'}, inplace=True)
    
    # For business dataset: normalize column names
    business_renames = {
        '# of orders': 'orders',
        '# of new orders': 'new_orders', 
        'total revenue': 'total_revenue',
        'gross profit': 'gross_profit'
    }
    business.rename(columns=business_renames, inplace=True)

    # Check schemas
    schema_status = []
    schema_status.append(check_schema(facebook, REQUIRED["facebook"], "Facebook"))
    schema_status.append(check_schema(google, REQUIRED["google"], "Google"))
    schema_status.append(check_schema(tiktok, REQUIRED["tiktok"], "TikTok"))
    schema_status.append(check_schema(business, REQUIRED["business"], "Business"))

    # Convert dates and add date features
    for df, name in [(facebook, "Facebook"), (google, "Google"), (tiktok, "TikTok"), (business, "Business")]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["week"] = df["date"].dt.isocalendar().week
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.day_name()
            df["weekday"] = df["date"].dt.weekday

    return facebook, google, tiktok, business, load_status, schema_status

# -----------------------------
# Enhanced Data Processing
# -----------------------------
def create_marketing_dataset(facebook, google, tiktok):
    """Combine marketing datasets with channel attribution."""
    facebook["channel"] = "Facebook"
    google["channel"] = "Google" 
    tiktok["channel"] = "TikTok"
    
    dfs_to_concat = [df for df in [facebook, google, tiktok] if "date" in df.columns]
    if not dfs_to_concat:
        return pd.DataFrame()
    
    marketing = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Calculate performance metrics
    marketing["ctr"] = safe_divide(marketing["clicks"], marketing["impression"]) * 100
    marketing["cpc"] = safe_divide(marketing["spend"], marketing["clicks"])
    marketing["roas"] = safe_divide(marketing["attributed_revenue"], marketing["spend"])
    
    return marketing

def aggregate_daily_data(marketing, business):
    """Create comprehensive daily dataset with business context."""
    daily_data = pd.DataFrame()
    
    # Aggregate marketing data
    if not marketing.empty:
        marketing_agg = marketing.groupby("date").agg({
            "impression": "sum",
            "clicks": "sum", 
            "spend": "sum",
            "attributed_revenue": "sum"
        }).reset_index()
        
        # Add channel mix metrics
        channel_spend = marketing.groupby(["date", "channel"])["spend"].sum().unstack(fill_value=0)
        channel_spend.columns = [f"{col}_spend" for col in channel_spend.columns]
        channel_spend = channel_spend.reset_index()
        
        marketing_agg = pd.merge(marketing_agg, channel_spend, on="date", how="left")
        daily_data = marketing_agg
    
    # Merge business data
    if not business.empty and not daily_data.empty:
        business_subset = business[["date", "orders", "new_orders", "new customers", "total_revenue", "gross_profit", "COGS"]].copy()
        daily_data = pd.merge(daily_data, business_subset, on="date", how="outer")
    elif not business.empty:
        daily_data = business.copy()
    
    daily_data = daily_data.fillna(0)
    return daily_data

def calculate_advanced_kpis(df):
    """Calculate advanced KPIs and business metrics."""
    # Basic performance
    if "clicks" in df.columns and "impression" in df.columns:
        df["CTR"] = safe_divide(df["clicks"], df["impression"]) * 100
    
    if "spend" in df.columns and "clicks" in df.columns:
        df["CPC"] = safe_divide(df["spend"], df["clicks"])
    
    if "attributed_revenue" in df.columns and "spend" in df.columns:
        df["ROAS"] = safe_divide(df["attributed_revenue"], df["spend"])
    
    # Business metrics
    if "spend" in df.columns and "new customers" in df.columns:
        df["CAC"] = safe_divide(df["spend"], df["new customers"])
    
    if "total_revenue" in df.columns and "new customers" in df.columns:
        df["AOV"] = safe_divide(df["total_revenue"], df["orders"]) if "orders" in df.columns else 0
    
    if "gross_profit" in df.columns and "total_revenue" in df.columns:
        df["Gross_Margin"] = safe_divide(df["gross_profit"], df["total_revenue"]) * 100
    
    # Marketing efficiency
    if "attributed_revenue" in df.columns and "total_revenue" in df.columns:
        df["Attribution_Rate"] = safe_divide(df["attributed_revenue"], df["total_revenue"]) * 100
    
    # Rolling metrics (7-day)
    numeric_cols = ["spend", "attributed_revenue", "total_revenue", "orders", "new customers"]
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean()
            df[f"{col}_growth"] = df[col].pct_change(periods=7) * 100
    
    return df

# -----------------------------
# Enhanced Visualizations
# -----------------------------
def create_performance_overview(df):
    """Create executive summary charts."""
    st.subheader("Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "spend_ma7" in df.columns and "attributed_revenue_ma7" in df.columns:
            chart_data = df[["date", "spend_ma7", "attributed_revenue_ma7"]].melt(
                id_vars=["date"], var_name="Metric", value_name="Value"
            )
            chart_data["Metric"] = chart_data["Metric"].map({
                "spend_ma7": "Marketing Spend", 
                "attributed_revenue_ma7": "Attributed Revenue"
            })
            
            chart = alt.Chart(chart_data).mark_line(strokeWidth=3).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Value:Q", title="Amount ($)", scale=alt.Scale(zero=False)),
                color=alt.Color("Metric:N", 
                    scale=alt.Scale(range=["#FF6B6B", "#4ECDC4"]),
                    legend=alt.Legend(title="Metric")
                ),
                tooltip=["date:T", "Metric:N", alt.Tooltip("Value:Q", format="$,.0f")]
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
    
    with col2:
        if "ROAS" in df.columns and df["ROAS"].notna().any():
            roas_chart = alt.Chart(df).mark_line(color="#FF9F43", strokeWidth=3).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("ROAS:Q", title="Return on Ad Spend", scale=alt.Scale(zero=False)),
                tooltip=["date:T", alt.Tooltip("ROAS:Q", format=".2f")]
            ).interactive()
            
            # Add benchmark line at ROAS = 3
            benchmark = alt.Chart(pd.DataFrame({"y": [3]})).mark_rule(
                color="red", strokeDash=[5, 5]
            ).encode(y="y:Q")
            
            combined = roas_chart + benchmark
            st.altair_chart(combined, use_container_width=True)

def create_channel_analysis(marketing):
    """Advanced channel performance analysis."""
    st.subheader("Channel Performance Deep Dive")
    
    if marketing.empty:
        st.info("No marketing data available")
        return
    
    # Channel summary metrics
    channel_summary = marketing.groupby("channel").agg({
        "spend": "sum",
        "attributed_revenue": "sum", 
        "clicks": "sum",
        "impression": "sum"
    }).reset_index()
    
    channel_summary["ROAS"] = safe_divide(channel_summary["attributed_revenue"], channel_summary["spend"])
    channel_summary["CTR"] = safe_divide(channel_summary["clicks"], channel_summary["impression"]) * 100
    channel_summary["CPC"] = safe_divide(channel_summary["spend"], channel_summary["clicks"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Spend distribution
        spend_chart = alt.Chart(channel_summary).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("spend:Q", scale=alt.Scale(type="sqrt")),
            color=alt.Color("channel:N", scale=alt.Scale(range=["#FF6B6B", "#4ECDC4", "#45B7D1"])),
            tooltip=["channel:N", alt.Tooltip("spend:Q", format="$,.0f")]
        )
        st.markdown("**Spend Distribution**")
        st.altair_chart(spend_chart, use_container_width=True)
    
    with col2:
        # ROAS comparison
        roas_chart = alt.Chart(channel_summary).mark_bar().encode(
            x=alt.X("channel:N", title="Channel"),
            y=alt.Y("ROAS:Q", title="ROAS"),
            color=alt.Color("channel:N", scale=alt.Scale(range=["#FF6B6B", "#4ECDC4", "#45B7D1"])),
            tooltip=["channel:N", alt.Tooltip("ROAS:Q", format=".2f")]
        )
        st.markdown("**ROAS by Channel**")
        st.altair_chart(roas_chart, use_container_width=True)
    
    with col3:
        # Channel metrics table
        st.markdown("**Channel Metrics Summary**")
        display_summary = channel_summary.copy()
        display_summary["spend"] = display_summary["spend"].apply(lambda x: f"${x:,.0f}")
        display_summary["attributed_revenue"] = display_summary["attributed_revenue"].apply(lambda x: f"${x:,.0f}")
        display_summary["ROAS"] = display_summary["ROAS"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_summary["CTR"] = display_summary["CTR"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        st.dataframe(display_summary, hide_index=True)

def create_business_correlation_analysis(df):
    """Analyze correlation between marketing and business metrics."""
    st.subheader("Marketing Impact on Business Outcomes")
    
    # Correlation heatmap data
    correlation_vars = ["spend", "attributed_revenue", "total_revenue", "orders", "new customers", "gross_profit"]
    available_vars = [var for var in correlation_vars if var in df.columns and df[var].notna().any()]
    
    if len(available_vars) < 2:
        st.info("Insufficient data for correlation analysis")
        return
    
    corr_data = df[available_vars].corr()
    
    # Convert correlation matrix to long format for Altair
    corr_long = corr_data.reset_index().melt(id_vars="index")
    corr_long.columns = ["var1", "var2", "correlation"]
    
    heatmap = alt.Chart(corr_long).mark_rect().encode(
        x=alt.X("var2:N", title=""),
        y=alt.Y("var1:N", title=""),
        color=alt.Color("correlation:Q", 
            scale=alt.Scale(domain=[-1, 1], range=["red", "white", "green"]),
            legend=alt.Legend(title="Correlation")
        ),
        tooltip=["var1:N", "var2:N", alt.Tooltip("correlation:Q", format=".3f")]
    ).properties(
        width=300,
        height=300,
        title="Marketing & Business Metrics Correlation"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.altair_chart(heatmap, use_container_width=True)
    
    with col2:
        st.markdown("**Key Insights:**")
        
        # Find strongest correlations
        if "spend" in available_vars and "total_revenue" in available_vars:
            spend_revenue_corr = df["spend"].corr(df["total_revenue"])
            st.metric("Marketing Spend ↔ Revenue", f"{spend_revenue_corr:.3f}")
        
        if "attributed_revenue" in available_vars and "total_revenue" in available_vars:
            attr_total_corr = df["attributed_revenue"].corr(df["total_revenue"])
            st.metric("Attributed ↔ Total Revenue", f"{attr_total_corr:.3f}")
        
        if "spend" in available_vars and "new customers" in available_vars:
            spend_customers_corr = df["spend"].corr(df["new customers"])
            st.metric("Marketing Spend ↔ New Customers", f"{spend_customers_corr:.3f}")

def create_actionable_insights(df, marketing):
    """Generate actionable business insights."""
    st.subheader("Actionable Insights & Recommendations")
    
    insights = []
    
    # Budget allocation insights
    if not marketing.empty and "channel" in marketing.columns:
        channel_roas = marketing.groupby("channel").apply(
            lambda x: safe_divide(x["attributed_revenue"].sum(), x["spend"].sum())
        ).to_dict()
        
        best_channel = max(channel_roas.items(), key=lambda x: x[1] if pd.notna(x[1]) else 0)
        worst_channel = min(channel_roas.items(), key=lambda x: x[1] if pd.notna(x[1]) else float('inf'))
        
        insights.append({
            "type": "success",
            "title": "Best Performing Channel",
            "message": f"{best_channel[0]} delivers the highest ROAS at {best_channel[1]:.2f}. Consider increasing budget allocation.",
            "action": f"Shift 15-20% more budget to {best_channel[0]}"
        })
        
        if worst_channel[1] < 2.0:
            insights.append({
                "type": "warning", 
                "title": "Underperforming Channel",
                "message": f"{worst_channel[0]} has low ROAS at {worst_channel[1]:.2f}. Review campaign strategy.",
                "action": f"Audit {worst_channel[0]} campaigns for optimization opportunities"
            })
    
    # Seasonality insights
    if "day_of_week" in df.columns and "orders" in df.columns:
        daily_orders = df.groupby("day_of_week")["orders"].mean().to_dict()
        best_day = max(daily_orders.items(), key=lambda x: x[1])
        
        insights.append({
            "type": "info",
            "title": "Peak Performance Day", 
            "message": f"{best_day[0]} shows highest average orders ({best_day[1]:.1f}). Optimize ad scheduling.",
            "action": f"Increase ad spend on {best_day[0]} by 25%"
        })
    
    # Display insights
    for insight in insights:
        if insight["type"] == "success":
            st.success(f"**{insight['title']}**\n\n{insight['message']}\n\n*Recommended Action:* {insight['action']}")
        elif insight["type"] == "warning":
            st.warning(f"**{insight['title']}**\n\n{insight['message']}\n\n*Recommended Action:* {insight['action']}")
        else:
            st.info(f"**{insight['title']}**\n\n{insight['message']}\n\n*Recommended Action:* {insight['action']}")

# -----------------------------
# Main App
# -----------------------------
st.title("Marketing Intelligence Dashboard")
st.markdown("**Connecting Marketing Performance with Business Outcomes**")

# Sidebar filters
st.sidebar.title("Dashboard Controls")

# Load and process data
with st.spinner("Loading and processing data..."):
    facebook, google, tiktok, business, load_status, schema_status = load_data()

# Display load status in sidebar with checkboxes
st.sidebar.subheader("Data Load Status")
st.sidebar.checkbox(load_status, value=True, disabled=True)

st.sidebar.subheader("Schema Validation")
for status in schema_status:
    if "Missing" in status:
        st.sidebar.checkbox(status, value=False, disabled=True)
    else:
        st.sidebar.checkbox(status, value=True, disabled=True)

marketing = create_marketing_dataset(facebook, google, tiktok)
daily = aggregate_daily_data(marketing, business)

if daily.empty:
    st.error("No data available to display. Please check your CSV files.")
    st.stop()

daily = calculate_advanced_kpis(daily)

# Date filter
if "date" in daily.columns and len(daily) > 0:
    min_date = daily["date"].min()
    max_date = daily["date"].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range", 
        [min_date.date(), max_date.date()],
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (daily["date"] >= pd.to_datetime(start_date)) & (daily["date"] <= pd.to_datetime(end_date))
        df = daily[mask].copy()
    else:
        df = daily.copy()
else:
    df = daily.copy()

# Executive KPIs
st.subheader("Executive Dashboard")
if not df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_spend = df["spend"].sum() if "spend" in df.columns else 0
        st.metric("Total Marketing Spend", f"${total_spend:,.0f}")
    
    with col2:
        total_revenue = df["total_revenue"].sum() if "total_revenue" in df.columns else 0
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col3:
        avg_roas = df["ROAS"].mean() if "ROAS" in df.columns and df["ROAS"].notna().any() else 0
        st.metric("Average ROAS", f"{avg_roas:.2f}")
    
    with col4:
        total_customers = df["new customers"].sum() if "new customers" in df.columns else 0
        st.metric("New Customers", f"{total_customers:,.0f}")
    
    with col5:
        avg_margin = df["Gross_Margin"].mean() if "Gross_Margin" in df.columns and df["Gross_Margin"].notna().any() else 0
        st.metric("Avg Gross Margin", f"{avg_margin:.1f}%")

st.markdown("---")

# Performance Overview
create_performance_overview(df)

st.markdown("---")

# Channel Analysis  
create_channel_analysis(marketing)

st.markdown("---")

# Business Correlation
create_business_correlation_analysis(df)

st.markdown("---")

# Actionable Insights
create_actionable_insights(df, marketing)

st.markdown("---")

# Campaign Drilldown (Enhanced)
st.subheader("Campaign Performance Drilldown")

if not marketing.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        channel_filter = st.selectbox("Select Channel", ["All"] + sorted(marketing["channel"].unique()))
    
    with col2:
        if "state" in marketing.columns:
            state_filter = st.selectbox("Select State", ["All"] + sorted(marketing["state"].unique()))
        else:
            state_filter = "All"
    
    # Filter data
    filtered_marketing = marketing.copy()
    if channel_filter != "All":
        filtered_marketing = filtered_marketing[filtered_marketing["channel"] == channel_filter]
    if state_filter != "All" and "state" in filtered_marketing.columns:
        filtered_marketing = filtered_marketing[filtered_marketing["state"] == state_filter]
    
    if not filtered_marketing.empty:
        # Campaign performance metrics
        campaign_metrics = filtered_marketing.groupby("campaign").agg({
            "spend": "sum",
            "attributed_revenue": "sum",
            "clicks": "sum", 
            "impression": "sum"
        }).reset_index()
        
        campaign_metrics["ROAS"] = safe_divide(campaign_metrics["attributed_revenue"], campaign_metrics["spend"])
        campaign_metrics["CTR"] = safe_divide(campaign_metrics["clicks"], campaign_metrics["impression"]) * 100
        
        # Sort by ROAS
        campaign_metrics = campaign_metrics.sort_values("ROAS", ascending=False, na_position="last")
        
        st.dataframe(
            campaign_metrics.style.format({
                "spend": "${:,.0f}",
                "attributed_revenue": "${:,.0f}",
                "clicks": "{:,.0f}",
                "impression": "{:,.0f}",
                "ROAS": "{:.2f}",
                "CTR": "{:.2f}%"
            }),
            use_container_width=True
        )

# Data export
st.sidebar.markdown("---")
st.sidebar.subheader("Export Data")

if st.sidebar.button("Download Daily Data"):
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"marketing_intelligence_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("*Dashboard built with Streamlit • Data updated daily • For business intelligence purposes*")