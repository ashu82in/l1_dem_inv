import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ------------------------------------------------
# 1. Page Config & Custom Styling
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Simulator Pro")

st.markdown(
    """
    <style>
    .block-container {
        padding-left: 5rem;
        padding-right: 5rem;
        padding-top: 2rem;
    }
    
    /* ZOOM PADDING for Sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        padding-left: 2rem !important;
        padding-right: 1rem !important;
    }

    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    /* GREEN BUTTON STYLING */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #2E7D32 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        width: 100% !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 2. Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Simulation Settings")
avg_demand = st.sidebar.number_input("Average Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation (CoV)", value=0.1, step=0.1)
num_days = st.sidebar.slider("Simulation Horizon (Total Days)", 10, 1000, 100)
regen_button = st.sidebar.button("🔄 Regenerate Demand")

st.sidebar.divider()
st.sidebar.header("Analysis Window")
# NEW: Slider to choose the window size (e.g., 5 days, 7 days)
window_size = st.sidebar.slider("Rolling Window Size (Days)", 1, 30, 5)

st.sidebar.divider()
st.sidebar.header("Chart Settings")
fixed_zero = st.sidebar.checkbox("Start Y-Axis at Zero", value=True)

st.sidebar.divider()
st.sidebar.header("Policy & Costs")
opening_balance = st.sidebar.number_input("Opening Balance", value=500)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=3, min_value=0)
reorder_point = st.sidebar.number_input("Reorder Point (ROP)", value=200)
order_qty = st.sidebar.number_input("Order Quantity (Q)", value=300)
unit_value = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_cost_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
ordering_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)

# ------------------------------------------------
# 3. Demand Logic
# ------------------------------------------------
st.sidebar.divider()
data_source = st.sidebar.radio("Demand Source", ["Generated", "Excel Upload"])

if data_source == "Generated":
    demand_params = f"{avg_demand}_{cov}_{num_days}"
    if "last_params" not in st.session_state or st.session_state.last_params != demand_params or regen_button:
        st.session_state.last_params = demand_params
        if cov <= 0:
            st.session_state.demand_seq = np.full(num_days, float(avg_demand))
        else:
            std_dev = avg_demand * cov
            st.session_state.demand_seq = np.maximum(0, np.random.normal(avg_demand, std_dev, num_days)).round()
    
    final_demand = st.session_state.demand_seq
    final_dates = pd.date_range(start="2024-01-01", periods=num_days)
else:
    if "uploaded_demand" in st.session_state:
        final_demand = st.session_state.uploaded_demand
        final_dates = st.session_state.uploaded_dates
    else:
        final_demand = np.zeros(10)
        final_dates = pd.date_range(start="2024-01-01", periods=10)

# ------------------------------------------------
# 4. Simulation Engine
# ------------------------------------------------
def run_sim(q_val, d_seq, d_dates):
    inv = opening_balance
    pipeline = [] 
    rows = []
    for day in range(len(d_seq)):
        received = 0
        for order in pipeline[:]:
            if order[0] <= day:
                received += order[1]
                pipeline.remove(order)
        opening = inv
        inv += received
        daily_demand = d_seq[day]
        shortage = max(0, daily_demand - inv)
        inv = max(0, inv - daily_demand)
        inv_pos = inv + sum(o[1] for o in pipeline)
        placed_qty = 0
        if inv_pos < reorder_point:
            placed_qty = q_val
            if lead_time == 0: inv += placed_qty
            else: pipeline.append((day + lead_time, placed_qty))
        
        rows.append({
            "Date": d_dates[day], "Opening": int(opening), "Demand": int(daily_demand), 
            "Shortage": int(shortage), "Received": int(received), "Physical Inventory": int(inv), 
            "Inventory Position": int(inv + sum(o[1] for o in pipeline)), "New Order": int(placed_qty)
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty, final_demand, final_dates)

# ------------------------------------------------
# 5. Tabs
# ------------------------------------------------
tab1, tab2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

with tab1:
    st.title("Inventory Policy Simulator")
    h_rate = (holding_cost_pct / 100)
    total_cost = (df["Inventory Position"] * unit_value * h_rate / 365).sum() + ((df["New Order"] > 0).sum() * ordering_cost)
    fill_rate = ((df["Demand"].sum() - df["Shortage"].sum()) / df["Demand"].sum() * 100) if df["Demand"].sum() > 0 else 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Stockout Days", (df["Physical Inventory"] == 0).sum())
    m2.metric("Fill Rate", f"{fill_rate:.1f}%")
    m3.metric("Min Inventory", int(df["Physical Inventory"].min()))
    m4.metric("Avg Inventory", int(df["Physical Inventory"].mean()))
    m5.metric("Total Cost", f"${int(total_cost):,}")
    
    st.divider()

    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Physical Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig_inv.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")

    y_config = dict(rangemode="tozero", range=[0, df["Inventory Position"].max() * 1.1]) if fixed_zero else dict(rangemode="normal")
    fig_inv.update_layout(hovermode="x unified", template="plotly_dark", height=500, legend=dict(orientation="h", y=1.1), yaxis=y_config)
    st.plotly_chart(fig_inv, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    st.title("Demand & Window Analysis")
    
    # --- DATA UPLOAD SECTION ---
    if data_source == "Excel Upload":
        uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
        if uploaded_file:
            u_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            u_df['Date'] = pd.to_datetime(u_df['Date'])
            st.session_state.uploaded_demand = u_df['Demand'].values
            st.session_state.uploaded_dates = u_df['Date'].values
            st.rerun()

    # --- ROLLING WINDOW ANALYSIS ---
    st.subheader(f"Rolling {window_size}-Day Demand Analysis")
    
    # Calculate Rolling Metrics
    df['Rolling Mean'] = df['Demand'].rolling(window=window_size).mean()
    df['Rolling Sum'] = df['Demand'].rolling(window=window_size).sum()

    st.write(f"This graph shows the **Actual Daily Demand** vs the **{window_size}-Day Moving Average**.")
    
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=df["Date"], y=df["Demand"], name="Daily Demand", line=dict(color='rgba(171, 99, 250, 0.3)')))
    fig_rolling.add_trace(go.Scatter(x=df["Date"], y=df["Rolling Mean"], name=f"{window_size}-Day Rolling Avg", line=dict(color='#AB63FA', width=3)))
    
    fig_rolling.update_layout(template="plotly_dark", height=400, hovermode="x unified", yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
    st.plotly_chart(fig_rolling, use_container_width=True)

    # --- WINDOW-WISE TOTALS ---
    st.divider()
    st.subheader(f"Total Demand per {window_size}-Day Window")
    
    # Create chunks for window-wise bars
    df['Window_Group'] = np.arange(len(df)) // window_size
    window_totals = df.groupby('Window_Group').agg({'Demand': 'sum', 'Date': 'first'}).reset_index()

    fig_window_bar = px.bar(window_totals, x='Date', y='Demand', 
                             title=f"Total Demand in Each {window_size}-Day Block",
                             labels={'Demand': 'Total Window Demand'},
                             color_discrete_sequence=['#00CC96'])
    fig_window_bar.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_window_bar, use_container_width=True)

    st.divider()
    
    # --- DISTRIBUTION ANALYSIS ---
    st.subheader("Statistical Distribution")
    s1, s2, s3 = st.columns(3)
    s1.metric("Max Daily Demand", int(df["Demand"].max()))
    s2.metric("Avg Daily Demand", round(df["Demand"].mean(), 1))
    s3.metric("Actual CoV", round(df["Demand"].std() / df["Demand"].mean(), 2) if df["Demand"].mean() > 0 else 0)

    fig_hist = px.histogram(df, x="Demand", nbins=30, color_discrete_sequence=['#00CC96'], marginal="box")
    fig_hist.update_layout(template="plotly_dark", height=400, bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)
