import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# 1. Page Config & Styling (Unchanged)
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Simulator Pro")

st.markdown(
    """
    <style>
    .block-container { padding: 2rem 5rem; }
    section[data-testid="stSidebar"] > div:first-child { padding-left: 3.5rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #FFFFFF !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; font-weight: bold !important; color: #9ea4ad !important; }
    section[data-testid="stSidebar"] .stButton button {
        background-color: #2E7D32 !important; color: white !important; width: 100% !important; font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 2. Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Simulation Settings")
avg_demand = st.sidebar.number_input("Average Daily Demand", value=25.0, key="avg_val")
cov = st.sidebar.number_input("CoV", value=0.0, step=0.1, key="cov_val")
num_days = st.sidebar.slider("Horizon (Days)", 10, 1000, 100)
regen_button = st.sidebar.button("Reset Demand Scenario")

st.sidebar.divider()
st.sidebar.header("Policy & Costs")
opening_balance = st.sidebar.number_input("Opening Balance", value=300)
lead_time = int(st.sidebar.number_input("Lead Time (Days)", value=3))
reorder_point = st.sidebar.number_input("Reorder Point (ROP)", value=150)
order_qty = st.sidebar.number_input("Order Quantity (Q)", value=200)
unit_value = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_cost_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
ordering_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)

# ------------------------------------------------
# 3. Optimized Simulation Engine (Vectorized)
# ------------------------------------------------
@st.cache_data
def run_sim_fast(q_val, _demand_seq, _demand_dates, op_bal, lt, rop, u_val, h_pct, o_cost):
    n = len(_demand_seq)
    inv = np.zeros(n)
    pipeline_stock = np.zeros(n)
    # Use an array for arrivals to avoid list filtering (O(1) lookup)
    arrivals = np.zeros(n + lt + 1)
    orders = np.zeros(n)
    
    curr_inv = op_bal
    curr_pipe = 0
    daily_h_rate = (h_pct / 100) / 365

    for day in range(n):
        # 1. Process Arrivals
        recv = arrivals[day]
        curr_inv += recv
        curr_pipe -= recv
        
        # 2. Reorder Logic (Inventory Position)
        pos = curr_inv + curr_pipe
        placed = 0
        if pos <= rop:
            placed = q_val
            arrivals[day + lt] += placed
            curr_pipe += placed
        
        # 3. Fulfill Demand
        demand = _demand_seq[day]
        actual_demand_met = min(curr_inv, demand)
        curr_inv -= actual_demand_met
        
        # Record results
        inv[day] = curr_inv
        pipeline_stock[day] = curr_pipe
        orders[day] = placed

    # Vectorized post-processing (Much faster than row-by-row dicts)
    df_res = pd.DataFrame({
        "Date": _demand_dates,
        "Demand": _demand_seq,
        "Inventory": inv,
        "Pipeline": pipeline_stock,
        "Position": inv + pipeline_stock,
        "Order": orders
    })
    
    df_res["Shortage"] = np.maximum(0, df_res["Demand"] - (df_res["Inventory"].shift(1, fill_value=op_bal) + arrivals[:n]))
    # Note: Shortage logic simplified for speed, adjust if backordering is needed
    df_res["IsStockout"] = df_res["Inventory"] == 0
    df_res["InLT"] = df_res["Pipeline"] > 0
    df_res["HoldingCost"] = df_res["Inventory"] * u_val * daily_h_rate
    df_res["OrderingCost"] = np.where(df_res["Order"] > 0, o_cost, 0)
    
    return df_res

# ------------------------------------------------
# 4. State Management & Execution
# ------------------------------------------------
if "demand_seq" not in st.session_state or regen_button:
    if cov <= 0:
        st.session_state.demand_seq = np.full(num_days, float(avg_demand))
    else:
        st.session_state.demand_seq = np.maximum(0, np.random.normal(avg_demand, avg_demand * cov, num_days)).round()
    st.session_state.demand_dates = pd.date_range(start="2024-01-01", periods=num_days).date

# Execute Sim for Current Q and EOQ
df = run_sim_fast(order_qty, st.session_state.demand_seq, st.session_state.demand_dates, 
                  opening_balance, lead_time, reorder_point, unit_value, holding_cost_pct, ordering_cost)

annual_d = avg_demand * 365
annual_h = unit_value * (holding_cost_pct / 100)
eoq_val = np.sqrt((2 * annual_d * ordering_cost) / annual_h)
eoq_df = run_sim_fast(eoq_val, st.session_state.demand_seq, st.session_state.demand_dates, 
                      opening_balance, lead_time, reorder_point, unit_value, holding_cost_pct, ordering_cost)

# ------------------------------------------------
# 5. UI Layout (All Features Preserved)
# ------------------------------------------------
t1, t2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

with t1:
    st.write("### Inventory Dashboard Controls")
    use_pipeline = st.checkbox("Include Pipeline Inventory in KPIs", value=False)
    target_col = "Position" if use_pipeline else "Inventory"
    label_suffix = "(Incl. Pipeline)" if use_pipeline else "(Physical Only)"

    # KPIs
    st.subheader(f"Inventory Operational & Service KPIs {label_suffix}")
    global_fr = ((df["Demand"].sum() - df["Shortage"].sum()) / df["Demand"].sum() * 100) if df["Demand"].sum() > 0 else 100.0
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stockout Days", int(df["IsStockout"].sum()))
    k2.metric("Global Fill Rate", f"{global_fr:.1f}%")
    k3.metric("Avg Inv", f"{df[target_col].mean():.1f}") # Simplified for speed
    k4.metric("Min Inv", f"{df[target_col].min():.0f}")
    k5.metric("Avg WC", f"${(df[target_col].mean()*unit_value):,.0f}")

    # Financials (Expander preserved)
    with st.expander("💰 Financial Metrics & EOQ Comparison"):
        f1, f2, f3 = st.columns(3)
        total_cost_curr = df['HoldingCost'].sum() + df['OrderingCost'].sum()
        f1.metric("Total Policy Cost", f"${total_cost_curr:,.0f}")
        
        total_cost_eoq = eoq_df['HoldingCost'].sum() + eoq_df['OrderingCost'].sum()
        f2.metric("Savings using EOQ", f"${(total_cost_curr - total_cost_eoq):,.0f}")
        f3.metric("Cost Reduction %", f"{((total_cost_curr - total_cost_eoq)/total_cost_curr if total_cost_curr > 0 else 0):.1%}")

    # Charts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig.add_hline(y=reorder_point, line_dash="dash", line_color="red")
    
    # Event Markers
    orders_only = df[df["Order"] > 0]
    if not orders_only.empty:
        fig.add_trace(go.Scatter(x=orders_only["Date"], y=orders_only["Inventory"], mode="markers", name="Order Placed", marker=dict(color="#00FF00", size=8)))

    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulation Log")
    st.dataframe(df.head(100), use_container_width=True) # head(100) for faster initial browser render

with t2:
    st.title("Risk & Window Analysis")
    window_size = st.slider("Select Analysis Window (Days)", 1, 30, 7)
    
    # Block Volume
    df['Block_Group'] = np.arange(len(df)) // window_size
    block_df = df.groupby('Block_Group').agg({'Date': 'first', 'Demand': 'sum'}).reset_index()
    st.plotly_chart(px.bar(block_df, x='Date', y='Demand', template="plotly_dark"), use_container_width=True)

    # Risk Gap
    df['RollSum'] = df['Demand'].rolling(window=window_size).sum()
    hist_data = df['RollSum'].dropna()
    target_sl = st.slider("Target Service Level", 0.50, 0.99, 0.95, step=0.01)
    
    cutoff = np.percentile(hist_data, target_sl * 100)
    st.metric("Risk Exposure", f"{int(hist_data.max() - cutoff)} Units")
    
    fig_risk = px.histogram(hist_data, nbins=30, template="plotly_dark")
    fig_risk.add_vline(x=cutoff, line_dash="dash", line_color="yellow")
    st.plotly_chart(fig_risk, use_container_width=True)
