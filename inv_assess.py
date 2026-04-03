import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ------------------------------------------------
# 1. Page Config & Custom Styling (Zoom-Ready)
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Simulator Pro")

st.markdown(
    """
    <style>
    .block-container { padding-left: 5rem; padding-right: 5rem; padding-top: 2rem; }
    section[data-testid="stSidebar"] > div:first-child {
        padding-left: 2.5rem !important; padding-right: 1.5rem !important;
    }
    .stMetric {
        background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;
    }
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
cov = st.sidebar.number_input("CoV", value=0.1, step=0.1, key="cov_val")
num_days = st.sidebar.slider("Horizon (Days)", 10, 1000, 100)
regen_button = st.sidebar.button("🔄 Regenerate Demand")

st.sidebar.divider()
window_size = st.sidebar.slider("Rolling Window Size (Days)", 1, 30, 7)

st.sidebar.divider()
st.sidebar.header("Policy & Costs")
opening_balance = st.sidebar.number_input("Opening Balance", value=1000)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=10)
reorder_point = st.sidebar.number_input("Reorder Point (ROP)", value=700)
order_qty = st.sidebar.number_input("Order Quantity (Q)", value=300)
unit_value = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_cost_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
ordering_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)

# ------------------------------------------------
# 3. Persistent Demand State
# ------------------------------------------------
if "demand_seq" not in st.session_state or regen_button:
    if cov <= 0:
        st.session_state.demand_seq = np.full(num_days, float(avg_demand))
    else:
        st.session_state.demand_seq = np.maximum(0, np.random.normal(avg_demand, avg_demand * cov, num_days)).round()
    st.session_state.demand_dates = pd.date_range(start="2024-01-01", periods=num_days)

# ------------------------------------------------
# 4. Simulation Engine
# ------------------------------------------------
def run_sim(q_val):
    inv = opening_balance
    pipeline = [] 
    rows = []
    d_seq = st.session_state.demand_seq
    d_dates = st.session_state.demand_dates
    daily_h_rate = (holding_cost_pct / 100) / 365

    for day in range(len(d_seq)):
        received = sum(o[1] for o in pipeline if o[0] == day)
        pipeline = [o for o in pipeline if o[0] != day]
        inv += received
        opening = inv
        demand = d_seq[day]
        
        # Position check (Physical + Pipeline) BEFORE satisfying demand
        pos_at_check = inv + sum(o[1] for o in pipeline)
        placed_qty = 0
        if pos_at_check <= reorder_point:
            placed_qty = q_val
            pipeline.append((day + lead_time, placed_qty))
            
        shortage = max(0, demand - inv)
        inv = max(0, inv - demand)
        
        rows.append({
            "Date": d_dates[day].date(), "Demand": demand, "Shortage": shortage, 
            "Inventory": inv, "Position": inv + sum(o[1] for o in pipeline), 
            "Order": placed_qty, "InLT": len(pipeline) > 0,
            "HoldingCost": inv * unit_value * daily_h_rate,
            "OrderingCost": ordering_cost if placed_qty > 0 else 0
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty)

# ------------------------------------------------
# 5. EOQ & Cost Comparison Math
# ------------------------------------------------
annual_demand = avg_demand * 365
annual_holding_cost_unit = unit_value * (holding_cost_pct / 100)
eoq_val = np.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost_unit)

# Cost Comparison for the simulation period
total_cost_current = df["HoldingCost"].sum() + df["OrderingCost"].sum()
# Estimate EOQ cost over the same period
eoq_df = run_sim(eoq_val)
total_cost_eoq = eoq_df["HoldingCost"].sum() + eoq_df["OrderingCost"].sum()
savings = total_cost_current - total_cost_eoq

# ------------------------------------------------
# 6. UI Layout
# ------------------------------------------------
t1, t2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

with t1:
    # --- COMPREHENSIVE COLLAPSIBLE KPI SECTION ---
    with st.expander("📊 Comprehensive Inventory KPIs", expanded=True):
        # Operational KPIs
        st.subheader("Inventory Operational KPIs")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stockout Days", (df["Shortage"] > 0).sum())
        lt_df = df[df["InLT"] == True]
        lt_fr = ((lt_df["Demand"].sum() - lt_df["Shortage"].sum()) / lt_df["Demand"].sum() * 100) if not lt_df.empty else 100.0
        k2.metric("Lead Time Fill Rate", f"{lt_fr:.1f}%")
        k3.metric("Average Inventory", f"{df['Inventory'].mean():.1f}")
        k4.metric("Avg Working Capital", f"${(df['Inventory'].mean() * unit_value):,.0f}")

        # Range KPIs
        st.divider()
        st.subheader("Inventory Range")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Minimum Inventory", f"{df['Inventory'].min()}")
        r2.metric("Maximum Inventory", f"{df['Inventory'].max()}")
        r3.metric("Min Working Capital", f"${(df['Inventory'].min() * unit_value):,.0f}")
        r4.metric("Max Working Capital", f"${(df['Inventory'].max() * unit_value):,.0f}")

        # Cost & EOQ KPIs
        st.divider()
        st.subheader("Inventory Cost Metrics & EOQ Comparison")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Holding Cost", f"${df['HoldingCost'].sum():,.0f}")
        c2.metric("Total Ordering Cost", f"${df['OrderingCost'].sum():,.0f}")
        c3.metric("Total Inventory Cost", f"${total_cost_current:,.0f}")
        
        e1, e2, e3 = st.columns(3)
        e1.metric("EOQ (Calculated)", f"{int(eoq_val)}")
        e2.metric("Cost with EOQ", f"${total_cost_eoq:,.0f}")
        e3.metric("Savings using EOQ", f"${savings:,.0f}", delta_color="normal")

    # --- MAIN CHART ---
    st.subheader("Inventory Levels Over Time")
    fig = go.Figure()
    for i, r in df.iterrows():
        if r["InLT"]: fig.add_vrect(x0=r["Date"], x1=r["Date"], fillcolor="red", opacity=0.05, layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    # Markers
    orders = df[df["Order"] > 0]
    if not orders.empty:
        fig.add_trace(go.Scatter(x=orders["Date"], y=orders["Inventory"], mode="markers", name="Order Triggered", 
                                 marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
    
    shorts = df[df["Shortage"] > 0]
    if not shorts.empty:
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["Inventory"], mode="markers", name="Shortage", 
                                 marker=dict(color="red", size=10, symbol="x")))

    fig.update_layout(template="plotly_dark", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Pipeline Graph
    st.subheader("Pipeline Inventory (Units in Transit)")
    st.plotly_chart(px.area(df, x="Date", y="Position", color_discrete_sequence=['#FFCC00']).update_layout(template="plotly_dark", height=250), use_container_width=True)

with t2:
    st.title("Demand Analytics")
    
    # Restored Histogram and Analytics
    c_hist, c_line = st.columns(2)
    with c_hist:
        st.subheader("Daily Demand Distribution")
        fig_dh = px.histogram(df, x="Demand", nbins=20, color_discrete_sequence=['#00CC96'], marginal="box")
        fig_dh.update_layout(template="plotly_dark", height=350, bargap=0.1)
        st.plotly_chart(fig_dh, use_container_width=True)
    
    with c_line:
        st.subheader("Daily Demand Volatility")
        fig_dl = px.line(df, x="Date", y="Demand", color_discrete_sequence=['#AB63FA'])
        fig_dl.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_dl, use_container_width=True)

    st.divider()
    
    # Window-based Analysis
    st.subheader(f"Service Level Analysis ({window_size}-Day Window)")
    df['RollSum'] = df['Demand'].rolling(window=window_size).sum()
    hist_data = df['RollSum'].dropna()
    target_sl = st.select_slider("Target Service Level", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
    cutoff = np.percentile(hist_data, target_sl * 100)
    
    r1, r2, r3 = st.columns(3)
    r1.metric(f"{int(target_sl*100)}% SL Threshold", int(cutoff))
    r2.metric("Max Observed", int(hist_data.max()))
    r3.metric("Exposure Gap", int(hist_data.max() - cutoff), delta="Risk", delta_color="inverse")
    
    fig_h_win = px.histogram(hist_data, nbins=25, color_discrete_sequence=['#00CC96'])
    fig_h_win.add_vline(x=cutoff, line_dash="dash", line_color="red", annotation_text="Target SL")
    fig_h_win.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_h_win, use_container_width=True)
