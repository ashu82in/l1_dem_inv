import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Config & Layout
st.set_page_config(layout="wide", page_title="Inventory Simulator")

# 2. CSS for Pen Tablet Padding
# Adjust padding-left/right to ensure your zoom tools don't overlap the UI
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 6rem;
        padding-right: 6rem;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Inventory Policy Simulator")

# ------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Inventory Inputs")

opening_balance = st.sidebar.number_input("Opening Balance", value=500)
avg_demand = st.sidebar.number_input("Average Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation", value=0.8)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=3, min_value=0)
reorder_point = st.sidebar.number_input("Reorder Point", value=200)
order_qty = st.sidebar.number_input("Order Quantity", value=300)
unit_value = st.sidebar.number_input("Value Per Unit", value=100)
holding_cost_percent = st.sidebar.number_input("Holding Cost (% of Value/Year)", value=20.0)
ordering_cost = st.sidebar.number_input("Ordering Cost Per Order", value=500)
num_days = st.sidebar.slider("Simulation Days", 100, 2000, 365)

holding_cost_rate = holding_cost_percent / 100

# Reset Logic
if "demand_sequence" not in st.session_state or st.sidebar.button("Reset Demand"):
    std_demand = avg_demand * cov
    st.session_state.demand_sequence = np.maximum(
        0, np.random.normal(avg_demand, std_demand, num_days)
    ).round()

demand = st.session_state.demand_sequence
dates = pd.date_range(start="2024-01-01", periods=num_days)

# ------------------------------------------------
# Inventory Simulation (Logic Fixed for Lead Time 0)
# ------------------------------------------------
def run_simulation(sim_order_qty):
    inv = opening_balance
    pipeline = [] # List of (arrival_day, quantity)
    sim_data = []

    for day in range(num_days):
        # 1. Process Arrivals
        shipment_received = 0
        for order in pipeline.copy():
            if order[0] <= day:
                shipment_received += order[1]
                pipeline.remove(order)

        opening = inv
        inv += shipment_received
        
        # 2. Daily Demand
        demand_today = demand[day]
        inv = max(0, inv - demand_today)

        # 3. Check Inventory Position (On Hand + In Pipeline)
        pipeline_qty = sum(q for d, q in pipeline)
        inv_position = inv + pipeline_qty

        # 4. Reorder Logic
        new_order_amt = 0
        if inv_position < reorder_point:
            new_order_amt = sim_order_qty
            if lead_time == 0:
                inv += new_order_amt # Instant arrival
            else:
                pipeline.append((day + lead_time, new_order_amt))
        
        # Re-calculate pipeline for data log
        current_pipeline_total = sum(q for d, q in pipeline)
        
        sim_data.append({
            "Date": dates[day],
            "Opening Balance": opening,
            "Demand": demand_today,
            "Shipment Received": shipment_received,
            "Pipeline Order": current_pipeline_total,
            "Inventory Position": inv + current_pipeline_total,
            "New Order": new_order_amt,
            "Closing Balance": inv,
            "Total Inventory": inv + current_pipeline_total
        })
    return pd.DataFrame(sim_data)

df = run_simulation(order_qty)

# ------------------------------------------------
# KPI Calculations
# ------------------------------------------------
stockout_days = (df["Closing Balance"] == 0).sum()
avg_inv = df["Total Inventory"].mean()
avg_age = avg_inv / df["Demand"].mean() if df["Demand"].mean() > 0 else 0
df["Blocked WC"] = df["Inventory Position"] * unit_value
avg_wc = df["Blocked WC"].mean()

# Costs
df["Holding Cost"] = (df["Total Inventory"] * unit_value * holding_cost_rate) / 365
total_holding = df["Holding Cost"].sum()
total_ordering = (df["New Order"] > 0).sum() * ordering_cost
total_cost = total_holding + total_ordering

# EOQ Calculation
annual_demand = avg_demand * 365
h_unit = unit_value * holding_cost_rate
eoq = np.sqrt((2 * annual_demand * ordering_cost) / h_unit) if h_unit > 0 else 0

# Comparison
df_eoq = run_simulation(int(eoq))
cost_eoq = ((df_eoq["Total Inventory"] * unit_value * holding_cost_rate) / 365).sum() + \
           ((df_eoq["New Order"] > 0).sum() * ordering_cost)

# ------------------------------------------------
# UI Layout
# ------------------------------------------------
st.subheader("Inventory Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stockout Days", stockout_days)
m2.metric("Avg Age (Days)", round(avg_age, 1))
m3.metric("Avg Inventory", int(avg_inv))
m4.metric("Avg Working Capital", f"${int(avg_wc):,}")

st.subheader("Cost & EOQ Analysis")
c1, c2, c3 = st.columns(3)
c1.metric("Total Cost (Current)", f"${int(total_cost):,}")
c2.metric("EOQ Value", int(eoq))
c3.metric("Potential Savings", f"${max(0, int(total_cost - cost_eoq)):,}")

# --- Main Chart ---
st.subheader("Inventory Behaviour Over Time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Physical Stock", line=dict(color='#1f77b4')))
fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#ff7f0e', dash='dot')))
fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")

# Markers
stockouts = df[df["Closing Balance"] == 0]
fig.add_trace(go.Scatter(x=stockouts["Date"], y=stockouts["Closing Balance"], mode="markers", name="Stockout", marker=dict(color="red", size=8)))

st.plotly_chart(fig, use_container_width=True)

# --- Bottom Charts ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Demand Distribution")
    st.plotly_chart(px.histogram(df, x="Demand", nbins=20), use_container_width=True)

with col_right:
    st.subheader("Working Capital Trend")
    st.plotly_chart(px.line(df, x="Date", y="Blocked WC"), use_container_width=True)

st.subheader("Simulation Log")
st.dataframe(df, use_container_width=True)
