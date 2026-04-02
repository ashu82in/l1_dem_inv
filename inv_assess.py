import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# 1. Page Config & Custom Side Padding
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
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Inventory Policy Simulator")

# ------------------------------------------------
# 2. Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Simulation Settings")

avg_demand = st.sidebar.number_input("Average Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation (CoV)", value=0.1, step=0.1)
num_days = st.sidebar.slider("Simulation Days", 10, 1000, 100)

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
# 3. Reactive Demand Generation
# ------------------------------------------------
demand_params = f"{avg_demand}_{cov}_{num_days}"

if "last_params" not in st.session_state or st.session_state.last_params != demand_params:
    st.session_state.last_params = demand_params
    if cov <= 0:
        st.session_state.demand_seq = np.full(num_days, float(avg_demand))
    else:
        std_dev = avg_demand * cov
        st.session_state.demand_seq = np.maximum(0, np.random.normal(avg_demand, std_dev, num_days)).round()

demand = st.session_state.demand_seq
dates = pd.date_range(start="2024-01-01", periods=num_days)

# ------------------------------------------------
# 4. Simulation Engine
# ------------------------------------------------
def run_sim(q_val):
    inv = opening_balance
    pipeline = [] 
    rows = []

    for day in range(num_days):
        received = 0
        for order in pipeline[:]:
            if order[0] <= day:
                received += order[1]
                pipeline.remove(order)
        
        opening = inv
        inv += received
        daily_demand = demand[day]
        inv = max(0, inv - daily_demand)
        
        current_pipeline_qty = sum(o[1] for o in pipeline)
        inv_position = inv + current_pipeline_qty
        
        placed_qty = 0
        if inv_position < reorder_point:
            placed_qty = q_val
            if lead_time == 0:
                inv += placed_qty 
            else:
                pipeline.append((day + lead_time, placed_qty))
        
        final_pipeline = sum(o[1] for o in pipeline)
        
        rows.append({
            "Date": dates[day],
            "Opening": opening,
            "Demand": daily_demand,
            "Received": received,
            "Physical Inventory": inv,
            "Pipeline": final_pipeline,
            "Inventory Position": inv + final_pipeline,
            "New Order": placed_qty
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty)

# ------------------------------------------------
# 5. KPIs & Economics
# ------------------------------------------------
h_rate = (holding_cost_pct / 100)
holding_cost_total = (df["Inventory Position"] * unit_value * h_rate / 365).sum()
ordering_cost_total = (df["New Order"] > 0).sum() * ordering_cost
total_cost = holding_cost_total + ordering_cost_total
min_inv_level = df["Physical Inventory"].min()

# EOQ Calculation
annual_d = avg_demand * 365
h_unit = unit_value * h_rate
eoq = int(np.sqrt((2 * annual_d * ordering_cost) / h_unit)) if h_unit > 0 else 0

# Comparison for Savings
df_eoq = run_sim(eoq)
cost_eoq = (df_eoq["Inventory Position"] * unit_value * h_rate / 365).sum() + \
           (df_eoq["New Order"] > 0).sum() * ordering_cost

# ------------------------------------------------
# 6. Visualization & Dashboard
# ------------------------------------------------
st.subheader("Inventory KPI Dashboard")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stockout Days", (df["Physical Inventory"] == 0).sum())
m2.metric("Minimum Inventory", int(min_inv_level), 
          delta=int(min_inv_level), delta_color="normal" if min_inv_level > 0 else "inverse")
m3.metric("Average Inventory", int(df["Physical Inventory"].mean()))
m4.metric("Total Cost", f"${int(total_cost):,}")

st.divider()

# --- Main Inventory Chart ---
st.subheader("Inventory Behaviour")
fig_inv = go.Figure()

fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Physical Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2)))
fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot', width=1.5)))
fig_inv.add_hline(y=reorder_point, line_dash="dash", line_color="rgba(255, 0, 0, 0.5)", annotation_text="ROP")

# Markers
stockouts = df[df["Physical Inventory"] == 0]
if not stockouts.empty:
    fig_inv.add_trace(go.Scatter(x=stockouts["Date"], y=stockouts["Physical Inventory"], mode="markers", name="Stockout", marker=dict(color="red", size=10)))

reorders = df[df["New Order"] > 0]
if not reorders.empty:
    fig_inv.add_trace(go.Scatter(x=reorders["Date"], y=reorders["Physical Inventory"], mode="markers", name="Order Placed", marker=dict(color="#00FF00", size=10, symbol="triangle-up")))

fig_inv.update_layout(hovermode="x unified", template="plotly_dark", height=450, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig_inv, use_container_width=True)

# --- Demand Analysis Section ---
st.divider()
st.subheader("Demand Analysis")
col_plot, col_hist = st.columns(2)

with col_plot:
    st.write("**Daily Demand Timeline**")
    fig_dem_line = px.line(df, x="Date", y="Demand", color_discrete_sequence=['#AB63FA'])
    fig_dem_line.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_dem_line, use_container_width=True)

with col_hist:
    st.write("**Demand Frequency (Histogram)**")
    fig_dem_hist = px.histogram(df, x="Demand", nbins=30, color_discrete_sequence=['#00CC96'])
    fig_dem_hist.update_layout(template="plotly_dark", height=350, bargap=0.1)
    st.plotly_chart(fig_dem_hist, use_container_width=True)

st.subheader("Detailed Simulation Log")
st.dataframe(df, use_container_width=True)
