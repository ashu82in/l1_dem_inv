import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# 1. Page Config & Custom Side Padding
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Simulator Pro")

# CSS to prevent overlap with pen tablet/zoom tools
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
num_days = st.sidebar.slider("Simulation Days", 100, 1000, 365)

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

# New Metric: Minimum Inventory Level
min_inv_level = df["Physical Inventory"].min()

# EOQ Calculation
annual_d = avg_demand * 365
h_unit = unit_value * h_rate
eoq = int(np.sqrt((2 * annual_d * ordering_cost) / h_unit)) if h_unit > 0 else 0

# Savings logic
df_eoq = run_sim(eoq)
cost_eoq = (df_eoq["Inventory Position"] * unit_value * h_rate / 365).sum() + \
           (df_eoq["New Order"] > 0).sum() * ordering_cost

# ------------------------------------------------
# 6. Visualization & Dashboard
# ------------------------------------------------
st.subheader("Inventory KPI Dashboard")

# First row of metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stockout Days", (df["Physical Inventory"] == 0).sum())
m2.metric("Minimum Inventory", int(min_inv_level), 
          delta=int(min_inv_level - 0), delta_color="normal" if min_inv_level > 0 else "inverse")
m3.metric("Average Inventory", int(df["Physical Inventory"].mean()))
m4.metric("Total Cost", f"${int(total_cost):,}")

# Second row for EOQ
e1, e2, e3 = st.columns(3)
e1.metric("EOQ Recommendation", eoq)
e2.metric("Current Policy Cost", f"${int(total_cost):,}")
e3.metric("EOQ Potential Savings", f"${max(0, int(total_cost - cost_eoq)):,}")

st.divider()

st.subheader("Inventory Behaviour")
fig = go.Figure()
# Physical Stock line
fig.add_trace(go.Scatter(x=df["Date"], y=df["Physical Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
# Inventory Position line
fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
# Reorder Point horizontal line
fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="Reorder Point")

fig.update_layout(
    hovermode="x unified", 
    template="plotly_dark", 
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Detailed Simulation Log")
st.dataframe(df, use_container_width=True)
