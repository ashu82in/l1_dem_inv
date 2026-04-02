import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
        transition: 0.3s !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #388E3C !important;
        color: white !important;
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
regen_button = st.sidebar.button("🔄 Regenerate Demand")

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
# 3. Demand Generation Logic
# ------------------------------------------------
demand_params = f"{avg_demand}_{cov}_{num_days}"
if "last_params" not in st.session_state or st.session_state.last_params != demand_params or regen_button:
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
        inv_pos = inv + sum(o[1] for o in pipeline)
        placed_qty = 0
        if inv_pos < reorder_point:
            placed_qty = q_val
            if lead_time == 0: inv += placed_qty
            else: pipeline.append((day + lead_time, placed_qty))
        
        final_pipeline = sum(o[1] for o in pipeline)
        rows.append({
            "Date": dates[day].date(), 
            "Opening": int(opening), 
            "Demand": int(daily_demand), 
            "Received": int(received), 
            "Physical Inventory": int(inv), 
            "Pipeline": int(final_pipeline), 
            "Inventory Position": int(inv + final_pipeline), 
            "New Order": int(placed_qty)
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty)

# ------------------------------------------------
# 5. KPIs
# ------------------------------------------------
h_rate = (holding_cost_pct / 100)
total_cost = (df["Inventory Position"] * unit_value * h_rate / 365).sum() + ((df["New Order"] > 0).sum() * ordering_cost)

st.subheader("Inventory KPI Dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stockout Days", (df["Physical Inventory"] == 0).sum())
m2.metric("Min Inventory", int(df["Physical Inventory"].min()))
m3.metric("Avg Inventory", int(df["Physical Inventory"].mean()))
m4.metric("Total Cost", f"${int(total_cost):,}")
st.divider()

# ------------------------------------------------
# 6. Graphs
# ------------------------------------------------
st.subheader("Inventory Behaviour Over Time")
fig_inv = go.Figure()
fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Physical Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2)))
fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
fig_inv.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")

# Markers
stockouts = df[df["Physical Inventory"] == 0]
if not stockouts.empty:
    fig_inv.add_trace(go.Scatter(x=stockouts["Date"], y=stockouts["Physical Inventory"], mode="markers", name="Stockout", marker=dict(color="red", size=10)))

reorders = df[df["New Order"] > 0]
if not reorders.empty:
    fig_inv.add_trace(go.Scatter(x=reorders["Date"], y=reorders["Physical Inventory"], mode="markers", name="Order Placed", marker=dict(color="#00FF00", size=10, symbol="triangle-up")))

if fixed_zero:
    y_axis_config = dict(rangemode="tozero", range=[0, df["Inventory Position"].max() * 1.1])
else:
    y_axis_config = dict(rangemode="normal")

fig_inv.update_layout(hovermode="x unified", template="plotly_dark", height=500, legend=dict(orientation="h", y=1.1), yaxis=y_axis_config)
st.plotly_chart(fig_inv, use_container_width=True)

st.divider()
st.subheader("Daily Demand Timeline")
fig_dem = px.line(df, x="Date", y="Demand", color_discrete_sequence=['#AB63FA'])
fig_dem.update_layout(template="plotly_dark", height=400, yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
st.plotly_chart(fig_dem, use_container_width=True)

st.divider()
st.subheader("Demand Frequency (Histogram)")
st.plotly_chart(px.histogram(df, x="Demand", nbins=30, color_discrete_sequence=['#00CC96']).update_layout(template="plotly_dark", height=400, bargap=0.1), use_container_width=True)

st.divider()

# ------------------------------------------------
# 7. Data Table (Restored)
# ------------------------------------------------
st.subheader("Simulation Data Log")

# Allow user to download the data as CSV
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Simulation Data (CSV)",
    data=csv,
    file_name='inventory_sim_results.csv',
    mime='text/csv',
)

st.dataframe(df, use_container_width=True, hide_index=True)
