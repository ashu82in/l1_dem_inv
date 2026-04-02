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
    
    /* GREEN BUTTON STYLING (Emerald Shade) */
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
        shortage = max(0, daily_demand - inv)
        inv = max(0, inv - daily_demand)
        
        # Position check
        current_pipeline_qty = sum(o[1] for o in pipeline)
        inv_pos = inv + current_pipeline_qty
        
        placed_qty = 0
        if inv_pos < reorder_point:
            placed_qty = q_val
            if lead_time == 0: inv += placed_qty
            else: pipeline.append((day + lead_time, placed_qty))
        
        rows.append({
            "Date": dates[day].date(), 
            "Opening": int(opening), 
            "Demand": int(daily_demand), 
            "Shortage": int(shortage),
            "Received": int(received), 
            "Physical Inventory": int(inv), 
            "Inventory Position": int(inv + sum(o[1] for o in pipeline)), 
            "New Order": int(placed_qty)
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty)

# ------------------------------------------------
# 5. Tab Navigation
# ------------------------------------------------
tab1, tab2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

# ================================================
# TAB 1: INVENTORY SIMULATOR
# ================================================
with tab1:
    st.title("Inventory Policy Simulator")
    
    # KPIs
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

    # Inventory Graph
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Physical Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig_inv.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")

    # Markers
    stockouts = df[df["Physical Inventory"] == 0]
    if not stockouts.empty:
        fig_inv.add_trace(go.Scatter(x=stockouts["Date"], y=stockouts["Physical Inventory"], mode="markers", name="Stockout", marker=dict(color="red", size=10)))

    reorders = df[df["New Order"] > 0]
    if not reorders.empty:
        fig_inv.add_trace(go.Scatter(x=reorders["Date"], y=reorders["Physical Inventory"], mode="markers", name="Order Placed", marker=dict(color="#00FF00", size=10, symbol="triangle-up")))

    y_config = dict(rangemode="tozero", range=[0, df["Inventory Position"].max() * 1.1]) if fixed_zero else dict(rangemode="normal")
    fig_inv.update_layout(hovermode="x unified", template="plotly_dark", height=600, legend=dict(orientation="h", y=1.1), yaxis=y_config)
    st.plotly_chart(fig_inv, use_container_width=True)

    st.subheader("Detailed Simulation Log")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ================================================
# TAB 2: DEMAND ANALYZER
# ================================================
with tab2:
    st.title("Demand Statistics & Service Level Analysis")
    
    # Statistical Summary Row
    act_std = df["Demand"].std()
    act_mean = df["Demand"].mean()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Max Daily Demand", int(df["Demand"].max()))
    s2.metric("Avg Daily Demand", round(act_mean, 1))
    s3.metric("Std. Deviation (σ)", round(act_std, 2))
    s4.metric("CoV (Actual)", round(act_std / act_mean, 2) if act_mean > 0 else 0)

    st.divider()

    # --- SAFETY STOCK CALCULATOR ---
    st.subheader("Service Level & Variability Impact")
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.write("""
        **Why Variability Matters:**
        As demand variability (CoV) increases, the risk of a stockout during the Lead Time increases. 
        To maintain a specific **Service Level**, you must carry **Safety Stock**.
        """)
        
    with c_right:
        desired_sl = st.select_slider("Select Target Service Level", options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99], value=0.95)
        z_map = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.64, 0.98: 2.05, 0.99: 2.33}
        z = z_map[desired_sl]
        # SS = Z * σ * sqrt(LT)
        safety_stock = z * act_std * np.sqrt(lead_time)
        st.info(f"Recommended Safety Stock: **{int(safety_stock)} units**")

    st.divider()

    # Demand Distribution with Risk Zone
    service_cutoff = np.percentile(df["Demand"], desired_sl * 100)
    fig_hist = px.histogram(df, x="Demand", nbins=30, color_discrete_sequence=['#00CC96'], marginal="box", title="Demand Distribution & Risk Threshold")
    fig_hist.add_vline(x=service_cutoff, line_dash="dash", line_color="red", annotation_text=f"{int(desired_sl*100)}% SL")
    fig_hist.update_layout(template="plotly_dark", height=450, bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Demand Timeline
    st.subheader("Daily Demand Timeline")
    fig_dem_line = px.line(df, x="Date", y="Demand", color_discrete_sequence=['#AB63FA'])
    fig_dem_line.update_layout(template="plotly_dark", height=400, yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
    st.plotly_chart(fig_dem_line, use_container_width=True)
