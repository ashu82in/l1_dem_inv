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
    .block-container { padding-left: 5rem; padding-right: 5rem; padding-top: 2rem; }
    section[data-testid="stSidebar"] > div:first-child {
        padding-left: 2.5rem !important; padding-right: 1.5rem !important;
    }
    .stMetric {
        background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: #2E7D32 !important; color: white !important; border: none !important;
        padding: 0.5rem 1rem !important; border-radius: 8px !important; width: 100% !important; font-weight: bold !important;
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
st.sidebar.header("Analysis Settings")
window_size = st.sidebar.slider("Rolling Window Size (Days)", 1, 30, 7)
fixed_zero = st.sidebar.checkbox("Start Y-Axis at Zero", value=True)

st.sidebar.divider()
st.sidebar.header("Policy & Costs")
opening_balance = st.sidebar.number_input("Opening Balance", value=1000)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=10)
reorder_point = st.sidebar.number_input("Reorder Point (ROP)", value=700)
order_qty = st.sidebar.number_input("Order Quantity (Q)", value=200)
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
# 4. Simulation Engine (The Brain)
# ------------------------------------------------
def run_sim():
    inv = opening_balance
    pipeline = [] 
    rows = []
    d_seq = st.session_state.demand_seq
    d_dates = st.session_state.demand_dates
    daily_h_rate = (holding_cost_pct / 100) / 365

    for day in range(len(d_seq)):
        # A. Start of day Arrivals
        received = sum(o[1] for o in pipeline if o[0] == day)
        pipeline = [o for o in pipeline if o[0] != day]
        inv += received
        
        opening = inv
        demand = d_seq[day]
        
        # B. TRIGGER CHECK (Physical <= ROP and No Order currently in lead time)
        is_already_ordered = len(pipeline) > 0
        placed_qty = 0
        order_expense = 0
        if inv <= reorder_point and not is_already_ordered:
            placed_qty = order_qty
            pipeline.append((day + lead_time, placed_qty))
            order_expense = ordering_cost
            
        # C. Satisfy Demand
        is_unfulfilled = demand > inv
        shortage = max(0, demand - inv)
        inv = max(0, inv - demand)
        
        # D. Financials
        h_cost = inv * unit_value * daily_h_rate
        final_pipeline_qty = sum(o[1] for o in pipeline)
        
        rows.append({
            "Date": d_dates[day].date(), "Demand": demand, "Shortage": shortage, 
            "Is Stockout": is_unfulfilled, "Inventory": inv, 
            "Pipeline Inventory": final_pipeline_qty,
            "Position": inv + final_pipeline_qty, 
            "Order": placed_qty, "InLT": final_pipeline_qty > 0,
            "Holding Cost": h_cost, "Ordering Cost": order_expense
        })
    return pd.DataFrame(rows)

df = run_sim()

# ------------------------------------------------
# 5. UI Layout
# ------------------------------------------------
t1, t2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

with t1:
    # COLLAPSABLE KPI SECTION
    with st.expander("🚀 Summary Performance Metrics", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        total_h = df["Holding Cost"].sum()
        total_o = df["Ordering Cost"].sum()
        m1.metric("Stockout Days", df["Is Stockout"].sum(), delta="Demand > Supply", delta_color="inverse")
        m2.metric("Fill Rate", f"{((df['Demand'].sum()-df['Shortage'].sum())/df['Demand'].sum()*100):.1f}%")
        m3.metric("Total Holding Cost", f"${int(total_h):,}")
        m4.metric("Total Ordering Cost", f"${int(total_o):,}")

    # Main Simulation Chart
    st.subheader("Inventory Levels & Reorder Position")
    fig = go.Figure()
    for i, r in df.iterrows():
        if r["InLT"]: fig.add_vrect(x0=r["Date"], x1=r["Date"], fillcolor="red", opacity=0.05, layer="below", line_width=0)
    
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    # Order markers
    orders = df[df["Order"] > 0]
    if not orders.empty:
        fig.add_trace(go.Scatter(x=orders["Date"], y=orders["Inventory"], mode="markers", name="Order Placed", 
                                 text=[f"Order Qty: {order_qty}" for _ in range(len(orders))],
                                 marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
    
    # Stockout markers
    shorts = df[df["Is Stockout"] == True]
    if not shorts.empty:
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["Inventory"], mode="markers", name="Shortage (Stockout)", 
                                 marker=dict(color="red", size=10, symbol="x")))

    fig.update_layout(template="plotly_dark", height=450, hovermode="x unified", yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
    st.plotly_chart(fig, use_container_width=True)

    # Pipeline Chart
    st.subheader("Total Pipeline Inventory (Units in Transit)")
    st.plotly_chart(px.area(df, x="Date", y="Pipeline Inventory", color_discrete_sequence=['#FFCC00']).update_layout(template="plotly_dark", height=250), use_container_width=True)

    # Financial Deep Dive
    st.divider()
    st.subheader("💰 Financial Analysis")
    c_pie, c_bar = st.columns([1, 2])
    with c_pie:
        cost_df = pd.DataFrame({"Category": ["Holding", "Ordering"], "Value": [total_h, total_o]})
        fig_pie = px.pie(cost_df, values='Value', names='Category', hole=0.4, color_discrete_sequence=['#00CC96', '#FF9900'])
        fig_pie.update_layout(template="plotly_dark", showlegend=False, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c_bar:
        df['Cumulative Cost'] = (df['Holding Cost'] + df['Ordering Cost']).cumsum()
        fig_cum = px.line(df, x="Date", y="Cumulative Cost", title="Cumulative Total Cost Profile")
        fig_cum.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_cum, use_container_width=True)

    st.subheader("Detailed Simulation Logs")
    st.dataframe(df, use_container_width=True, hide_index=True)

with t2:
    st.title("Demand & Risk Analysis")
    df['RollSum'] = df['Demand'].rolling(window=window_size).sum()
    target_sl = st.select_slider("Target Service Level", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
    hist_data = df['RollSum'].dropna()
    cutoff = np.percentile(hist_data, target_sl * 100)
    
    r1, r2, r3 = st.columns(3)
    r1.metric(f"{int(target_sl*100)}% SL Threshold", int(cutoff))
    r2.metric("Max Demand Observed", int(hist_data.max()))
    r3.metric("Uncovered Exposure", int(hist_data.max() - cutoff), delta="Units at Risk", delta_color="inverse")
    
    fig_h = px.histogram(hist_data, nbins=20, color_discrete_sequence=['#00CC96'], marginal="box")
    fig_h.add_vline(x=cutoff, line_dash="dash", line_color="red", annotation_text="Service Level Target")
    fig_h.update_layout(template="plotly_dark", height=450, xaxis_title=f"Units in {window_size}-Day Window")
    st.plotly_chart(fig_h, use_container_width=True)
