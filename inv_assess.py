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
st.sidebar.header("Analysis Window")
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
# 4. Simulation Engine (The Professional Logic)
# ------------------------------------------------
def run_sim():
    inv = opening_balance
    pipeline = [] 
    rows = []
    d_seq = st.session_state.demand_seq
    d_dates = st.session_state.demand_dates

    for day in range(len(d_seq)):
        # A. Start of day: Goods arrival
        received = sum(o[1] for o in pipeline if o[0] == day)
        pipeline = [o for o in pipeline if o[0] != day]
        inv += received
        
        opening = inv
        demand = d_seq[day]
        
        # B. Position check (Physical + Pipeline) BEFORE demand
        # This determines the decision to trigger a new order
        current_pipeline_qty = sum(o[1] for o in pipeline)
        pos_at_check = inv + current_pipeline_qty
        
        # C. TRIGGER: If Position <= ROP
        placed_qty = 0
        if pos_at_check <= reorder_point:
            placed_qty = order_qty
            pipeline.append((day + lead_time, placed_qty))
            
        # D. Satisfy Demand
        shortage = max(0, demand - inv)
        inv = max(0, inv - demand)
        
        rows.append({
            "Date": d_dates[day].date(), "Opening": opening, "Demand": demand, 
            "Shortage": shortage, "Inventory": inv, 
            "Position": inv + sum(o[1] for o in pipeline), 
            "Order": placed_qty, "InLT": len(pipeline) > 0
        })
    return pd.DataFrame(rows)

df = run_sim()

# ------------------------------------------------
# 5. UI Layout
# ------------------------------------------------
t1, t2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

with t1:
    # 1. KPI Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Stockout Days", (df["Shortage"] > 0).sum())
    m2.metric("Fill Rate", f"{((df['Demand'].sum()-df['Shortage'].sum())/df['Demand'].sum()*100):.1f}%")
    m3.metric("Min Stock", int(df["Inventory"].min()))
    m4.metric("Avg Stock", int(df["Inventory"].mean()))
    m5.metric("Total Cost", f"${int((df['Position'].mean()*unit_value*holding_cost_pct/100) + ((df['Order']>0).sum()*ordering_cost)):,}")

    # 2. Main Inventory Chart
    fig = go.Figure()
    # Shadow for Lead Time periods
    for i, r in df.iterrows():
        if r["InLT"]: fig.add_vrect(x0=r["Date"], x1=r["Date"], fillcolor="red", opacity=0.05, layer="below", line_width=0)
    
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    # Order markers with correct Tooltip Qty
    orders = df[df["Order"] > 0]
    if not orders.empty:
        fig.add_trace(go.Scatter(x=orders["Date"], y=orders["Inventory"], mode="markers", name="Order Triggered", 
                                 text=[f"Order Qty: {order_qty}<br>Total Position: {p}" for p in orders["Position"]],
                                 hovertemplate="<b>%{x}</b><br>%{text}<br>Physical Stock: %{y}<extra></extra>",
                                 marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
    
    fig.update_layout(template="plotly_dark", height=500, hovermode="x unified", yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. 1-Day Contextual Demand Charts (Bottom of Tab 1)
    st.divider()
    st.subheader("Daily Demand Overview (1-Day Resolution)")
    c_l, c_r = st.columns(2)
    with c_l: st.plotly_chart(px.line(df, x="Date", y="Demand", title="Demand Over Time", color_discrete_sequence=['#AB63FA']).update_layout(template="plotly_dark", height=300), use_container_width=True)
    with c_r: st.plotly_chart(px.histogram(df, x="Demand", title="Demand Frequency", color_discrete_sequence=['#00CC96']).update_layout(template="plotly_dark", height=300), use_container_width=True)

    st.subheader("Detailed Logs")
    st.dataframe(df, use_container_width=True, hide_index=True)

with t2:
    st.title("Demand & Risk Analysis")
    
    # 1. Window Bars
    df['RollSum'] = df['Demand'].rolling(window=window_size).sum()
    st.subheader(f"Demand Volume in {window_size}-Day Blocks")
    df['Window_Group'] = np.arange(len(df)) // window_size
    window_totals = df.groupby('Window_Group').agg({'Demand': 'sum', 'Date': 'first'}).reset_index()
    st.plotly_chart(px.bar(window_totals, x='Date', y='Demand', color_discrete_sequence=['#00CC96']).update_layout(template="plotly_dark", height=400), use_container_width=True)

    st.divider()
    
    # 2. Service Level Histogram (Re-calculates based on Window)
    st.subheader("Service Level Analysis")
    h_col, s_col = st.columns([1, 2])
    with h_col: hist_mode = st.radio("Focus:", ["Daily Demand", f"{window_size}-Day Window Sum"])
    with s_col: target_sl = st.select_slider("Target Service Level", options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99], value=0.95)

    hist_data = df["Demand"] if hist_mode == "Daily Demand" else df["RollSum"].dropna()
    cutoff = np.percentile(hist_data, target_sl * 100)
    max_val = hist_data.max()

    r1, r2, r3 = st.columns(3)
    r1.metric(f"{int(target_sl*100)}% SL Threshold", int(cutoff))
    r2.metric("Max Demand", int(max_val))
    r3.metric("Exposure Gap", int(max_val - cutoff), delta="Risk", delta_color="inverse")

    fig_h = px.histogram(hist_data, nbins=30, color_discrete_sequence=['#00CC96'], marginal="box")
    fig_h.add_vline(x=cutoff, line_dash="dash", line_color="red", annotation_text="Target SL")
    fig_h.add_vline(x=max_val, line_dash="dot", line_color="yellow", annotation_text="MAX")
    fig_h.update_layout(template="plotly_dark", height=450, bargap=0.1)
    st.plotly_chart(fig_h, use_container_width=True)
