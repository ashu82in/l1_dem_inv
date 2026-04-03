import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ------------------------------------------------
# 1. Page Config & Styling
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Simulator Pro")

st.markdown(
    """
    <style>
    .block-container { padding: 2rem 5rem; }
    section[data-testid="stSidebar"] > div:first-child { padding-left: 2.5rem !important; }
    
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
cov = st.sidebar.number_input("CoV", value=0.1, step=0.1, key="cov_val")
num_days = st.sidebar.slider("Horizon (Days)", 10, 1000, 100)
regen_button = st.sidebar.button("Reset Demand Scenario")

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
        
        pos_at_check = inv + sum(o[1] for o in pipeline)
        
        placed_qty = 0
        if pos_at_check <= reorder_point:
            placed_qty = q_val
            pipeline.append((day + lead_time, placed_qty))
        
        demand = d_seq[day]
        shortage = max(0, demand - inv)
        inv = max(0, inv - demand)
        
        final_pipe = sum(o[1] for o in pipeline)
        
        rows.append({
            "Date": d_dates[day].date(), "Demand": demand, "Shortage": shortage, 
            "IsStockout": shortage > 0, "Inventory": inv, "Pipeline": final_pipe,
            "Position": inv + final_pipe, "Order": placed_qty,
            "InLT": final_pipe > 0, "HoldingCost": inv * unit_value * daily_h_rate,
            "OrderingCost": ordering_cost if placed_qty > 0 else 0
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty)

# EOQ Setup
annual_d = avg_demand * 365
annual_h = unit_value * (holding_cost_pct / 100)
eoq_val = np.sqrt((2 * annual_d * ordering_cost) / annual_h)
eoq_df = run_sim(eoq_val)

# ------------------------------------------------
# 5. UI Layout
# ------------------------------------------------
t1, t2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

with t1:
    # --- DYNAMIC TOGGLE ---
    st.write("### Inventory Dashboard Controls")
    use_pipeline = st.checkbox("Include Pipeline Inventory in KPIs", value=False)
    
    # Logic for Toggle
    target_col = "Position" if use_pipeline else "Inventory"
    label_suffix = "(Incl. Pipeline)" if use_pipeline else "(Physical Only)"

    # --- ROW 1: OPERATIONAL & SERVICE KPIs ---
    st.subheader(f"Inventory Operational & Service KPIs {label_suffix}")
    lt_df = df[df["InLT"] == True]
    lt_fr = ((lt_df["Demand"].sum() - lt_df["Shortage"].sum()) / lt_df["Demand"].sum() * 100) if not lt_df.empty else 100.0
    global_fr = ((df["Demand"].sum() - df["Shortage"].sum()) / df["Demand"].sum() * 100) if df["Demand"].sum() > 0 else 100.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stockout Days", int(df["IsStockout"].sum()))
    k2.metric("Global Fill Rate", f"{global_fr:.1f}%")
    k3.metric("LT Fill Rate", f"{lt_fr:.1f}%")
    k4.metric(f"Avg Inv {label_suffix}", f"{df[target_col].mean():.1f}")
    k5.metric(f"Avg Age {label_suffix}", f"{(df[target_col].mean() / avg_demand):.1f} Days")

    # --- ROW 2: DYNAMIC RANGE & WORKING CAPITAL ---
    st.write(f"### Range & Working Capital Analysis {label_suffix}")
    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric(f"Min Inv {label_suffix}", f"{df[target_col].min():.0f}")
    k7.metric(f"Max Inv {label_suffix}", f"{df[target_col].max():.0f}")
    k8.metric(f"Min WC {label_suffix}", f"${(df[target_col].min()*unit_value):,.0f}")
    k9.metric(f"Max WC {label_suffix}", f"${(df[target_col].max()*unit_value):,.0f}")
    k10.metric(f"Avg WC {label_suffix}", f"${(df[target_col].mean()*unit_value):,.0f}")

    # --- COLLAPSIBLE FINANCIAL SECTION ---
    with st.expander("💰 Financial Metrics & EOQ Comparison", expanded=False):
        st.write("### Cost Analysis")
        f1, f2, f3, f4, f5 = st.columns(5)
        total_cost_curr = df['HoldingCost'].sum() + df['OrderingCost'].sum()
        f1.metric("Total Holding Cost", f"${df['HoldingCost'].sum():,.0f}")
        f2.metric("Total Ordering Cost", f"${df['OrderingCost'].sum():,.0f}")
        f3.metric("Total Policy Cost", f"${total_cost_curr:,.0f}")
        # f4.metric("Avg Total WC (Position)", f"${(df['Position'].mean()*unit_value):,.0f}")
        # f5.metric("Inv Turnover", f"{(annual_d / df['Inventory'].mean() if df['Inventory'].mean() > 0 else 0):.1f}x")

        st.write("### EOQ Strategy Comparison")
        e1, e2, e3, e4, e5 = st.columns(5)
        total_cost_eoq = eoq_df['HoldingCost'].sum() + eoq_df['OrderingCost'].sum()
        e1.metric("EOQ Value", f"{int(eoq_val)}")
        e2.metric("Selected Q", f"{order_qty}")
        e3.metric("Cost with EOQ", f"${total_cost_eoq:,.0f}")
        e4.metric("Savings using EOQ", f"${(total_cost_curr - total_cost_eoq):,.0f}")
        e5.metric("Cost Reduction %", f"{((total_cost_curr - total_cost_eoq)/total_cost_curr if total_cost_curr > 0 else 0):.1%}")

    st.divider()

    # --- CHARTS SECTION ---
    st.subheader("Inventory Levels Over Time")
    fig = go.Figure()
    for i, r in df.iterrows():
        if r["InLT"]: fig.add_vrect(x0=r["Date"], x1=r["Date"], fillcolor="red", opacity=0.05, layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    # Event Markers
    orders = df[df["Order"] > 0]
    if not orders.empty:
        fig.add_trace(go.Scatter(x=orders["Date"], y=orders["Inventory"], mode="markers", name="Order Placed", 
                                 marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
    shorts = df[df["IsStockout"] == True]
    if not shorts.empty:
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["Inventory"], mode="markers", name="Shortage", 
                                 marker=dict(color="red", size=10, symbol="x")))

    fig.update_layout(template="plotly_dark", height=450, hovermode="x unified", yaxis=dict(rangemode="tozero"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pipeline Inventory (Units in Transit)")
    fig_pipe = px.area(df, x="Date", y="Pipeline", color_discrete_sequence=['#FFCC00'])
    fig_pipe.update_layout(template="plotly_dark", height=250, yaxis_title="Units", yaxis=dict(rangemode="tozero"))
    st.plotly_chart(fig_pipe, use_container_width=True)

    # --- TAB 1 DEMAND VISUALS ---
    st.divider()
    st.subheader("Demand Analysis (Historical Context)")
    cl_hist, cl_line = st.columns(2)
    with cl_hist:
        fig_h1 = px.histogram(df, x="Demand", nbins=15, title="Demand Frequency", color_discrete_sequence=['#00CC96'])
        fig_h1.update_layout(template="plotly_dark", height=300, bargap=0.1)
        st.plotly_chart(fig_h1, use_container_width=True)
    with cl_line:
        fig_l1 = px.line(df, x="Date", y="Demand", title="Daily Volatility", color_discrete_sequence=['#AB63FA'])
        fig_l1.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_l1, use_container_width=True)

    st.subheader("Simulation Log")
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- TAB 2: RISK ANALYZER ---
# --- TAB 2: DEMAND ANALYZER ---
# --- TAB 2: DEMAND ANALYZER ---
with t2:
    st.title("Risk & Window Analysis")
    
    # CRITICAL: Define window_size here since it's used in this tab
    window_size = st.slider("Select Analysis Window (Days)", 1, 30, 7)
    
    # 1. Block Volume Analysis
    st.subheader(f"Demand Volume in {window_size}-Day Blocks")
    df['Block_Group'] = np.arange(len(df)) // window_size
    block_df = df.groupby('Block_Group').agg({'Date': 'first', 'Demand': 'sum'}).reset_index()
    fig_blocks = px.bar(block_df, x='Date', y='Demand', color_discrete_sequence=['#50C878'])
    fig_blocks.update_layout(template="plotly_dark", height=400, xaxis_title="Timeline", yaxis_title="Sum of Demand")
    st.plotly_chart(fig_blocks, use_container_width=True)

    st.divider()

    # 2. Risk Gap Analysis
    st.subheader("Service Level vs. Maximum Exposure")
    df['RollSum'] = df['Demand'].rolling(window=window_size).sum()
    hist_data = df['RollSum'].dropna()
    target_sl = st.select_slider("Target Service Level", options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99], value=0.95)
    
    cutoff = np.percentile(hist_data, target_sl * 100)
    max_demand = hist_data.max()
    risk_gap = max_demand - cutoff

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"{int(target_sl*100)}% SL Threshold", f"{int(cutoff)} Units")
    r2.metric("Max Demand Observed", f"{int(max_demand)} Units")
    r3.metric("The Risk Gap", f"{int(risk_gap)} Units", delta="Uncovered", delta_color="inverse")
    r4.metric("Risk Value Exposure", f"${int(risk_gap * unit_value):,}")

    fig_risk = px.histogram(hist_data, nbins=30, color_discrete_sequence=['#00CC96'], title=f"Demand Distribution over {window_size}-Day Window")
    fig_risk.add_vline(x=cutoff, line_dash="dash", line_color="yellow", annotation_text=f"{int(target_sl*100)}% SL")
    fig_risk.add_vline(x=max_demand, line_dash="dot", line_color="red", annotation_text="Absolute MAX")
    fig_risk.add_vrect(x0=cutoff, x1=max_demand, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="UNPROTECTED ZONE")
    fig_risk.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_risk, use_container_width=True)
