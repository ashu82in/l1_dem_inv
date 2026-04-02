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
cov = st.sidebar.number_input("Coefficient of Variation (CoV)", value=0.1, step=0.1, key="cov_val")
num_days = st.sidebar.slider("Simulation Horizon (Days)", 10, 1000, 100, key="horizon_val")
regen_button = st.sidebar.button("🔄 Regenerate Demand")

st.sidebar.divider()
st.sidebar.header("Analysis Window")
window_size = st.sidebar.slider("Window Size (Days)", 1, 30, 7)

st.sidebar.divider()
st.sidebar.header("Chart Settings")
fixed_zero = st.sidebar.checkbox("Start Y-Axis at Zero", value=True)

st.sidebar.divider()
st.sidebar.header("Policy & Costs")
opening_balance = st.sidebar.number_input("Opening Balance", value=500)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=3, min_value=0)
reorder_point = st.sidebar.number_input("Reorder Point (ROP)", value=75)
order_qty = st.sidebar.number_input("Order Quantity (Q)", value=300)
unit_value = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_cost_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
ordering_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)

# ------------------------------------------------
# 3. Demand Generation (Reactive State)
# ------------------------------------------------
demand_state_key = f"{avg_demand}_{cov}_{num_days}"
if "last_demand_key" not in st.session_state or st.session_state.last_demand_key != demand_state_key or regen_button:
    st.session_state.last_demand_key = demand_state_key
    if cov <= 0:
        st.session_state.demand_seq = np.full(num_days, float(avg_demand))
    else:
        st.session_state.demand_seq = np.maximum(0, np.random.normal(avg_demand, avg_demand * cov, num_days)).round()
    st.session_state.demand_dates = pd.date_range(start="2024-01-01", periods=num_days)

if st.session_state.get("use_uploaded", False) and "uploaded_demand" in st.session_state:
    current_demand = st.session_state.uploaded_demand
    current_dates = st.session_state.uploaded_dates
else:
    current_demand = st.session_state.demand_seq
    current_dates = st.session_state.demand_dates

# ------------------------------------------------
# 4. Corrected Simulation Engine (Timing & Logic)
# ------------------------------------------------
def run_sim(q_val, d_seq, d_dates):
    inv = opening_balance
    pipeline = [] # Stores (arrival_day, qty)
    rows = []
    for day in range(len(d_seq)):
        # 1. Physical Arrival
        received = sum(order[1] for order in pipeline if order[0] == day)
        pipeline = [order for order in pipeline if order[0] != day]
        
        opening = inv
        inv += received
        daily_demand = d_seq[day]
        
        # 2. Demand Satisfaction & Stockout Logic
        is_unfulfilled = daily_demand > inv
        shortage = max(0, daily_demand - inv)
        inv = max(0, inv - daily_demand)
        
        # 3. Lead Time Indicator
        is_in_lead_time = len(pipeline) > 0
        
        # 4. Position Check & Order Trigger
        # Position = Physical + Pipeline (orders already placed but not arrived)
        current_pipeline_qty = sum(o[1] for o in pipeline)
        inv_pos_before = inv + current_pipeline_qty
        
        placed_qty = 0
        if inv_pos_before < reorder_point:
            placed_qty = q_val
            pipeline.append((day + lead_time, placed_qty))
        
        rows.append({
            "Date": d_dates[day].date(), "Opening": int(opening), "Demand": int(daily_demand), 
            "Shortage": int(shortage), "Is Stockout": is_unfulfilled, "In Lead Time": is_in_lead_time, 
            "Physical Inventory": int(inv), "Inventory Position": int(inv + sum(o[1] for o in pipeline)), 
            "New Order": int(placed_qty)
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty, current_demand, current_dates)

# ------------------------------------------------
# 5. Tabbed Interface
# ------------------------------------------------
tab1, tab2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.title("Inventory Policy Dashboard")
    
    # Advanced Metrics
    h_rate = (holding_cost_pct / 100)
    total_cost = (df["Inventory Position"] * unit_value * h_rate / 365).sum() + ((df["New Order"] > 0).sum() * ordering_cost)
    global_fr = ((df["Demand"].sum() - df["Shortage"].sum()) / df["Demand"].sum() * 100) if df["Demand"].sum() > 0 else 100
    
    lt_df = df[df["In Lead Time"] == True]
    lt_fr = ((lt_df["Demand"].sum() - lt_df["Shortage"].sum()) / lt_df["Demand"].sum() * 100) if not lt_df.empty and lt_df["Demand"].sum() > 0 else 100.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Unfulfilled Days", df["Is Stockout"].sum(), help="Demand > Supply")
    m2.metric("Global Fill Rate", f"{global_fr:.1f}%")
    m3.metric("Lead Time Fill Rate", f"{lt_fr:.1f}%", help="Fill rate during replenish cycles")
    m4.metric("Min Inventory", int(df["Physical Inventory"].min()))
    m5.metric("Avg Inventory", int(df["Physical Inventory"].mean()))
    m6.metric("Total Cost", f"${int(total_cost):,}")
    
    st.divider()
    
    # Primary Inventory Chart
    fig_inv = go.Figure()
    for i, row in df.iterrows():
        if row["In Lead Time"]:
            fig_inv.add_vrect(x0=row["Date"], x1=row["Date"], fillcolor="red", opacity=0.08, layer="below", line_width=0)

    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Physical Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig_inv.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    # Event Markers
    stockout_pts = df[df["Is Stockout"] == True]
    if not stockout_pts.empty:
        fig_inv.add_trace(go.Scatter(x=stockout_pts["Date"], y=stockout_pts["Physical Inventory"], mode="markers", name="Shortage Day", marker=dict(color="red", size=10, symbol="x")))
    
    trigger_pts = df[df["New Order"] > 0]
    if not trigger_pts.empty:
        fig_inv.add_trace(go.Scatter(x=trigger_pts["Date"], y=trigger_pts["Physical Inventory"], mode="markers", name="Order Placed", marker=dict(color="#00FF00", size=10, symbol="triangle-up")))

    fig_inv.update_layout(hovermode="x unified", template="plotly_dark", height=450, legend=dict(orientation="h", y=1.1), yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
    st.plotly_chart(fig_inv, use_container_width=True)

    # 1-Day Demand Context (Bottom of Tab 1)
    st.divider()
    st.subheader("Daily Demand Profile")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.line(df, x="Date", y="Demand", title="Daily Volatility", color_discrete_sequence=['#AB63FA']).update_layout(template="plotly_dark", height=300), use_container_width=True)
    with c2: st.plotly_chart(px.histogram(df, x="Demand", nbins=20, title="Daily Distribution", color_discrete_sequence=['#00CC96']).update_layout(template="plotly_dark", height=300, bargap=0.1), use_container_width=True)

    st.subheader("Simulation Log")
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- TAB 2: ANALYZER ---
with tab2:
    st.title("Demand & Risk Analysis")
    
    ca, cb = st.columns([2, 1])
    with ca:
        up = st.file_uploader("Upload External Demand (Excel/CSV)", type=["xlsx", "csv"])
        if up:
            u_df = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
            u_df['Date'] = pd.to_datetime(u_df['Date'])
            st.session_state.uploaded_demand = u_df['Demand'].values
            st.session_state.uploaded_dates = u_df['Date'].values
            st.success("File Ready! Toggle 'Activate' to use.")
    with cb:
        st.toggle("Activate Uploaded Data", key="use_uploaded")
        template_df = pd.DataFrame({"Date": ["2024-01-01"], "Demand": [25]})
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as wr: template_df.to_excel(wr, index=False)
        st.download_button("📥 Template", data=buf.getvalue(), file_name="template.xlsx")

    st.divider()
    
    # Window-wise blocks
    df['Rolling Sum'] = df['Demand'].rolling(window=window_size).sum()
    st.subheader(f"Demand Volume in {window_size}-Day Blocks")
    df['Window_Group'] = np.arange(len(df)) // window_size
    window_totals = df.groupby('Window_Group').agg({'Demand': 'sum', 'Date': 'first'}).reset_index()
    st.plotly_chart(px.bar(window_totals, x='Date', y='Demand', color_discrete_sequence=['#00CC96']).update_layout(template="plotly_dark", height=400), use_container_width=True)

    st.divider()
    
    # Service Level Math
    st.subheader("Service Level & Safety Stock Math")
    h_col, s_col = st.columns([1, 2])
    with h_col: hist_mode = st.radio("Histogram Analysis:", ["Daily Demand", f"{window_size}-Day Window Sum"])
    with s_col: target_sl = st.select_slider("Target Service Level", options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99], value=0.95)

    hist_data = df["Demand"] if hist_mode == "Daily Demand" else df["Rolling Sum"].dropna()
    cutoff = np.percentile(hist_data, target_sl * 100)
    max_val = hist_data.max()

    r1, r2, r3 = st.columns(3)
    r1.metric(f"{int(target_sl*100)}% SL Threshold", int(cutoff))
    r2.metric(f"Absolute Max ({hist_mode})", int(max_val))
    r3.metric("Uncovered Risk Gap", int(max_val - cutoff), delta="Units Exposed", delta_color="inverse")

    fig_h_win = px.histogram(hist_data, nbins=30, color_discrete_sequence=['#00CC96'], marginal="box")
    fig_h_win.add_vline(x=cutoff, line_dash="dash", line_color="red", annotation_text="SL Cutoff")
    fig_h_win.add_vline(x=max_val, line_dash="dot", line_color="yellow", annotation_text="MAX")
    fig_h_win.update_layout(template="plotly_dark", height=450, bargap=0.1, xaxis_title=f"Units ({hist_mode})")
    st.plotly_chart(fig_h_win, use_container_width=True)
