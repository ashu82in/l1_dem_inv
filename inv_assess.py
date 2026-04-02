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
    .block-container {
        padding-left: 5rem;
        padding-right: 5rem;
        padding-top: 2rem;
    }
    
    /* ZOOM PADDING: Buffer for Zoom UI on the left sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        padding-left: 2.5rem !important;
        padding-right: 1.5rem !important;
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 2. Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Simulation Settings")
avg_demand = st.sidebar.number_input("Average Daily Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation (CoV)", value=0.1, step=0.1)
num_days = st.sidebar.slider("Simulation Horizon (Days)", 10, 1000, 100)
regen_button = st.sidebar.button("🔄 Regenerate Demand")

st.sidebar.divider()
st.sidebar.header("Analysis Window")
window_size = st.sidebar.slider("Rolling Window Size (Days)", 1, 30, 7)

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
# 3. Demand Generation Logic (Persistent State)
# ------------------------------------------------
if "demand_seq" not in st.session_state or regen_button:
    if cov <= 0:
        st.session_state.demand_seq = np.full(num_days, float(avg_demand))
    else:
        st.session_state.demand_seq = np.maximum(0, np.random.normal(avg_demand, avg_demand * cov, num_days)).round()
    st.session_state.demand_dates = pd.date_range(start="2024-01-01", periods=num_days)

# Logic to switch between generated and uploaded
if st.session_state.get("use_uploaded", False) and "uploaded_demand" in st.session_state:
    current_demand = st.session_state.uploaded_demand
    current_dates = st.session_state.uploaded_dates
else:
    current_demand = st.session_state.demand_seq
    current_dates = st.session_state.demand_dates

# ------------------------------------------------
# 4. Simulation Engine
# ------------------------------------------------
def run_sim(q_val, d_seq, d_dates):
    inv = opening_balance
    pipeline = [] # List of (arrival_day, qty)
    rows = []
    for day in range(len(d_seq)):
        received = 0
        for order in pipeline[:]:
            if order[0] <= day:
                received += order[1]
                pipeline.remove(order)
        opening = inv
        inv += received
        daily_demand = d_seq[day]
        shortage = max(0, daily_demand - inv)
        inv = max(0, inv - daily_demand)
        inv_pos = inv + sum(o[1] for o in pipeline)
        
        placed_qty = 0
        if inv_pos < reorder_point:
            placed_qty = q_val
            if lead_time == 0: inv += placed_qty
            else: pipeline.append((day + lead_time, placed_qty))
        
        rows.append({
            "Date": d_dates[day], "Opening": int(opening), "Demand": int(daily_demand), 
            "Shortage": int(shortage), "Received": int(received), "Physical Inventory": int(inv), 
            "Inventory Position": int(inv + sum(o[1] for o in pipeline)), "New Order": int(placed_qty)
        })
    return pd.DataFrame(rows)

df = run_sim(order_qty, current_demand, current_dates)

# ------------------------------------------------
# 5. Tabs Layout
# ------------------------------------------------
tab1, tab2 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer"])

# --- TAB 1: INVENTORY SIMULATOR ---
with tab1:
    st.title("Inventory Policy Simulator")
    if st.session_state.get("use_uploaded", False):
        st.success("🟢 MODE: Using Uploaded Excel Data")
    
    # 1. KPIs
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
    
    # 2. Main Inventory Chart
    st.subheader("Inventory Levels Over Time")
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Physical Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    fig_inv.add_trace(go.Scatter(x=df["Date"], y=df["Inventory Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    fig_inv.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    y_config = dict(rangemode="tozero", range=[0, df["Inventory Position"].max() * 1.1]) if fixed_zero else dict(rangemode="normal")
    fig_inv.update_layout(hovermode="x unified", template="plotly_dark", height=450, legend=dict(orientation="h", y=1.1), yaxis=y_config)
    st.plotly_chart(fig_inv, use_container_width=True)

    # 3. 1-Day Demand Visuals (As requested)
    st.divider()
    st.subheader("Daily Demand Overview")
    c_line, c_hist = st.columns(2)
    with c_line:
        fig_daily_line = px.line(df, x="Date", y="Demand", title="Daily Demand Volatility", color_discrete_sequence=['#AB63FA'])
        fig_daily_line.update_layout(template="plotly_dark", height=350, yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
        st.plotly_chart(fig_daily_line, use_container_width=True)
    with c_hist:
        fig_daily_hist = px.histogram(df, x="Demand", nbins=20, title="Daily Demand Distribution", color_discrete_sequence=['#00CC96'], marginal="box")
        fig_daily_hist.update_layout(template="plotly_dark", height=350, bargap=0.1)
        st.plotly_chart(fig_daily_hist, use_container_width=True)

    st.subheader("Detailed Logs")
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- TAB 2: DEMAND ANALYZER ---
with tab2:
    st.title("Demand & Window Analysis")
    
    # 1. Management Section
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded_file = st.file_uploader("Upload External Demand (Excel/CSV)", type=["xlsx", "csv"])
        if uploaded_file:
            u_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            u_df['Date'] = pd.to_datetime(u_df['Date'])
            st.session_state.uploaded_demand = u_df['Demand'].values
            st.session_state.uploaded_dates = u_df['Date'].values
            st.success("File Ready!")
    with c2:
        st.toggle("Activate Uploaded Data", key="use_uploaded")
        template_df = pd.DataFrame({"Date": ["2024-01-01"], "Demand": [25]})
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as wr: template_df.to_excel(wr, index=False)
        st.download_button("📥 Excel Template", data=buf.getvalue(), file_name="demand_template.xlsx")

    st.divider()

    # 2. Rolling Trend & Window Bar Chart
    st.subheader(f"Rolling {window_size}-Day Trend")
    df['Rolling Mean'] = df['Demand'].rolling(window=window_size).mean()
    df['Rolling Sum'] = df['Demand'].rolling(window=window_size).sum()
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=df["Date"], y=df["Demand"], name="Daily Demand", line=dict(color='rgba(171, 99, 250, 0.2)')))
    fig_roll.add_trace(go.Scatter(x=df["Date"], y=df["Rolling Mean"], name="Rolling Average", line=dict(color='#AB63FA', width=3)))
    fig_roll.update_layout(template="plotly_dark", height=400, hovermode="x unified", yaxis=dict(rangemode="tozero" if fixed_zero else "normal"))
    st.plotly_chart(fig_roll, use_container_width=True)

    st.subheader(f"Total Demand in {window_size}-Day Blocks")
    df['Window_Group'] = np.arange(len(df)) // window_size
    window_totals = df.groupby('Window_Group').agg({'Demand': 'sum', 'Date': 'first'}).reset_index()
    fig_window_bar = px.bar(window_totals, x='Date', y='Demand', color_discrete_sequence=['#00CC96'])
    fig_window_bar.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_window_bar, use_container_width=True)

    st.divider()

    # 3. Dynamic Histogram & Risk Exposure (考慮到 Window)
    st.subheader("Service Level & Safety Stock Analysis")
    h_col, s_col = st.columns([1, 2])
    with h_col:
        hist_mode = st.radio("Select Distribution Focus:", ["Daily Demand", f"{window_size}-Day Window Demand"])
    with s_col:
        target_sl = st.select_slider("Target Service Level", options=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99], value=0.95)

    hist_data = df["Demand"] if hist_mode == "Daily Demand" else df["Rolling Sum"].dropna()
    cutoff = np.percentile(hist_data, target_sl * 100)
    max_val = hist_data.max()
    exposure_gap = max_val - cutoff

    r1, r2, r3 = st.columns(3)
    r1.metric(f"{int(target_sl*100)}% SL Threshold", int(cutoff))
    r2.metric(f"Max {hist_mode}", int(max_val))
    r3.metric("Uncovered Risk Gap", int(exposure_gap), delta="Exposure", delta_color="inverse")

    fig_hist = px.histogram(hist_data, nbins=30, color_discrete_sequence=['#00CC96'], marginal="box")
    fig_hist.add_vline(x=cutoff, line_dash="dash", line_color="red", annotation_text=f"{int(target_sl*100)}% SL")
    fig_hist.add_vline(x=max_val, line_dash="dot", line_color="yellow", annotation_text="MAX")
    fig_hist.update_layout(template="plotly_dark", height=450, bargap=0.1, xaxis_title=f"Units ({hist_mode})")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.info(f"📈 **Service Level Concept:** This threshold covers **{int(target_sl*100)}% of demand occurrences**. "
            f"The **{int(exposure_gap)} unit gap** to the Maximum is the risk accepted to optimize inventory.")
