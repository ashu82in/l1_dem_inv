import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# ------------------------------------------------
# 1. Page Config & Styling (RESTORED)
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Inventory Simulator Pro")

st.markdown(
    """
    <style>
    .block-container { padding: 2rem 5rem; }
    section[data-testid="stSidebar"] > div:first-child { padding-left: 3.5rem !important; }
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
# 2. Sidebar Inputs (RESTORED)
# ------------------------------------------------
st.sidebar.header("Simulation Settings")
avg_demand = st.sidebar.number_input("Average Daily Demand", value=25.0, key="avg_val")
cov = st.sidebar.number_input("CoV", value=0.0, step=0.1, key="cov_val")
num_days = st.sidebar.slider("Horizon (Days)", 10, 1000, 100)
regen_button = st.sidebar.button("Reset Demand Scenario")

st.sidebar.divider()
st.sidebar.header("Policy & Costs")
opening_balance = st.sidebar.number_input("Opening Balance", value=300)
lead_time = int(st.sidebar.number_input("Lead Time (Days)", value=3))
reorder_point = st.sidebar.number_input("Reorder Point (ROP)", value=150)
order_qty = st.sidebar.number_input("Order Quantity (Q)", value=200)
unit_value = st.sidebar.number_input("Value Per Unit ($)", value=100)
holding_cost_pct = st.sidebar.number_input("Annual Holding Cost %", value=20.0)
ordering_cost = st.sidebar.number_input("Cost Per Order ($)", value=500)

# ------------------------------------------------
# 3. Optimized Simulation Engine (RE-VECTORIZED)
# ------------------------------------------------
@st.cache_data
def run_sim_fast(q_val, _demand_seq, _demand_dates, op_bal, lt, rop, u_val, h_pct, o_cost):
    n = len(_demand_seq)
    inv = np.zeros(n)
    pipeline_stock = np.zeros(n)
    arrivals = np.zeros(n + lt + 1)
    orders = np.zeros(n)
    shortages = np.zeros(n)
    
    curr_inv = op_bal
    curr_pipe = 0
    daily_h_rate = (h_pct / 100) / 365

    for day in range(n):
        # 1. Process Arrivals
        recv = arrivals[day]
        curr_inv += recv
        curr_pipe -= recv
        
        # 2. Reorder Logic
        pos = curr_inv + curr_pipe
        placed = 0
        if pos <= rop:
            placed = q_val
            arrivals[day + lt] += placed
            curr_pipe += placed
        
        # 3. Fulfill Demand
        demand = _demand_seq[day]
        shortage = max(0, demand - curr_inv)
        curr_inv = max(0, curr_inv - demand)
        
        # Record
        inv[day] = curr_inv
        pipeline_stock[day] = curr_pipe
        orders[day] = placed
        shortages[day] = shortage

    # Create DF
    df_res = pd.DataFrame({
        "Date": _demand_dates,
        "Demand": _demand_seq,
        "Inventory": inv,
        "Shortage": shortages,
        "IsStockout": shortages > 0,
        "Pipeline": pipeline_stock,
        "Position": inv + pipeline_stock,
        "Order": orders,
        "InLT": pipeline_stock > 0,
        "HoldingCost": inv * u_val * daily_h_rate,
        "OrderingCost": np.where(orders > 0, o_cost, 0)
    })
    return df_res

# ------------------------------------------------
# 4. State Management & Simulation Execution (FIXED)
# ------------------------------------------------
# Track settings to force a refresh when sliders move
current_params = (avg_demand, cov, num_days)

if "demand_params" not in st.session_state or st.session_state.demand_params != current_params or regen_button:
    if cov <= 0:
        st.session_state.demand_seq = np.full(num_days, float(avg_demand))
    else:
        # This creates the zigzag (volatility)
        st.session_state.demand_seq = np.maximum(0, np.random.normal(avg_demand, avg_demand * cov, num_days)).round()
    
    st.session_state.demand_dates = pd.date_range(start="2024-01-01", periods=num_days)
    st.session_state.demand_params = current_params 
    
    # CRITICAL: Clear the cache for the simulation when demand changes
    # This ensures 'df' is recalculated with the new wavy demand
    run_sim_fast.clear()

# --- Run Simulations ---
# These now use the updated demand_seq from session_state
df = run_sim_fast(
    order_qty, 
    st.session_state.demand_seq, 
    st.session_state.demand_dates, 
    opening_balance, 
    lead_time, 
    reorder_point, 
    unit_value, 
    holding_cost_pct, 
    ordering_cost
)

# EOQ Setup
annual_d = avg_demand * 365
annual_h = unit_value * (holding_cost_pct / 100)
eoq_val = np.sqrt((2 * annual_d * ordering_cost) / annual_h)

eoq_df = run_sim_fast(
    eoq_val, 
    st.session_state.demand_seq, 
    st.session_state.demand_dates, 
    opening_balance, 
    lead_time, 
    reorder_point, 
    unit_value, 
    holding_cost_pct, 
    ordering_cost
)
# ------------------------------------------------
# 5. UI Layout (FULLY RESTORED)
# ------------------------------------------------
# Update the tabs list
t1, t2, t3 = st.tabs(["📊 Inventory Simulator", "📈 Demand Analyzer", "🕵️ Pattern Decoder"])


# ------------------------------------------------
# 6. Tab 3: Pattern Decoder (NEW)
# ------------------------------------------------


import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_pattern_decoder():
    st.header("🕵️ The Pattern Decoder: Beyond the Average")
    st.info("The Goal: Prove that an 'Average' is a lie in a seasonal business.")

    # --- 1. DATA GENERATOR (VECTORIZED) ---
    with st.expander("🛠️ Step 1: Create your Market DNA", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            dec_avg = st.number_input("Baseline Daily Demand", value=500, key="dec_avg")
        with c2:
            dec_cov = st.slider("Chaos % (CoV)", 0.05, 0.50, 0.15, key="dec_cov")
        with c3:
            dec_boost = st.slider("Peak Season Surge (%)", 20, 150, 60, key="dec_boost") / 100

        if st.button("✨ Generate Seasonal Pattern"):
            n_days = 730
            dates = pd.date_range("2024-01-01", periods=n_days)
            
            # Vectorized Sine Wave Generation
            t = np.linspace(0, 4 * np.pi, n_days)
            wave = np.sin(t)
            
            # Vectorized Multiplier Logic using np.select
            # This replaces the slow 'for' loop and 'if' statements
            conditions = [ (wave > 0.5), (wave < -0.5) ]
            multipliers = [ 1 + dec_boost, 1 - (dec_boost * 0.6) ]
            scale_array = np.select(conditions, multipliers, default=1.0)
            
            # Vectorized Seasonality Labeling
            label_choices = ["High", "Low"]
            labels = np.select(conditions, label_choices, default="Normal")
            
            # Vectorized Normal Distribution Generation
            # We generate all 730 random points at once scaled by our seasonality array
            final_demand = np.random.normal(dec_avg * scale_array, dec_avg * dec_cov)
            
            st.session_state.decoder_df = pd.DataFrame({
                "Date": dates,
                "Demand": np.maximum(0, np.round(final_demand, 0)), # Vectorized clip and round
                "Seasonality": labels
            })

    # --- 2. THE 3-STAGE ANALYSIS ---
    if "decoder_df" in st.session_state:
        df_dec = st.session_state.decoder_df
        
        # STAGE 1: RAW DATA & DOWNLOAD
        st.subheader("Stage 1: The 'Spreadsheet Blindness'")
        
        # Fast Buffer Export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_dec.to_excel(writer, index=False, sheet_name='Seasonal_Demand')
            
        st.download_button(
            label="📥 Download Demand Data as Excel",
            data=buffer.getvalue(),
            file_name="seasonal_demand_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Scrollable table with fixed height
        st.dataframe(df_dec, use_container_width=True, hide_index=True, height=350)

        # STAGE 2: LINE PLOT
        st.divider()
        st.subheader("Stage 2: The 'Trend & Seasonality' View")
        show_line = st.checkbox("🔍 Reveal Visual Pattern (Line Graph)", value=False)
        if show_line:
            fig_line = px.line(df_dec, x='Date', y='Demand', color_discrete_sequence=['#A0AEC0'])
            fig_line.update_layout(template="plotly_dark", height=400, hovermode="x")
            st.plotly_chart(fig_line, use_container_width=True)

        # STAGE 3: THE HISTOGRAM COMPARISON
        st.divider()
        st.subheader("Stage 3: The Probability Truth")
        col_all, col_sep = st.columns(2)
        
        with col_all:
            st.write("**A. The General Histogram**")
            fig_h1 = px.histogram(df_dec, x="Demand", nbins=40, color_discrete_sequence=['#63B3ED'])
            fig_h1.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_h1, use_container_width=True)

        with col_sep:
            st.write("**B. The Seasonal Breakdown**")
            fig_h2 = px.histogram(df_dec, x="Demand", color="Seasonality", nbins=40,
                                  color_discrete_map={"High": "#F56565", "Normal": "#63B3ED", "Low": "#4FD1C5"},
                                  barmode='overlay', opacity=0.75)
            fig_h2.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_h2, use_container_width=True)
            
#End of Tab 3 function

with t1:
    st.write("### Inventory Dashboard Controls")
    use_pipeline = st.checkbox("Include Pipeline Inventory in KPIs", value=False)
    target_col = "Position" if use_pipeline else "Inventory"
    label_suffix = "(Incl. Pipeline)" if use_pipeline else "(Physical Only)"

    # ROW 1: OPERATIONAL & SERVICE KPIs
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

    # ROW 2: DYNAMIC RANGE & WORKING CAPITAL
    st.write(f"### Range & Working Capital Analysis {label_suffix}")
    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric(f"Min Inv {label_suffix}", f"{df[target_col].min():.0f}")
    k7.metric(f"Max Inv {label_suffix}", f"{df[target_col].max():.0f}")
    k8.metric(f"Min WC {label_suffix}", f"${(df[target_col].min()*unit_value):,.0f}")
    k9.metric(f"Max WC {label_suffix}", f"${(df[target_col].max()*unit_value):,.0f}")
    k10.metric(f"Avg WC {label_suffix}", f"${(df[target_col].mean()*unit_value):,.0f}")

    # FINANCIAL EXPANDER
    with st.expander("💰 Financial Metrics & EOQ Comparison", expanded=False):
        st.write("### Cost Analysis")
        f1, f2, f3 = st.columns(3)
        total_cost_curr = df['HoldingCost'].sum() + df['OrderingCost'].sum()
        f1.metric("Total Holding Cost", f"${df['HoldingCost'].sum():,.0f}")
        f2.metric("Total Ordering Cost", f"${df['OrderingCost'].sum():,.0f}")
        f3.metric("Total Policy Cost", f"${total_cost_curr:,.0f}")

        st.write("### EOQ Strategy Comparison")
        e1, e2, e3, e4, e5 = st.columns(5)
        total_cost_eoq = eoq_df['HoldingCost'].sum() + eoq_df['OrderingCost'].sum()
        e1.metric("EOQ Value", f"{int(eoq_val)}")
        e2.metric("Selected Q", f"{order_qty}")
        e3.metric("Cost with EOQ", f"${total_cost_eoq:,.0f}")
        e4.metric("Savings using EOQ", f"${(total_cost_curr - total_cost_eoq):,.0f}")
        e5.metric("Cost Reduction %", f"{((total_cost_curr - total_cost_eoq)/total_cost_curr if total_cost_curr > 0 else 0):.1%}")

    st.divider()

    # CHARTS (FULL RESTORATION OF MARKERS & SHADING)
# --- MAIN INVENTORY CHART (FIXED RENDER MODE) ---
    st.subheader("Inventory Levels Over Time")
    fig = go.Figure()
    
    # 1. Fast Vectorized Shading
    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=np.where(df["InLT"], df["Position"].max(), np.nan),
        fill='tozeroy', 
        fillcolor='rgba(255, 0, 0, 0.05)', 
        line=dict(width=0), 
        name="Lead Time Window", 
        showlegend=False, 
        hoverinfo='skip'
    ))
    
    # 2. Main Lines
    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["Inventory"], 
        name="Physical Stock", 
        line=dict(color='#00CCFF', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["Position"], 
        name="Inventory Position", 
        line=dict(color='#FF9900', dash='dot')
    ))
    
    fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    # 3. Vectorized Event Markers
    orders = df[df["Order"] > 0]
    if not orders.empty:
        fig.add_trace(go.Scatter(
            x=orders["Date"], y=orders["Inventory"], 
            mode="markers", name="Order Placed", 
            marker=dict(color="#00FF00", size=10, symbol="triangle-up")
        ))
    
    shorts = df[df["IsStockout"]]
    if not shorts.empty:
        fig.add_trace(go.Scatter(
            x=shorts["Date"], y=shorts["Inventory"], 
            mode="markers", name="Shortage", 
            marker=dict(color="red", size=10, symbol="x")
        ))
    
    # --- THE CORRECTION ---
    # We apply WebGL to all scatter traces at once
    fig.update_traces(patch={"line": {"shape": "linear"}}, selector=dict(type='scatter'))
    
    fig.update_layout(
        template="plotly_dark", 
        height=450, 
        hovermode="x unified",
        # Remove 'render_mode' from here; it's not a layout property
    )

    # Alternative way to force WebGL if the above doesn't feel fast enough:
    # use_container_width=True
    st.plotly_chart(fig, use_container_width=True)

    # st.subheader("Inventory Levels Over Time")
    # fig = go.Figure()
    # # Shading for Lead Time Windows
    # for i, r in df.iterrows():
    #     if r["InLT"]: fig.add_vrect(x0=r["Date"], x1=r["Date"], fillcolor="red", opacity=0.05, layer="below", line_width=0)
    
    # fig.add_trace(go.Scatter(x=df["Date"], y=df["Inventory"], name="Physical Stock", line=dict(color='#00CCFF', width=2.5)))
    # fig.add_trace(go.Scatter(x=df["Date"], y=df["Position"], name="Inventory Position", line=dict(color='#FF9900', dash='dot')))
    # fig.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="ROP")
    
    # # Restored Logic for Markers
    # orders = df[df["Order"] > 0]
    # if not orders.empty:
    #     fig.add_trace(go.Scatter(x=orders["Date"], y=orders["Inventory"], mode="markers", name="Order Placed", 
    #                              marker=dict(color="#00FF00", size=10, symbol="triangle-up")))
    # shorts = df[df["IsStockout"] == True]
    # if not shorts.empty:
    #     fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["Inventory"], mode="markers", name="Shortage", 
    #                              marker=dict(color="red", size=10, symbol="x")))

    # fig.update_layout(template="plotly_dark", height=450, hovermode="x unified", yaxis=dict(rangemode="tozero"))
    # st.plotly_chart(fig, use_container_width=True)

    # # Restored Pipeline Area Chart
    # st.subheader("Pipeline Inventory (Units in Transit)")
    # fig_pipe = px.area(df, x="Date", y="Pipeline", color_discrete_sequence=['#FFCC00'])
    # fig_pipe.update_layout(template="plotly_dark", height=250, yaxis_title="Units", yaxis=dict(rangemode="tozero"))
    # st.plotly_chart(fig_pipe, use_container_width=True)

    # Restored Demand Visuals in Tab 1
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

with t2:
    st.title("Risk & Window Analysis")
    
    # 1. Setup Analysis Window
    window_size = st.slider("Select Analysis Window (Days)", 1, 30, 5, key="tab2_window")
    
    # 2. FORCE RE-CALCULATION of Rolling Demand
    # We pull directly from session_state to ensure we aren't using "stale" data
    current_demand_series = pd.Series(st.session_state.demand_seq)
    rolling_demand = current_demand_series.rolling(window=window_size).sum().dropna()

    if not rolling_demand.empty:
        # 3. Target Service Level Slider
        target_sl = st.slider(
            "Target Service Level", 
            0.50, 0.99, 0.95, 
            step=0.01, 
            format="%.2f"
        )
        
        # 4. Logic for Risk Gap
        cutoff = np.percentile(rolling_demand, target_sl * 100)
        max_demand = rolling_demand.max()
        min_demand = rolling_demand.min()
        risk_gap = max_demand - cutoff
        
        # 5. Metric Row
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"{int(target_sl*100)}% SL Threshold", f"{int(cutoff):.0f} Units")
        r2.metric("Max Demand Observed", f"{int(max_demand):.0f} Units")
        r3.metric("The Risk Gap", f"{int(risk_gap):.0f} Units", delta="Uncovered", delta_color="inverse")
        r4.metric("Risk Value Exposure", f"${int(risk_gap * unit_value):,}")

        # 6. Optimized Histogram
        fig_risk = px.histogram(
            rolling_demand, 
            nbins=30, 
            color_discrete_sequence=['#00CC96'], 
            title=f"Demand Distribution over {window_size}-Day Window"
        )
        
        # Add Visual Markers
        fig_risk.add_vline(x=cutoff, line_dash="dash", line_color="yellow", 
                           annotation_text=f"{int(target_sl*100)}% SL")
        fig_risk.add_vline(x=max_demand, line_dash="dot", line_color="red", 
                           annotation_text="Absolute MAX")
        
        # Shading the Unprotected Zone
        fig_risk.add_vrect(
            x0=cutoff, x1=max_demand, 
            fillcolor="red", opacity=0.15, 
            layer="below", line_width=0, 
            annotation_text="UNPROTECTED ZONE"
        )

        # CRITICAL: Force X-Axis scaling so it doesn't look like a solid block
        # We set the range to be 20% wider than the actual data spread
        x_range_extension = (max_demand - min_demand) * 0.2 if max_demand > min_demand else 10
        fig_risk.update_xaxes(
            range=[min_demand - x_range_extension, max_demand + x_range_extension],
            title="Total Demand in Window"
        )
        
        fig_risk.update_layout(template="plotly_dark", height=450, showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)
        
    else:
        st.warning("Please increase 'Horizon (Days)' in the sidebar to generate enough data for this window.")


with t3:
    # Call the new function
    render_pattern_decoder()
