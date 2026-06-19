import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.colors as pc
import altair as alt
import random

# --- Session State Initialization ---
if 'next_clicked' not in st.session_state:
    st.session_state.next_clicked = False
if 'seed_counter' not in st.session_state:
    st.session_state.seed_counter = 42

st.set_page_config(page_title="Supply Chain Analytics Platform", layout="wide")

st.title("🚀 Supply Chain Analytics Platform")

# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Average Demand", "📊 Demand Analyzer and Histogram", "Continuous Review", "🔄 Periodic Review", "Inventory Audit", "Multi SKU Simulator"])

game_tab, tab1, tab2 = st.tabs(["🎲 Game Simulator", "Average Demand", "📊 Demand Analyzer and Histogram" ])


with game_tab:
    st.header("🎯 Bag of Balls: Game Simulator")
    st.write("Configure your game rules, set your budget, and simulate the outcomes!")

    # --- INPUT CONFIGURATIONS ---
    st.subheader("Game Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_val = st.number_input("Min Ball Number", value=1, step=1)
        max_val = st.number_input("Max Ball Number", value=5, step=1)
        
    with col2:
        picks = st.number_input("Number of Picks", value=1, min_value=1, step=1)
        win_multiple = st.number_input("Winning Multiple", value=6.0, step=0.5)
        
    with col3:
        budget = st.number_input("Total Budget ($)", value=100.0, min_value=0.0, step=10.0)
        bet_size = st.number_input("Per Bet Size ($)", value=10.0, min_value=1.0, step=1.0)

    # --- DYNAMIC BOUNDS FOR USER CHOICE ---
    min_possible_sum = int(min_val * picks)
    max_possible_sum = int(max_val * picks)

    st.divider()
    chosen_number = st.number_input(
        f"Choose your lucky target sum! (Possible range: {min_possible_sum} to {max_possible_sum})", 
        min_value=min_possible_sum, 
        max_value=max_possible_sum, 
        value=min_possible_sum, 
        step=1
    )

    # --- SIMULATION LOGIC (VECTORIZED) ---
    if st.button("🎲 Play / Simulate", use_container_width=True):
        if bet_size <= 0:
            st.error("Bet size must be greater than 0.")
        elif budget < bet_size:
            st.error("Your budget is too low to make even one bet.")
        elif min_val >= max_val:
            st.error("Max ball number must be strictly greater than Min ball number.")
        else:
            chances = int(budget // bet_size)
            
            # 1. Vectorized Drawing: Generate a matrix of ALL drawn balls at once
            # Shape: (chances, picks). e.g., 10 chances of 2 picks = 10x2 matrix
            drawn_matrix = np.random.choice(np.arange(min_val, max_val + 1), size=(chances, picks))
            
            # 2. Vectorized Sum: Sum across the rows (axis=1)
            sums_array = np.sum(drawn_matrix, axis=1)
            
            # 3. Vectorized Win Checking
            wins_array = (sums_array == chosen_number)
            
            # 4. Vectorized Payouts
            payouts_array = np.where(wins_array, bet_size * win_multiple, 0.0)
            
            # Calculate total winnings instantly
            total_winnings = float(np.sum(payouts_array))
            
            # --- RESULTS DISPLAY ---
            st.subheader("📊 Simulation Results")
            
            # Build DataFrame quickly from NumPy arrays
            df = pd.DataFrame({
                "Chance #": np.arange(1, chances + 1),
                "Drawn Balls": [str(list(row)) for row in drawn_matrix], # String formatting needs a quick loop
                "Sum of Balls": sums_array,
                "Win?": np.where(wins_array, "✅ Yes", "❌ No"),
                "Payout ($)": payouts_array
            })
            
            st.dataframe(df, use_container_width=True)
            
            # --- UPDATED FREQUENCY TABLE (PANDAS VECTORIZED) ---
            st.subheader("📈 Sum Frequency Analysis")
            
            # Count frequencies using Pandas value_counts
            sum_counts = pd.Series(sums_array).value_counts()
            
            # Ensure all possible sums are represented, even if 0
            all_possible_sums = pd.Index(range(min_possible_sum, max_possible_sum + 1), name="Sum of Balls")
            sum_counts = sum_counts.reindex(all_possible_sums, fill_value=0).reset_index()
            sum_counts.columns = ["Sum of Balls", "Times Drawn"]
            
            # Vectorized Percentage Math
            if chances > 0:
                sum_counts["Draw Rate (%)"] = (sum_counts["Times Drawn"] / chances * 100).round(2)
            
            st.dataframe(sum_counts, use_container_width=True)
            
            # --- FINANCIAL SUMMARY ---
            st.divider()
            st.write("### 💰 Financial Summary")
            
            total_spent = chances * bet_size
            net_profit = total_winnings - total_spent
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Total Chances Played", chances)
            col_res2.metric("Total Amount Spent", f"${total_spent:.2f}")
            col_res3.metric("Total Gross Winnings", f"${total_winnings:.2f}")
            
            if net_profit > 0:
                st.success(f"**Net Profit:** ${net_profit:.2f} 🤑")
            elif net_profit < 0:
                st.error(f"**Net Loss:** ${abs(net_profit):.2f} 📉")
            else:
                st.info("**Break Even!** You made back exactly what you spent. ⚖️")



with tab1:
    st.header("The Basic Thumb Rule Used For Inventory Planning")
    
    # --- Step 1: Baseline Strategy Input Section ---
    col1, col2 = st.columns(2)
    
    with col1:
        annual_sales = st.number_input("Annual Sales (Units)", value=12000, step=500)
        working_days = st.number_input("Working Days per Year", value=300)
        
    with col2:
        # Calculate Average Daily Sales (ADS) baseline
        avg_daily_sales = annual_sales / working_days
        st.metric("Avg. Daily Sales (ADS)", f"{avg_daily_sales:.2f}")
        
        suggested_baseline = avg_daily_sales * 10
        requisite_inventory = st.number_input(
            "Enter Requisite Inventory Strategy Limit", 
            value=int(suggested_baseline),
            help="This is the target inventory volume you have allocated to cover your business lead time window."
        )

    # Trigger persistent UI view state
    if st.button("Next"):
        st.session_state.next_clicked = True

    # --- Step 2: Persisted Stress-Testing Environment ---
    if st.session_state.next_clicked:
        st.divider()
        st.subheader("🎯 Stress Test Parameters & Reality Simulator")
        
        # User Parameter Input Boxes
        c1, c2, c3 = st.columns(3)
        with c1:
            std_dev = st.number_input("Demand Standard Deviation (Volatility)", value=10, min_value=0)
        with c2:
            sim_days = st.number_input("Number of Simulation Days", value=100, min_value=1)
        with c3:
            rolling_window = st.number_input("Look-Forward Window (Days)", value=10, min_value=1, max_value=int(sim_days))

        # Action Buttons Layout: Regenerate Button
        btn_col1, btn_col2 = st.columns([1, 5])
        with btn_col1:
            if st.button("🔄 Regenerate Demand", key="regen_tab1"):
                st.session_state.seed_counter += 1  # Shifts the seed to force a new layout run

        # Generate Volatile Demand Data Array
        np.random.seed(st.session_state.seed_counter)
        daily_demand = np.random.normal(avg_daily_sales, std_dev, sim_days)
        daily_demand = np.clip(daily_demand, 0, None).round(0)  # Prevents impossible negative demand days
        
        days = [f"Day {i+1}" for i in range(sim_days)]
        
        # --- Visual Asset 1: Daily Demand Timeline ---
        st.write("### 📈 Daily Demand Volatility")
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=days, y=daily_demand, mode='lines+markers', name='Daily Demand Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_daily.add_hline(y=avg_daily_sales, line_dash="dash", line_color="gray", annotation_text="Calculated Static Average")
        fig_daily.update_layout(template="plotly_white", height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_daily, use_container_width=True)

        # Pre-calculating Data for Tables & Charts
        df_summary = pd.DataFrame({
            "Lead Time Day": days,
            "Daily Demand (Units)": daily_demand.astype(int)
        })
        
        # Look-Forward Core Mathematical Optimization Matrix
        forward_sums = df_summary["Daily Demand (Units)"].iloc[::-1].rolling(window=rolling_window).sum().iloc[::-1]
        df_summary[f"Demand Next {rolling_window} Days"] = forward_sums
        df_summary["Inventory Level Provided"] = int(requisite_inventory)
        
        # Metric Scorecard Data Compilation
        valid_forward_days = forward_sums.dropna()
        total_valid_days = len(valid_forward_days)
        deficits_series = valid_forward_days > requisite_inventory
        total_deficits = deficits_series.sum()
        pct_deficits = (total_deficits / total_valid_days * 100) if total_valid_days > 0 else 0.0
        
        # Calculate Maximum Forward Window Value
        max_window_demand = valid_forward_days.max() if total_valid_days > 0 else 0.0

        # --- Visual Asset 2: Collapsible Diagnostic Data Table & Scorecard ---
        with st.expander("📋 Generated Demand Data Table", expanded=False):
            st.markdown("### 📊 Window Analysis Summary")
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Days with Valid Window", f"{total_valid_days} Days")
            with m2:
                peak_gap = int(max_window_demand - requisite_inventory)
                st.metric(
                    "Max Window Demand Peak", 
                    f"{int(max_window_demand)} Units",
                    delta=f"+{peak_gap} Over Limit" if peak_gap > 0 else f"{peak_gap} Under Limit",
                    delta_color="inverse" if peak_gap > 0 else "normal"
                )
            with m3:
                st.metric("Total Deficit Occurrences", f"{total_deficits} Days", 
                          delta=f"-{total_deficits} Stockouts" if total_deficits > 0 else None, 
                          delta_color="inverse" if total_deficits > 0 else "normal")
            with m4:
                st.metric("Deficit Risk Rate (%)", f"{pct_deficits:.1f}%",
                          delta="CRITICAL RISK" if pct_deficits > 30 else "STABLE BUFFER",
                          delta_color="inverse" if pct_deficits > 30 else "normal")
                
            st.divider()

            def calculate_status(row):
                forward_demand = row[f"Demand Next {rolling_window} Days"]
                if pd.isna(forward_demand):
                    return ""
                
                net_value = int(row["Inventory Level Provided"] - forward_demand)
                if net_value >= 0:
                    return f'<span style="color: #2e7d32; font-weight: bold;">🟢 Surplus (+{net_value})</span>'
                else:
                    return f'<span style="color: #d32f2f; font-weight: bold;">🔴 Deficit ({net_value})</span>'

            df_table = df_summary.copy()
            df_table["Net Status"] = df_table.apply(calculate_status, axis=1)
            df_table[f"Demand Next {rolling_window} Days"] = df_table[f"Demand Next {rolling_window} Days"].apply(
                lambda x: f"{int(x)}" if not pd.isna(x) else ""
            )
            
            st.write(df_table.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.write("<br>", unsafe_allow_html=True)

        # --- Visual Asset 3: Collapsible Charts for Forward Window Analytics ---
        with st.expander("📊 View Forward Window Trend & Distribution Analysis", expanded=False):
            df_clean_charts = df_summary.dropna().copy()
            
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                st.markdown(f"#### 📉 Forward Window Demand Trend")
                fig_trend = go.Figure()
                
                fig_trend.add_trace(go.Scatter(
                    x=df_clean_charts["Lead Time Day"], 
                    y=df_clean_charts[f"Demand Next {rolling_window} Days"],
                    mode='lines',
                    name=f'{rolling_window}-Day Demand',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_trend.add_hline(
                    y=requisite_inventory, 
                    line_dash="dash", 
                    line_color="#d62728", 
                    annotation_text="Your Stock Limit",
                    annotation_position="top left"
                )
                fig_trend.update_layout(
                    template="plotly_white", 
                    xaxis_title="Simulation Day Index",
                    yaxis_title="Total Window Units",
                    height=350,
                    margin=dict(t=30, b=10)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with graph_col2:
                st.markdown(f"#### 📊 Look-Forward Window Distribution")
                
                fig_hist = px.histogram(
                    df_clean_charts, 
                    x=f"Demand Next {rolling_window} Days",
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(
                    x=requisite_inventory, 
                    line_dash="dash", 
                    line_color="#d62728", 
                    annotation_text="Stock Ceiling",
                    annotation_position="top right"
                )
                fig_hist.update_layout(
                    template="plotly_white",
                    xaxis_title=f"Aggregated Demand in {rolling_window}-Day Windows",
                    yaxis_title="Frequency Occurrence Count",
                    height=350,
                    margin=dict(t=30, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        if total_deficits > 0:
            st.error(f"❌ **Internal Sabotage Confirmed:** Volatility breached your static 'Average' allocation baseline strategy on **{total_deficits} separate window cycles** ({pct_deficits:.1f}% risk rate).")
        else:
            st.success(f"✅ **Strategic Parameter Verified.** Under these isolated settings, the current allocation buffer safely absorbed the simulated variance across all window blocks.")


# ==========================================
# TAB 2: DEMAND HISTOGRAM ANALYZER
# ==========================================
with tab2:
    st.header("Demand Analyzer")
    
    st.subheader("1. Data Configuration")
    data_source = st.radio("Select Data Source:", ("Generate Synthetic Data", "Upload Your Own Data"), horizontal=True, key="ds_p1")

    df = None

    if data_source == "Generate Synthetic Data":
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            dist_type = st.selectbox("Distribution Type", ("Normal", "Poisson", "Uniform"), key="dist_p1")
        with col_b:
            avg_demand = st.number_input("Average Demand", min_value=1.0, value=100.0, key="avg_p1")
        with col_c:
            num_periods = st.number_input("Number of Periods", min_value=10, value=10000, key="periods_p1")
        with col_d:
            if dist_type == "Normal":
                variation = st.number_input("Std Dev (Variation)", min_value=0.1, value=15.0, key="v_norm")
            elif dist_type == "Uniform":
                variation = st.number_input("Range (+/-)", min_value=1.0, value=30.0, key="v_uni")
            else:
                st.markdown("<p style='padding-top:25px; color:gray;'>Poisson variation fixed by Mean.</p>", unsafe_allow_html=True)

        np.random.seed(42)
        if dist_type == "Normal":
            generated = np.random.normal(avg_demand, variation, num_periods)
        elif dist_type == "Poisson":
            generated = np.random.poisson(avg_demand, num_periods)
        else:
            generated = np.random.uniform(avg_demand - variation, avg_demand + variation, num_periods)
        
        df = pd.DataFrame({'Demand': np.floor(np.clip(generated, 0, None))})

    elif data_source == "Upload Your Own Data":
        up_col1, up_col2 = st.columns([2, 1])
        
        with up_col1:
            uploaded_file = st.file_uploader("Upload your historical demand file (.xlsx or .csv):", type=["xlsx", "csv"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_upload = pd.read_csv(uploaded_file)
                    else:
                        df_upload = pd.read_excel(uploaded_file)
                    
                    available_columns = df_upload.columns.tolist()
                    
                    # --- NEW LOGIC: Check for both 'Demand' and 'Store Sale' ---
                    target_col = None
                    if 'Demand' in available_columns:
                        target_col = 'Demand'
                    elif 'Store Sale' in available_columns:
                        target_col = 'Store Sale'
                        
                    if target_col is None:
                        st.warning("⚠️ Column named 'Demand' or 'Store Sale' not found. Please select the appropriate column below:")
                        default_idx = 0
                    else:
                        default_idx = available_columns.index(target_col)
                    
                    selected_column = st.selectbox("Select the Demand Data Column:", options=available_columns, index=default_idx)
                    
                    if selected_column:
                        df = df_upload[[selected_column]].dropna().copy()
                        df.rename(columns={selected_column: 'Demand'}, inplace=True)
                        df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
                        df = df.dropna()
                        st.success(f"✅ File successfully parsed using column: **{selected_column}**")
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
                    
        with up_col2:
            st.markdown("#### 📋 Download Template")
            st.caption("Please match your data format to this template. Alternatively, you can upload any file and select the target column.")
            
            template_df = pd.DataFrame({'Demand': [120, 95, 110, 135, 80, 105, 115]})
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                template_df.to_excel(writer, index=False, sheet_name='Template')
            
            st.download_button(
                label="📥 Download Excel Template",
                data=buffer.getvalue(),
                file_name="demand_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    if df is not None:
        with st.expander("🔢 View / Download Raw Data Table", expanded=False):
            raw_display_df = df.copy()
            raw_display_df.index.name = "Period"
            
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.dataframe(raw_display_df, use_container_width=True, height=250)
            with exp_col2:
                st.markdown("#### Export Current Data")
                st.caption("Download this active dataset as a CSV file for offline use.")
                csv_data = raw_display_df.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="demand_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    if df is not None:
        st.divider()
        st.subheader("2. Volatility & Statistical Analysis")
        
        cov_col1, cov_col2 = st.columns([1, 2])
        
        with cov_col1:
            st.markdown("#### Formula")
            st.latex(r"CoV = \frac{\sigma}{\mu}")
            st.caption(r"Where $\sigma$ = Standard Deviation and $\mu$ = Mean")
            
            st.markdown("#### 📋 Statistical Summary")
            summary_stats = df['Demand'].describe().to_frame().T
            st.dataframe(summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']], use_container_width=True)
            
        with cov_col2:
            mean_val = float(df['Demand'].mean())
            std_val = float(df['Demand'].std())
            cov_val = (std_val / mean_val) if mean_val > 0 else 0.0
            
            if cov_val <= 0.10:
                status_text = "🟢 Ultra-Stable / Constant"
                explanation = "Highly repetitive and predictable demand. Use automated just-in-time (JIT) scheduling or lean kanbans. Minimize safety stock to release working capital."
            elif cov_val <= 0.25:
                status_text = "🟢 Stable / Predictable"
                explanation = "Normal variation patterns present. Standard statistical forecasting and fixed reorder points will yield high accuracy with minimal safety stock buffers."
            elif cov_val <= 0.50:
                status_text = "🟡 Moderate Volatility"
                explanation = "Demand exhibits noticeable fluctuations. Requires proactive demand sensing and traditional statistical safety stocks to counter stockout risks."
            elif cov_val <= 1.00:
                status_text = "🟠 High Volatility"
                explanation = "Highly variable demand spikes. Avoid automated ordering systems without collaborative forecasting inputs. Expect to maintain higher, dynamic safety stock thresholds."
            else:
                status_text = "🔴 Erratic / Lumpy / Sporadic"
                explanation = "Highly unpredictable or intermittent demand. Traditional safety stock formulas do not work well here. Consider move-to-order (MTO) execution or project-based buffers."
                
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Mean ($\mu$)", f"{mean_val:.2f}")
            with m_col2:
                st.metric("Std Dev ($\sigma$)", f"{std_val:.2f}")
            with m_col3:
                st.metric("Calculated CoV", f"{cov_val:.3f}")
                
            st.markdown(f"### Profile: {status_text}")
            st.info(explanation)

        st.divider()
        st.subheader("3. Probability & Coverage Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("#### Threshold Lookup (Points Below X)")
            threshold = st.number_input("Enter Demand Value:", value=40.0, step=1.0)
            count_below = len(df[df['Demand'] < threshold])
            percent_below = (count_below / len(df)) * 100
            st.metric(f"Chances of Demand < {threshold}", f"{percent_below:.1f}%")
            st.caption(f"There are {count_below} periods where demand was less than {threshold}.")

        with analysis_col2:
            st.markdown("#### Percentile Lookup (Coverage Level)")
            target_perc = st.number_input("Enter Service Level % (e.g. 95):", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
            demand_at_perc = np.percentile(df['Demand'], target_perc)
            st.metric(f"Demand at {target_perc}% Service Level", f"{int(demand_at_perc)}")
            st.caption(f"To cover {target_perc}% of all periods, you need to satisfy a demand of {int(demand_at_perc)}.")

        st.divider()
        st.subheader("4. Visual Distribution")
        
        num_bins = st.slider("Select Number of Bins:", 5, 50, 15)
        
        counts, bin_edges = np.histogram(df['Demand'], bins=num_bins)
        bin_size = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1

        fig = px.histogram(df, x="Demand", template="plotly_white", color_discrete_sequence=['#4F8BF9'])
        
        fig.update_traces(
            xbins=dict(
                start=bin_edges[0],
                end=bin_edges[-1],
                size=bin_size
            )
        )
        
        fig.add_vline(
            x=threshold, 
            line_dash="dot", 
            line_color="#EF553B", 
            line_width=2.5,
            annotation_text=f"Threshold ({threshold})", 
            annotation_position="top left"
        )
        
        fig.add_vline(
            x=demand_at_perc, 
            line_dash="dot", 
            line_color="#00CC96", 
            line_width=2.5,
            annotation_text=f"{target_perc}% Service Level ({int(demand_at_perc)})", 
            annotation_position="top right"
        )
        
        fig.update_layout(bargap=0.1, xaxis_title="Demand Quantity", yaxis_title="Count of Periods")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Bin Frequency Table")
        pct_total = counts / len(df) * 100
        
        bin_df = pd.DataFrame({
            "Bin Range": [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
            "Frequency (Count)": counts,
            "% of Total": pct_total.round(1),
            "Cum. Count": counts.cumsum(),
            "Cum. %": pct_total.cumsum().round(1)
        })
        st.dataframe(bin_df, use_container_width=True, hide_index=True)

# with tab3:
#     st.markdown(
#         """
#         <style>
#         .block-container {
#             padding-left: 2rem;
#             padding-right: 2rem;
#             padding-top: 2rem;
#         }
#         [data-testid="column"]:first-child {
#             border-right: 1px solid rgba(255, 255, 255, 0.1);
#             padding-right: 2rem;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     st.header("Inventory Policy Simulator")
#     st.divider()

#     # Main split layout
#     input_col, output_col = st.columns([1, 3])

#     # ================================================
#     # LEFT PANEL: INPUTS
#     # ================================================
#     with input_col:
#         st.subheader("⚙️ Parameters")
        
#         st.markdown("**Basic Settings**")
#         opening_balance = st.number_input("Opening Balance", value=500, key="sim_ob")
#         unit_value = st.number_input("Value Per Unit", value=100, key="sim_vu")
#         num_days = st.slider("Simulation Days", 100, 2000, 365, key="sim_nd")
        
#         st.markdown("**Demand Settings**")
#         avg_demand = st.number_input("Average Demand", value=25, key="sim_ad")
#         cov = st.number_input("Demand CoV", value=0.8, key="sim_cov")
        
#         if "demand_sequence_tab3" not in st.session_state:
#             st.session_state.demand_sequence_tab3 = None

#         if st.button("🔄 Generate New Demand", key="reset_dem", use_container_width=True):
#             st.session_state.demand_sequence_tab3 = None
            
#         st.markdown("**Policy Settings**")
#         lead_time = st.number_input("Lead Time (Days)", value=3, key="sim_lt")
#         reorder_point = st.number_input("Reorder Point", value=200, key="sim_rp")
#         order_qty = st.number_input("Order Quantity", value=300, key="sim_oq")
        
#         st.markdown("**Cost Metrics**")
#         holding_cost_percent = st.number_input("Holding Cost (%)", value=20.0, key="sim_hc")
#         ordering_cost = st.number_input("Ordering Cost / Order", value=500, key="sim_oc")

#     # ================================================
#     # BACKGROUND CALCULATIONS
#     # ================================================
#     holding_cost_rate = holding_cost_percent / 100
#     std_demand = avg_demand * cov

#     if st.session_state.demand_sequence_tab3 is None:
#         st.session_state.demand_sequence_tab3 = np.maximum(
#             0,
#             np.random.normal(avg_demand, std_demand, num_days)
#         ).round()

#     demand = st.session_state.demand_sequence_tab3
#     dates = pd.date_range(start="2024-01-01", periods=num_days)

#     inventory = opening_balance
#     pipeline_orders = []
#     data = []

#     for day in range(num_days):
#         shipment_received = 0
#         for order in pipeline_orders.copy():
#             if order[0] == day:
#                 shipment_received += order[1]
#                 pipeline_orders.remove(order)

#         opening = inventory
#         inventory += shipment_received
#         demand_today = demand[day]
#         inventory -= demand_today

#         if inventory < 0:
#             inventory = 0

#         pipeline_qty = sum(qty for arrival, qty in pipeline_orders)
#         inventory_position = opening - demand_today + shipment_received + pipeline_qty
#         new_order = 0

#         if inventory_position < reorder_point:
#             new_order = order_qty
#             pipeline_orders.append((day + lead_time, order_qty))

#         closing = inventory
#         closing_with_pipeline = closing + sum(qty for arrival, qty in pipeline_orders)

#         data.append([
#             dates[day], opening, demand_today, shipment_received, pipeline_qty,
#             inventory_position, new_order, closing, closing_with_pipeline
#         ])

#     df = pd.DataFrame(data, columns=[
#         "Date", "Opening Balance", "Demand", "Shipment Received", "Pipeline Order",
#         "Inventory Position", "New Order", "Closing Balance", "Closing Balance Including Pipeline"
#     ])

#     # KPI logic execution
#     stockout_days = (df["Closing Balance"] == 0).sum()
#     average_inventory = df["Closing Balance Including Pipeline"].mean()
#     average_age_inventory = average_inventory / df["Demand"].mean() if df["Demand"].mean() > 0 else 0

#     df["Blocked Working Capital"] = df["Inventory Position"] * unit_value
#     average_working_capital = df["Blocked Working Capital"].mean()

#     min_inventory = df["Closing Balance"].min()
#     max_inventory = df["Closing Balance"].max()
#     min_wc = df["Blocked Working Capital"].min()
#     max_wc = df["Blocked Working Capital"].max()

#     df["Inventory Value"] = df["Closing Balance Including Pipeline"] * unit_value
#     df["Holding Cost"] = df["Inventory Value"] * holding_cost_rate / 365
#     total_holding_cost = df["Holding Cost"].sum()

#     number_of_orders = (df["New Order"] > 0).sum()
#     total_ordering_cost = number_of_orders * ordering_cost
#     total_inventory_cost = total_holding_cost + total_ordering_cost

#     annual_demand = avg_demand * 365
#     holding_cost_per_unit = unit_value * holding_cost_rate
#     eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit) if holding_cost_per_unit > 0 else 0

#     def simulate_inventory_cost(order_quantity):
#         sim_inv = opening_balance
#         sim_pipeline = []
#         holding_cost_total = 0
#         orders_count = 0

#         for day in range(num_days):
#             shipment_rec = 0
#             for order in sim_pipeline.copy():
#                 if order[0] == day:
#                     shipment_rec += order[1]
#                     sim_pipeline.remove(order)

#             sim_inv += shipment_rec
#             dem_today = demand[day]
#             sim_inv -= dem_today

#             if sim_inv < 0:
#                 sim_inv = 0

#             pip_qty = sum(qty for arrival, qty in sim_pipeline)
#             inv_pos = sim_inv + pip_qty

#             if inv_pos < reorder_point:
#                 sim_pipeline.append((day + lead_time, order_quantity))
#                 orders_count += 1

#             close_w_pip = sim_inv + sum(qty for arrival, qty in sim_pipeline)
#             inv_val = close_w_pip * unit_value
#             hold_cost_today = inv_val * holding_cost_rate / 365
#             holding_cost_total += hold_cost_today

#         order_cost_tot = orders_count * ordering_cost
#         return holding_cost_total + order_cost_tot

#     cost_current_policy = simulate_inventory_cost(order_qty)
#     cost_eoq_policy = simulate_inventory_cost(int(eoq))


#     # ================================================
#     # RIGHT PANEL: OUTPUTS & DASHBOARD
#     # ================================================
#     with output_col:
        
#         # Matrix Collapsible Section 1: Core KPIs
#         with st.expander("📊 View Core Inventory & Financial Metrics", expanded=True):
#             st.markdown("#### Primary KPIs")
#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("Stockout Days", stockout_days)
#             c2.metric("Avg Age of Inventory", round(average_age_inventory, 1))
#             c3.metric("Average Inventory", round(average_inventory, 0))
#             c4.metric("Avg Working Capital", f"${round(average_working_capital, 0):,}")

#             st.markdown("#### Inventory & Capital Ranges")
#             r1, r2, r3, r4 = st.columns(4)
#             r1.metric("Minimum Inventory", round(min_inventory, 0))
#             r2.metric("Maximum Inventory", round(max_inventory, 0))
#             r3.metric("Min Working Capital", f"${round(min_wc, 0):,}")
#             r4.metric("Max Working Capital", f"${round(max_wc, 0):,}")

#             st.markdown("#### Cost Metrics Breakdown")
#             cc1, cc2, cc3 = st.columns(3)
#             cc1.metric("Total Holding Cost", f"${round(total_holding_cost, 0):,}")
#             cc2.metric("Total Ordering Cost", f"${round(total_ordering_cost, 0):,}")
#             cc3.metric("Total Inventory Cost", f"${round(total_inventory_cost, 0):,}")

#         # Matrix Collapsible Section 2: Optimization
#         with st.expander("💡 View EOQ & Policy Optimization", expanded=False):
#             st.markdown("#### Economic Order Quantity (EOQ)")
#             e1, e2 = st.columns(2)
#             e1.metric("Economic Order Quantity (EOQ)", round(eoq, 0))
#             e2.metric("Selected Order Quantity", order_qty)
            
#             st.markdown("#### Policy Financial Comparison")
#             k1, k2, k3 = st.columns(3)
#             k1.metric("Cost with Current Policy", f"${round(cost_current_policy, 0):,}")
#             k2.metric("Cost with EOQ Policy", f"${round(cost_eoq_policy, 0):,}")
#             k3.metric("Savings Using EOQ", f"${round(cost_current_policy - cost_eoq_policy, 0):,}")


#         # Main Behaviour Chart
#         st.markdown("#### Inventory Behaviour")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing Inventory"))
#         fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance Including Pipeline"], name="Inventory Position"))
#         fig.add_hline(y=reorder_point, line_dash="dash", annotation_text="Reorder Point")

#         stockouts = df[df["Closing Balance"] == 0]
#         fig.add_trace(go.Scatter(x=stockouts["Date"], y=stockouts["Closing Balance"], mode="markers", name="Stockout", marker=dict(color="red", size=9)))

#         reorders = df[df["New Order"] > 0]
#         fig.add_trace(go.Scatter(x=reorders["Date"], y=reorders["Closing Balance"], mode="markers", name="Reorder Trigger", marker=dict(color="green", symbol="triangle-up", size=10)))

#         fig.add_hrect(y0=0, y1=reorder_point*0.5, fillcolor="red", opacity=0.08)
#         fig.add_hrect(y0=reorder_point*0.5, y1=reorder_point, fillcolor="yellow", opacity=0.08)
#         fig.add_hrect(y0=reorder_point, y1=df["Closing Balance Including Pipeline"].max()*1.2, fillcolor="green", opacity=0.05)
#         fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
#         fig.update_yaxes(rangemode="tozero")
        
#         st.plotly_chart(fig, use_container_width=True)
#         st.divider()

#         # Secondary Charts (Grid Layout)
#         chart_col1, chart_col2 = st.columns(2)
        
#         with chart_col1:
#             st.markdown("#### Blocked Working Capital")
#             fig_wc = px.line(df, x="Date", y="Blocked Working Capital")
#             fig_wc.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
#             st.plotly_chart(fig_wc, use_container_width=True)

#             st.markdown("#### Demand Distribution")
#             fig_hist = px.histogram(df, x="Demand", nbins=20)
#             fig_hist.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
#             st.plotly_chart(fig_hist, use_container_width=True)

#         with chart_col2:
#             st.markdown("#### Pipeline Orders")
#             fig_pipeline = px.line(df, x="Date", y="Pipeline Order")
#             fig_pipeline.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
#             st.plotly_chart(fig_pipeline, use_container_width=True)

#             st.markdown("#### Orders Placed")
#             orders = df[df["New Order"] > 0]
#             fig_orders = px.scatter(orders, x="Date", y="New Order")
#             fig_orders.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=280)
#             st.plotly_chart(fig_orders, use_container_width=True)

#         st.divider()

#         # Deep Dives (Expanders to conserve vertical space)
#         with st.expander("📊 View Interactive Waterfall Analysis & Raw Data"):
#             st.markdown("#### Inventory Flow Waterfall")
#             selected_day = st.slider("Select Day for Waterfall Analysis", 0, len(df)-1, 0, key="waterfall_slider")
#             row = df.iloc[selected_day]

#             fig_waterfall = go.Figure(go.Waterfall(
#                 measure=["absolute", "relative", "relative", "total"],
#                 x=["Opening Balance", "Demand", "Shipment Received", "Closing Balance"],
#                 y=[row["Opening Balance"], -row["Demand"], row["Shipment Received"], row["Closing Balance"]]
#             ))
#             fig_waterfall.update_layout(margin=dict(l=0, r=0, t=30, b=0))
#             st.plotly_chart(fig_waterfall, use_container_width=True)
            
#             st.markdown("#### Simulation Output Table")
#             st.dataframe(df, use_container_width=True)
        

# # ==========================================
# # TAB 3: PERIODIC REVIEW (VECTORIZED LOGIC)
# # ==========================================
# with tab4:
#     st.header("🔄 Periodic Review Analysis (Target-Level System)")
    
#     st.markdown("""
#     In a periodic review system, inventory is checked at fixed intervals. The strategy must account for the mechanical reality of the **Protection Interval**—the time from when an order is placed until the *next* order can be placed and received.
#     """)

#     # --- Action: Regenerate Demand Button ---
#     btn_col1, btn_col2 = st.columns([1, 5])
#     with btn_col1:
#         if st.button("🔄 Generate New Demand", key="regen_demand_pr"):
#             st.session_state.seed_counter += 1

#     # --- 1. Baseline System Parameters Input ---
#     st.subheader("1. Supply Chain Parameters & Recommended Baseline")
    
#     p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
#     with p_col1:
#         pr_avg_demand = st.number_input("Avg Daily Demand", value=100.0, step=10.0)
#         pr_std_dev = st.number_input("Demand Std Dev", value=15.0, step=5.0)
#     with p_col2:
#         review_period = st.number_input("Recommended Review (Days)", value=14, min_value=1, step=1)
#         lead_time = st.number_input("Lead Time (Days)", value=7, min_value=1, step=1)
#     with p_col3:
#         target_service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
#         z_score = norm.ppf(target_service_level / 100.0)
#     with p_col4:
#         unit_cost = st.number_input("Unit Cost ($)", value=50.0, step=5.0)
#         ordering_cost = st.number_input("Ordering Cost ($/order)", value=250.0, step=50.0, key="baseline_oc")
#     with p_col5:
#         holding_cost_pct = st.number_input("Annual Holding Cost (%)", value=20.0, step=1.0)
#         holding_cost_annual = unit_cost * (holding_cost_pct / 100.0)
#         holding_cost_daily = holding_cost_annual / 365.0

#     # Calculate Recommended Baseline Target
#     protection_interval = review_period + lead_time
#     expected_demand_pi = pr_avg_demand * protection_interval
#     std_dev_pi = pr_std_dev * np.sqrt(protection_interval)
#     safety_stock = z_score * std_dev_pi
#     recommended_target = expected_demand_pi + safety_stock

#     st.info(f"**Calculated Baseline Target:** {int(recommended_target)} Units (Accommodating a {review_period}-day review cycle and {lead_time}-day lead time).")

#     # --- 2. Multi-Scenario Customization Setup ---
#     st.divider()
    
#     def sync_ordering_costs():
#         for i in range(5):  # Max slider value is 5
#             st.session_state[f"oc_key_{i}"] = st.session_state.baseline_oc

#     head_col1, head_col2 = st.columns([2, 1])
#     with head_col1:
#         st.subheader("2. Multi-Scenario Strategy Comparison")
#     with head_col2:
#         st.write("") # Spacing
#         st.button("📋 Sync Baseline Cost to All Scenarios", on_click=sync_ordering_costs)
        
#     st.markdown("Test the recommended baseline against custom strategies. Modify the review period to see the mathematically optimum target update instantly.")
    
#     num_scenarios = st.slider("Select Number of Custom Scenarios to Compare:", min_value=1, max_value=5, value=2)
    
#     scenarios_data = []
#     s_cols = st.columns(num_scenarios)
    
#     for i, col in enumerate(s_cols):
#         with col:
#             st.markdown(f"##### Scenario {i+1}")
            
#             default_t = int(review_period + ((i+1) * 7)) 
#             t_val = st.number_input(f"Review Period (Days)", value=default_t, min_value=1, step=1, key=f"t_{i}")
            
#             u_pi = t_val + lead_time
#             opt_target = (pr_avg_demand * u_pi) + (z_score * (pr_std_dev * np.sqrt(u_pi)))
#             st.caption(f"✨ **Optimum Target:** {int(opt_target)} Units")
#             target_val = st.number_input(f"Target Level (Units)", value=int(opt_target), step=50, key=f"target_{i}")
            
#             if f"oc_key_{i}" not in st.session_state:
#                 st.session_state[f"oc_key_{i}"] = ordering_cost
                
#             oc_val = st.number_input(f"Ordering Cost ($)", step=10.0, key=f"oc_key_{i}")
            
#             scenarios_data.append({
#                 'name': f"Scenario {i+1}", 
#                 'T': t_val, 
#                 'Target': target_val, 
#                 'OrderCost': oc_val
#             })

#     # --- NumPy Optimized Simulation Engine ---
#     np.random.seed(st.session_state.seed_counter)
#     sim_days_pr = 365
#     daily_demand_pr = np.clip(np.random.normal(pr_avg_demand, pr_std_dev, sim_days_pr), 0, None).round(0)

#     def simulate_periodic_system_vectorized(demand_array, T, L, target, order_c, hold_c_daily):
#         sim_days = len(demand_array)
#         inventory_history = np.zeros(sim_days)
#         receipts = np.zeros(sim_days + L + 1) 
        
#         current_inv = target
#         order_sizes = []
#         units_fulfilled = 0
        
#         for day in range(sim_days):
#             current_inv += receipts[day]
#             current_demand = demand_array[day]
            
#             fulfilled = min(max(current_inv, 0), current_demand)
#             current_inv -= fulfilled
#             units_fulfilled += fulfilled
            
#             inventory_history[day] = current_inv
            
#             if day % T == 0:
#                 on_order = np.sum(receipts[day+1:day+L+1])
#                 inv_position = current_inv + on_order
                
#                 if inv_position < target:
#                     order_qty = target - inv_position
#                     receipts[day + L] += order_qty
#                     order_sizes.append(order_qty)
                    
#         holding_units_total = np.sum(np.maximum(inventory_history, 0))
#         orders_placed = len(order_sizes)
#         total_order_cost = orders_placed * order_c
#         total_holding_cost = holding_units_total * hold_c_daily
#         total_demand_sim = np.sum(demand_array)
        
#         return {
#             'history': inventory_history,
#             'total_demand': total_demand_sim,
#             'units_fulfilled': units_fulfilled,
#             'lost_sales': total_demand_sim - units_fulfilled,
#             'fill_rate': (units_fulfilled / total_demand_sim) * 100 if total_demand_sim > 0 else 0,
#             'orders_placed': orders_placed,
#             'min_order_size': np.min(order_sizes) if order_sizes else 0,
#             'max_order_size': np.max(order_sizes) if order_sizes else 0,
#             'avg_order_size': np.mean(order_sizes) if order_sizes else 0,
#             'avg_inventory': holding_units_total / sim_days,
#             'max_inventory': np.max(np.maximum(inventory_history, 0)),
#             'min_inventory': np.min(inventory_history),
#             'total_order_cost': total_order_cost,
#             'total_holding_cost': total_holding_cost,
#             'total_cost': total_order_cost + total_holding_cost
#         }

#     # Execute simulations 
#     res_baseline = simulate_periodic_system_vectorized(daily_demand_pr, review_period, lead_time, recommended_target, ordering_cost, holding_cost_daily)
    
#     scenario_results = []
#     for s in scenarios_data:
#         res = simulate_periodic_system_vectorized(daily_demand_pr, s['T'], lead_time, s['Target'], s['OrderCost'], holding_cost_daily)
#         scenario_results.append(res)

#     # --- 3. Logically Bifurcated Summary Tables ---
#     st.divider()
#     st.markdown("### 📊 Policy Comparison & KPI Summary")
    
#     def fmt_usd(val): return f"${val:,.2f}"
    
#     # 3A. Operational Health Matrix
#     st.markdown("#### A. Operational & Capital Health Matrix")
#     ops_data = {
#         "Metric": [
#             "Review Interval", 
#             "Target Inventory Level", 
#             "Fill Rate (%)", 
#             "Lost Sales (Units)", 
#             "Min Inventory Level (Depth)",
#             "Avg Working Capital", 
#             "Max Working Capital"
#         ]
#     }
#     ops_data["Recommended Baseline"] = [
#         f"{review_period} Days", f"{int(recommended_target)}", f"{res_baseline['fill_rate']:.2f}%", 
#         f"{int(res_baseline['lost_sales'])}", f"{int(res_baseline['min_inventory'])}",
#         fmt_usd(res_baseline['avg_inventory'] * unit_cost), fmt_usd(res_baseline['max_inventory'] * unit_cost)
#     ]
#     for idx, s in enumerate(scenarios_data):
#         res = scenario_results[idx]
#         ops_data[s['name']] = [
#             f"{s['T']} Days", f"{int(s['Target'])}", f"{res['fill_rate']:.2f}%", 
#             f"{int(res['lost_sales'])}", f"{int(res['min_inventory'])}",
#             fmt_usd(res['avg_inventory'] * unit_cost), fmt_usd(res['max_inventory'] * unit_cost)
#         ]
#     st.dataframe(pd.DataFrame(ops_data), use_container_width=True, hide_index=True)

#     # 3B. Order Dynamics Matrix
#     st.markdown("#### B. Order Dynamics Matrix")
#     order_data = {
#         "Metric": ["Total No. of Orders", "Average Order Size", "Minimum Order Size", "Maximum Order Size"]
#     }
#     order_data["Recommended Baseline"] = [
#         f"{res_baseline['orders_placed']}", f"{int(res_baseline['avg_order_size'])} Units", 
#         f"{int(res_baseline['min_order_size'])} Units", f"{int(res_baseline['max_order_size'])} Units"
#     ]
#     for idx, s in enumerate(scenarios_data):
#         res = scenario_results[idx]
#         order_data[s['name']] = [
#             f"{res['orders_placed']}", f"{int(res['avg_order_size'])} Units", 
#             f"{int(res['min_order_size'])} Units", f"{int(res['max_order_size'])} Units"
#         ]
#     st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

#     # 3C. Financial Matrix
#     st.markdown("#### C. Financial Projections Matrix")
#     fin_data = {
#         "Metric": ["Applied Ordering Cost ($/order)", "Total Ordering Cost", "Total Holding Cost", "Total System Cost"]
#     }
#     fin_data["Recommended Baseline"] = [
#         fmt_usd(ordering_cost), fmt_usd(res_baseline['total_order_cost']), 
#         fmt_usd(res_baseline['total_holding_cost']), fmt_usd(res_baseline['total_cost'])
#     ]
#     for idx, s in enumerate(scenarios_data):
#         res = scenario_results[idx]
#         fin_data[s['name']] = [
#             fmt_usd(s['OrderCost']), fmt_usd(res['total_order_cost']), 
#             fmt_usd(res['total_holding_cost']), fmt_usd(res['total_cost'])
#         ]
#     st.dataframe(pd.DataFrame(fin_data), use_container_width=True, hide_index=True)

#     # --- 4. Visual Bifurcation & Trajectory ---
#     chart_col1, chart_col2 = st.columns([1, 1])
#     colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
#     with chart_col1:
#         st.markdown("#### Cost Bifurcation Analysis")
#         names = ["Baseline"] + [s['name'] for s in scenarios_data]
#         order_costs = [res_baseline['total_order_cost']] + [r['total_order_cost'] for r in scenario_results]
#         hold_costs = [res_baseline['total_holding_cost']] + [r['total_holding_cost'] for r in scenario_results]
        
#         fig_cost = go.Figure(data=[
#             go.Bar(name='Ordering Cost', x=names, y=order_costs, marker_color='#2ca02c'),
#             go.Bar(name='Holding Cost', x=names, y=hold_costs, marker_color='#1f77b4')
#         ])
#         fig_cost.update_layout(
#             barmode='stack', template="plotly_white", yaxis_title="Total Cost ($)",
#             height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
#         st.plotly_chart(fig_cost, use_container_width=True)

#     with chart_col2:
#         st.markdown("#### Physical Inventory Trajectory")
#         fig_comp = go.Figure()
        
#         fig_comp.add_trace(go.Scatter(
#             x=list(range(sim_days_pr)), y=res_baseline['history'], mode='lines', 
#             name='Baseline', line=dict(color='#1f77b4', width=3)
#         ))
        
#         for idx, s in enumerate(scenarios_data):
#             fig_comp.add_trace(go.Scatter(
#                 x=list(range(sim_days_pr)), y=scenario_results[idx]['history'], mode='lines', 
#                 name=s['name'], line=dict(color=colors[idx], width=1.5, dash='dot')
#             ))
        
#         fig_comp.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
#         fig_comp.update_layout(
#             template="plotly_white", xaxis_title="Simulation Day", yaxis_title="Units On Hand",
#             height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
#         st.plotly_chart(fig_comp, use_container_width=True)

#     # --- 5. Blocked Working Capital Chart ---
#     st.write("<br>", unsafe_allow_html=True)
#     st.markdown("#### 💰 Blocked Working Capital Trajectory")
#     st.caption("Visualizes the daily capital tied up on the warehouse floor (ignores backorders).")
    
#     fig_wc = go.Figure()
    
#     # Baseline WC
#     baseline_wc = np.maximum(res_baseline['history'], 0) * unit_cost
#     fig_wc.add_trace(go.Scatter(
#         x=list(range(sim_days_pr)), y=baseline_wc, mode='lines', 
#         name='Baseline', line=dict(color='#1f77b4', width=3)
#     ))
    
#     # Scenarios WC
#     for idx, s in enumerate(scenarios_data):
#         scenario_wc = np.maximum(scenario_results[idx]['history'], 0) * unit_cost
#         fig_wc.add_trace(go.Scatter(
#             x=list(range(sim_days_pr)), y=scenario_wc, mode='lines', 
#             name=s['name'], line=dict(color=colors[idx], width=1.5, dash='dot')
#         ))
        
#     fig_wc.update_layout(
#         template="plotly_white", xaxis_title="Simulation Day", yaxis_title="Capital Blocked ($)",
#         height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#     )
#     st.plotly_chart(fig_wc, use_container_width=True)

#     # --- 6. Collapsible Raw Data Logs ---
#     with st.expander("📋 View Daily Simulation Log Tables"):
#         st.markdown("Raw 365-day tracking for Physical Inventory levels side-by-side.")
        
#         log_data = {
#             "Day": range(1, sim_days_pr + 1),
#             "Daily Demand": daily_demand_pr.astype(int),
#             "Baseline Inv": res_baseline['history'].astype(int)
#         }
        
#         for idx, s in enumerate(scenarios_data):
#             log_data[f"{s['name']} Inv"] = scenario_results[idx]['history'].astype(int)
            
#         log_df = pd.DataFrame(log_data)
        
#         def highlight_stockouts(val):
#             color = '#ffcccc' if isinstance(val, (int, float)) and val < 0 else ''
#             return f'background-color: {color}'
            
#         st.dataframe(
#             log_df.style.map(highlight_stockouts, subset=[c for c in log_df.columns if 'Inv' in c]), 
#             use_container_width=True, hide_index=True
#         )



# # with tab4:
# #     st.header("🔙 Historical Policy Backtesting")
# #     st.markdown("""
# #     Upload your actual historical demand data to simulate how different inventory policies would have performed. 
# #     Compare **Continuous Review** (ordering a fixed quantity when stock hits a minimum) against **Periodic Review** (ordering up to a target level at fixed intervals).
# #     """)

# #     # --- 1. Data Ingestion & Basic EDA ---
# #     st.subheader("1. Data Upload & Demand Profiling")
    
# #     col_up1, col_up2 = st.columns([2, 1])
# #     with col_up1:
# #         uploaded_file = st.file_uploader("Upload Historical Demand (.csv or .xlsx).", type=["csv", "xlsx"], key="backtest_upload")
    
# #     with col_up2:
# #         st.markdown("#### 📋 Template")
# #         st.caption("Ensure your file has a header named **Demand**, **Store Sale**, or select it manually upon upload. Include **Closing Balance** to enable historical comparison matrices.")
# #         template_df = pd.DataFrame({
# #             'Day': [1, 2, 3, 4, 5], 
# #             'Demand': [120, 95, 110, 135, 80],
# #             'Closing Balance': [100, 75, 50, 25, 0],
# #             'Receipts': [0, 0, 0, 0, 150]
# #         })
# #         buffer = io.BytesIO()
# #         with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
# #             template_df.to_excel(writer, index=False, sheet_name='Template')
# #         st.download_button(label="📥 Download Template", data=buffer.getvalue(), file_name="backtest_template.xlsx", mime="application/vnd.ms-excel", use_container_width=True)

# #     if uploaded_file is not None:
# #         try:
# #             if uploaded_file.name.endswith('.csv'):
# #                 df_bt = pd.read_csv(uploaded_file)
# #             else:
# #                 df_bt = pd.read_excel(uploaded_file)
                
# #             # --- Column Cleanup (Handles "\n" in headers) ---
# #             df_bt.columns = [str(c).replace('\n', ' ').strip() for c in df_bt.columns]
                
# #             # --- Dynamic Column Mapping ---
# #             if 'Store Sale' in df_bt.columns:
# #                 df_bt.rename(columns={'Store Sale': 'Demand'}, inplace=True)
# #             elif 'Bhiwandi Sales' in df_bt.columns:
# #                 df_bt.rename(columns={'Bhiwandi Sales': 'Demand'}, inplace=True)
# #             elif 'Sales' in df_bt.columns:
# #                 df_bt.rename(columns={'Sales': 'Demand'}, inplace=True)
                
# #             if 'Demand' not in df_bt.columns:
# #                 st.warning("⚠️ Could not automatically detect a 'Demand' column.")
# #                 selected_col = st.selectbox("Please select the column that represents your daily demand:", df_bt.columns)
# #                 if selected_col:
# #                     df_bt.rename(columns={selected_col: 'Demand'}, inplace=True)
# #                 else:
# #                     st.stop()
                
# #             df_bt['Demand'] = pd.to_numeric(df_bt['Demand'], errors='coerce').fillna(0)
            
# #             # --- 3. Demand Mean, Std Dev and CoV ---
# #             mean_dem = df_bt['Demand'].mean()
# #             std_dem = df_bt['Demand'].std()
# #             cov_dem = (std_dem / mean_dem) if mean_dem > 0 else 0
            
# #             st.markdown("#### 📊 Statistical Profile")
# #             m1, m2, m3, m4 = st.columns(4)
# #             m1.metric("Total Periods (Days)", len(df_bt))
# #             m2.metric("Mean Demand ($\mu$)", f"{mean_dem:.1f}")
# #             m3.metric("Std Dev ($\sigma$)", f"{std_dem:.1f}")
# #             m4.metric("CoV", f"{cov_dem:.3f}", delta="High Volatility" if cov_dem > 0.5 else "Stable", delta_color="inverse" if cov_dem > 0.5 else "normal")
            
# #             # --- Graphical Section 1: Demand Profiles ---
# #             st.write("<br>", unsafe_allow_html=True)
# #             chart_c1, chart_c2 = st.columns(2)
            
# #             x_axis_hist = df_bt.index
# #             if 'Date' in df_bt.columns:
# #                 x_axis_hist = df_bt['Date']
# #             elif 'Day' in df_bt.columns:
# #                 x_axis_hist = df_bt['Day']
            
# #             with chart_c1:
# #                 st.markdown("**📉 Historical Demand Curve**")
# #                 fig_line = go.Figure()
# #                 fig_line.add_trace(go.Scatter(x=x_axis_hist, y=df_bt['Demand'], mode='lines', name='Demand', line=dict(color='#1f77b4', width=2)))
# #                 fig_line.update_layout(template="plotly_white", height=300, xaxis_title="Period", yaxis_title="Units", margin=dict(t=10, b=10))
# #                 st.plotly_chart(fig_line, use_container_width=True)
                
# #             with chart_c2:
# #                 st.markdown("**📊 Demand Distribution (Histogram)**")
# #                 fig_hist = px.histogram(df_bt, x='Demand', nbins=20, template="plotly_white", color_discrete_sequence=['#85c1e9'])
# #                 fig_hist.update_layout(height=300, xaxis_title="Demand Quantity", yaxis_title="Frequency", margin=dict(t=10, b=10))
# #                 st.plotly_chart(fig_hist, use_container_width=True)

# #             # --- Graphical Section 2: Historical Closing Balance ---
# #             hist_has_inv = 'Closing Balance' in df_bt.columns
# #             if hist_has_inv:
# #                 df_bt['Closing Balance'] = pd.to_numeric(df_bt['Closing Balance'], errors='coerce').fillna(0)
# #                 st.write("<br>", unsafe_allow_html=True)
# #                 st.markdown("#### 📉 Actual Historical Closing Stock (From Uploaded Data)")
                
# #                 fig_hist_close = go.Figure()
# #                 fig_hist_close.add_trace(go.Scatter(
# #                     x=x_axis_hist, y=df_bt['Closing Balance'], mode='lines', name='Historical Closing Balance',
# #                     line=dict(color='#9467bd', width=2), fill='tozeroy', fillcolor='rgba(148, 103, 189, 0.15)'
# #                 ))
# #                 fig_hist_close.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
# #                 fig_hist_close.update_layout(template="plotly_white", xaxis_title="Historical Period", yaxis_title="Units On Hand", height=300, margin=dict(t=10, b=10))
# #                 st.plotly_chart(fig_hist_close, use_container_width=True)

# #             # --- 2. Cost & Policy Configuration ---
# #             st.divider()
# #             st.subheader("2. Financial Costs & Policy Configuration")
            
# #             cost_c1, cost_c2, cost_c3, cost_c4, cost_c5 = st.columns(5)
# #             with cost_c1:
# #                 bt_unit_cost = st.number_input("Unit Cost ($)", value=50.0, step=5.0, key="bt_uc")
# #             with cost_c2:
# #                 bt_order_cost = st.number_input("Ordering Cost ($/order)", value=250.0, step=50.0, key="bt_oc")
# #             with cost_c3:
# #                 bt_hold_pct = st.number_input("Annual Holding Cost (%)", value=20.0, step=1.0, key="bt_hp")
# #             with cost_c4:
# #                 bt_lead_time = st.number_input("Lead Time (Days)", value=7, min_value=1, step=1, key="bt_lt")
# #             with cost_c5:
# #                 bt_service_lvl = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1, key="bt_sl")
                
# #             bt_hold_daily = (bt_unit_cost * (bt_hold_pct / 100.0)) / 365.0
# #             bt_z_score = norm.ppf(bt_service_lvl / 100.0)

# #             st.write("<br>", unsafe_allow_html=True)
# #             policy_type = st.radio("Select Inventory Control Policy to Backtest:", ["Continuous Review (Reorder Point & Fixed Qty)", "Periodic Review (Fixed Interval & Target Level)"], horizontal=True)
            
# #             pol_c1, pol_c2, pol_c3 = st.columns(3)
            
# #             if "Continuous" in policy_type:
# #                 with pol_c1:
# #                     calc_rop = (mean_dem * bt_lead_time) + (bt_z_score * std_dem * np.sqrt(bt_lead_time))
# #                     rop = st.number_input("Reorder Point (ROP)", value=int(calc_rop), step=10)
# #                 with pol_c2:
# #                     eoq_est = np.sqrt((2 * (mean_dem*365) * bt_order_cost) / (bt_unit_cost * (bt_hold_pct/100))) if bt_hold_pct > 0 else 100
# #                     order_q = st.number_input("Order Quantity (Q)", value=int(eoq_est) if eoq_est > 0 else 100, step=10)
# #                 with pol_c3:
# #                     start_inv = st.number_input("Starting Inventory", value=int(df_bt['Closing Balance'].iloc[0]) if hist_has_inv else int(rop + order_q/2), step=10)
# #             else:
# #                 with pol_c1:
# #                     review_t = st.number_input("Review Period (Days)", value=14, min_value=1, step=1)
# #                 with pol_c2:
# #                     pi = review_t + bt_lead_time
# #                     calc_targ = (mean_dem * pi) + (bt_z_score * std_dem * np.sqrt(pi))
# #                     target_lvl = st.number_input("Target Level (Order-Up-To)", value=int(calc_targ), step=10)
# #                 with pol_c3:
# #                     start_inv = st.number_input("Starting Inventory", value=int(df_bt['Closing Balance'].iloc[0]) if hist_has_inv else int(target_lvl), step=10)

# #             # --- 3. Simulation Engine ---
# #             demand_arr = df_bt['Demand'].values
# #             sim_len = len(demand_arr)
            
# #             inv_history = np.zeros(sim_len)
# #             receipts = np.zeros(sim_len + bt_lead_time + 1)
# #             fulfilled_arr = np.zeros(sim_len)
            
# #             current_inv = float(start_inv)
# #             orders_placed = 0
# #             units_fulfilled = 0
            
# #             for day in range(sim_len):
# #                 current_inv += receipts[day]
                
# #                 fulfilled = min(max(current_inv, 0), demand_arr[day])
# #                 current_inv -= fulfilled
# #                 units_fulfilled += fulfilled
# #                 fulfilled_arr[day] = fulfilled
# #                 inv_history[day] = current_inv
                
# #                 if "Continuous" in policy_type:
# #                     inv_pos = current_inv + np.sum(receipts[day+1:day+bt_lead_time+1])
# #                     if inv_pos <= rop:
# #                         receipts[day + bt_lead_time] += order_q
# #                         orders_placed += 1
# #                 else:
# #                     if day % review_t == 0:
# #                         inv_pos = current_inv + np.sum(receipts[day+1:day+bt_lead_time+1])
# #                         if inv_pos < target_lvl:
# #                             order_qty = target_lvl - inv_pos
# #                             receipts[day + bt_lead_time] += order_qty
# #                             orders_placed += 1

# #             # Simulated KPIs
# #             total_dem = np.sum(demand_arr)
# #             sim_lost_sales = total_dem - units_fulfilled
# #             sim_fill_rate = (units_fulfilled / total_dem) * 100 if total_dem > 0 else 0
            
# #             sim_max_inv = np.max(inv_history)
# #             sim_min_inv = np.min(inv_history)
# #             sim_avg_inv = np.mean(inv_history)
# #             sim_days_at_zero = np.sum(inv_history <= 0)
            
# #             sim_max_wc = max(sim_max_inv, 0) * bt_unit_cost
# #             sim_avg_wc = max(sim_avg_inv, 0) * bt_unit_cost
            
# #             total_hold_cost = np.sum(np.maximum(inv_history, 0)) * bt_hold_daily
# #             total_ord_cost = orders_placed * bt_order_cost
# #             total_sys_cost = total_hold_cost + total_ord_cost

# #             # Historical KPIs Extraction (If Closing Balance Exists)
# #             hist_max_inv = hist_min_inv = hist_avg_inv = hist_orders = hist_hold_cost = hist_ord_cost = hist_total_cost = hist_avg_wc = hist_max_wc = hist_days_at_zero = 0
# #             if hist_has_inv:
# #                 hist_inv_arr = df_bt['Closing Balance'].values
# #                 hist_max_inv = np.max(hist_inv_arr)
# #                 hist_min_inv = np.min(hist_inv_arr)
# #                 hist_avg_inv = np.mean(hist_inv_arr)
# #                 hist_days_at_zero = np.sum(hist_inv_arr <= 0)
                
# #                 hist_max_wc = max(hist_max_inv, 0) * bt_unit_cost
# #                 hist_avg_wc = max(hist_avg_inv, 0) * bt_unit_cost
# #                 hist_hold_cost = np.sum(np.maximum(hist_inv_arr, 0)) * bt_hold_daily
                
# #                 # Estimate Historical Orders (Checks for 'Receipts' or 'Order Placed' columns)
# #                 order_cols = [c for c in df_bt.columns if 'Receipt' in c or 'Order Placed' in c]
# #                 if order_cols:
# #                     hist_orders = (pd.to_numeric(df_bt[order_cols[0]], errors='coerce').fillna(0) > 0).sum()
# #                 else:
# #                     inferred_receipts = np.diff(hist_inv_arr, prepend=start_inv) + demand_arr
# #                     hist_orders = (inferred_receipts > 5).sum() 
                    
# #                 hist_ord_cost = hist_orders * bt_order_cost
# #                 hist_total_cost = hist_hold_cost + hist_ord_cost

# #             # --- 4. Comparative Matrices ---
# #             st.divider()
# #             st.subheader("3. Strategy Performance & Comparative Matrices")
            
# #             def f_usd(v): return f"${v:,.2f}"
# #             def f_unit(v): return f"{int(v)} Units"
# #             def f_day(v): return f"{int(v)} Days"
            
# #             def diff_unit(sim, hist):
# #                 d = sim - hist
# #                 return f"+{int(d)} Units" if d > 0 else f"{int(d)} Units"
# #             def diff_usd(sim, hist):
# #                 d = sim - hist
# #                 return f"+${d:,.2f}" if d > 0 else f"-${abs(d):,.2f}"
# #             def diff_pct(sim, hist):
# #                 d = sim - hist
# #                 return f"+{d:.2f}%" if d > 0 else f"{d:.2f}%"
# #             def diff_day(sim, hist):
# #                 d = sim - hist
# #                 return f"+{int(d)} Days" if d > 0 else f"{int(d)} Days"
            
# #             # Render Comparative Tables if historical inventory data exists
# #             if hist_has_inv:
# #                 st.markdown("#### A. Operational & Inventory Health")
# #                 ops_df = pd.DataFrame({
# #                     "Metric": ["Max Inventory", "Average Inventory", "Minimum Inventory (Depth)", "Days at Zero Inventory", "Missed Sales (Lost)", "Fill Rate (%)"],
# #                     "Historical Actuals": [f_unit(hist_max_inv), f_unit(hist_avg_inv), f_unit(hist_min_inv), f_day(hist_days_at_zero), "0 Units (Assumed)", "100.00% (Assumed)"],
# #                     "Simulated Policy": [f_unit(sim_max_inv), f_unit(sim_avg_inv), f_unit(sim_min_inv), f_day(sim_days_at_zero), f_unit(sim_lost_sales), f"{sim_fill_rate:.2f}%"],
# #                     "Delta (Sim - Hist)": [diff_unit(sim_max_inv, hist_max_inv), diff_unit(sim_avg_inv, hist_avg_inv), diff_unit(sim_min_inv, hist_min_inv), diff_day(sim_days_at_zero, hist_days_at_zero), diff_unit(sim_lost_sales, 0), diff_pct(sim_fill_rate, 100.0)]
# #                 })
# #                 st.dataframe(ops_df, use_container_width=True, hide_index=True)
                    
# #                 st.write("<br>", unsafe_allow_html=True)
# #                 st.markdown("#### B. Capital & Financial Impact")
# #                 fin_df = pd.DataFrame({
# #                     "Metric": ["Average Working Capital", "Max Working Capital", "Total Ordering Cost", "Total Holding Cost", "Total System Cost"],
# #                     "Historical Actuals": [f_usd(hist_avg_wc), f_usd(hist_max_wc), f_usd(hist_ord_cost), f_usd(hist_hold_cost), f_usd(hist_total_cost)],
# #                     "Simulated Policy": [f_usd(sim_avg_wc), f_usd(sim_max_wc), f_usd(total_ord_cost), f_usd(total_hold_cost), f_usd(total_sys_cost)],
# #                     "Delta (Sim - Hist)": [diff_usd(sim_avg_wc, hist_avg_wc), diff_usd(sim_max_wc, hist_max_wc), diff_usd(total_ord_cost, hist_ord_cost), diff_usd(total_hold_cost, hist_hold_cost), diff_usd(total_sys_cost, hist_total_cost)]
# #                 })
# #                 st.dataframe(fin_df, use_container_width=True, hide_index=True)
# #             else:
# #                 # Fallback if no historical closing balance was provided
# #                 res_c1, res_c2, res_c3, res_c4 = st.columns(4)
# #                 res_c1.metric("Total System Cost", f_usd(total_sys_cost))
# #                 res_c2.metric("Fill Rate (%)", f"{sim_fill_rate:.2f}%", delta=f"{int(sim_lost_sales)} Units Lost", delta_color="inverse" if sim_lost_sales > 0 else "normal")
# #                 res_c3.metric("Holding Cost (Capital Blocked)", f_usd(total_hold_cost))
# #                 res_c4.metric("Ordering Cost (Logistics)", f_usd(total_ord_cost), delta=f"{orders_placed} Total Orders", delta_color="off")

# #             # --- 5. Visualizations ---
# #             st.write("<br>", unsafe_allow_html=True)
# #             st.markdown("#### 📉 Simulated Demand vs. Inventory Trajectory Overlay")
# #             st.caption("Visualizes your actual daily demand (bars) against the NEW resulting inventory levels (line) based on the policy above.")
            
# #             fig_close = go.Figure()
            
# #             fig_close.add_trace(go.Bar(
# #                 x=x_axis_hist, y=df_bt['Demand'], name='Actual Demand (Units)', marker_color='rgba(156, 163, 175, 0.5)'
# #             ))
            
# #             fig_close.add_trace(go.Scatter(
# #                 x=x_axis_hist, y=inv_history, mode='lines', name='Simulated Closing Stock',
# #                 line=dict(color='#2ca02c', width=2), fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.15)'
# #             ))
            
# #             fig_close.add_hline(y=0, line_dash="solid", line_color="#d62728", line_width=1.5)
            
# #             if "Continuous" in policy_type:
# #                 fig_close.add_hline(y=rop, line_dash="dash", line_color="#1f77b4", annotation_text="Reorder Point (ROP)", annotation_position="top left")
# #             else:
# #                 fig_close.add_hline(y=target_lvl, line_dash="dash", line_color="#1f77b4", annotation_text="Target Level", annotation_position="top left")

# #             fig_close.update_layout(template="plotly_white", xaxis_title="Historical Period", yaxis_title="Quantity (Units)", height=450, margin=dict(t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
# #             st.plotly_chart(fig_close, use_container_width=True)
            
# #             # --- 6. Data Table of the Simulated Policy ---
# #             with st.expander("📋 View Simulated Daily Log Table"):
# #                 st.markdown("Raw day-by-day tracking for Demand, Fulfillment, and Simulated Inventory.")
                
# #                 sim_log_df = pd.DataFrame({
# #                     "Period": x_axis_hist,
# #                     "Demand": demand_arr.astype(int),
# #                     "Units Fulfilled": fulfilled_arr.astype(int),
# #                     "Simulated Receipts": receipts[:sim_len].astype(int),
# #                     "Simulated Closing Stock": inv_history.astype(int)
# #                 })
                
# #                 def highlight_stockouts(val):
# #                     color = '#ffcccc' if isinstance(val, (int, float)) and val < 0 else ''
# #                     return f'background-color: {color}'
                    
# #                 st.dataframe(sim_log_df.style.map(highlight_stockouts, subset=['Simulated Closing Stock']), use_container_width=True, hide_index=True)

# #         except Exception as e:
# #             st.error(f"❌ An error occurred while processing the file: {e}")

# # with tab5:
# #     st.header("⚖️ Advanced Inventory Optimization Suite")
# #     st.markdown(
# #         "Analyze your inventory data through a twin-lens framework. First, review a historical backtest audit "
# #         "to identify legacy profit leaks."
# #     )
    
# #     # --- 🚀 THE VECTORIZED SIMULATION ENGINE ---
# #     def fast_simulate_inventory(demand_arr, purchase_arr, opening_stock, lead_time, policy_type, param1, param2):
# #         total_days = len(demand_arr)
# #         inv_levels = np.zeros(total_days)
# #         lost_sales = np.zeros(total_days)
# #         orders_placed = np.zeros(total_days)
        
# #         max_lt = int(lead_time)
# #         pipeline = np.zeros(total_days + max_lt + 1)
        
# #         if policy_type == "Actual":
# #             pipeline[:total_days] = purchase_arr
# #         else:
# #             warm_up = min(max_lt, total_days)
# #             pipeline[:warm_up] = purchase_arr[:warm_up]
            
# #         current_inv = opening_stock
        
# #         for i in range(total_days):
# #             demand = demand_arr[i]
# #             arriving = pipeline[i]
# #             current_inv += arriving
            
# #             if current_inv < demand:
# #                 lost_sales[i] = demand - current_inv
# #                 current_inv = 0
# #             else:
# #                 current_inv -= demand
                
# #             inv_levels[i] = current_inv
            
# #             if policy_type == "Continuous Review (Q, R)":
# #                 net_position = current_inv + np.sum(pipeline[i+1:])
# #                 if net_position <= param2:
# #                     pipeline[i + max_lt] += param1
# #                     orders_placed[i] = param1
# #             elif policy_type == "Periodic Review (P, T)":
# #                 if i % int(param1) == 0:
# #                     net_position = current_inv + np.sum(pipeline[i+1:])
# #                     order_qty = max(0, param2 - net_position)
# #                     if order_qty > 0:
# #                         pipeline[i + max_lt] += order_qty
# #                         orders_placed[i] = order_qty
                        
# #         return inv_levels, lost_sales, orders_placed

# #     # --- STEP 1: INPUT PARAMETERS ---
# #     st.subheader("1. Parameters & Cost Drivers")
    
# #     col1, col2, col3 = st.columns(3)
    
# #     with col1:
# #         item_unit_cost = st.number_input("Item Unit Cost ($/Unit)", min_value=0.01, value=25.00, step=1.00, key="unit_cost_global")
# #         holding_fixed_daily = st.number_input("Fixed Holding Cost ($/Unit/Day)", min_value=0.0, value=0.0, step=0.01, key="fixed_hold_global")
        
# #     with col2:
# #         holding_var_pct = st.number_input("Variable Holding Cost (% of Item Cost/year)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="var_hold_global") / 100.0
# #         ordering_cost = st.number_input("Ordering Cost ($/order)", min_value=0.1, value=75.0, step=5.0, key="order_cost_global")
        
# #     with col3:
# #         lost_sales_penalty = st.number_input("Lost Sales Penalty ($/Unit Lost)", min_value=0.0, value=0.0, step=1.0, key="penalty_global")
# #         lead_time_days = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1, key="lt_global")

# #     st.markdown("---")
# #     col_sys1, col_sys2 = st.columns([1, 2])
# #     with col_sys1:
# #         review_system = st.radio("Inventory Review System Strategy", ["Continuous Review (Q, R)", "Periodic Review (P, T)"], key="review_system_global")
# #     with col_sys2:
# #         service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="service_level_global") / 100.0

# #     if review_system == "Periodic Review (P, T)":
# #         st.markdown("##### ⏳ Periodic Configuration")
# #         p_col1, _ = st.columns(2)
# #         with p_col1:
# #             user_p_days = st.number_input("Review Period Cycle (P in Days)", min_value=1, value=14, step=1, key="p_days_global")
# #     else:
# #         user_p_days = 1

# #     # --- STEP 2: MULTI-FORMAT DATA INGESTION ENGINE ---
# #     st.subheader("2. Upload Historical Invoices & Demand Data")
# #     uploaded_file = st.file_uploader(
# #         "Upload Inventory Ledger (Supports standard templates, raw ERP transactional logs, or stock card snapshots)", 
# #         type=["csv", "xlsx", "xls"], 
# #         key="uploader_global"
# #     )
    
# #     if uploaded_file is None:
# #         st.info("📥 Please upload your inventory ledger file (CSV or Excel) above to populate the suite modules.")
# #     else:
# #         detected_sheet_opening_stock = None
# #         data_loaded_successfully = False
# #         df_mapped = None
        
# #         try:
# #             if uploaded_file.name.endswith('.csv'):
# #                 raw_df = pd.read_csv(uploaded_file)
# #             else:
# #                 raw_df = pd.read_excel(uploaded_file)
                
# #             raw_df.columns = raw_df.columns.str.strip()
                
# #             if "Date" not in raw_df.columns:
# #                 st.error("❌ Missing required column: 'Date'.")
# #             else:
# #                 raw_df["Date"] = pd.to_datetime(raw_df["Date"])
# #                 raw_df = raw_df.sort_values(by="Date").reset_index(drop=True)
                
# #                 open_balance_headers = ["Opening Balance", "Opening", "Opening_Stock", "Opening Stock"]
# #                 for header in open_balance_headers:
# #                     if header in raw_df.columns:
# #                         detected_sheet_opening_stock = int(raw_df[header].iloc[0])
# #                         break
                
# #                 if "Demand_Qty" in raw_df.columns and "Purchase_Qty" in raw_df.columns:
# #                     df_mapped = raw_df[["Date", "Demand_Qty", "Purchase_Qty"]].copy()
# #                 elif "Demand" in raw_df.columns and "Stock Received" in raw_df.columns:
# #                     df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df["Demand"], "Purchase_Qty": raw_df["Stock Received"]})
# #                 elif ("Receiving" in raw_df.columns) and any(col in raw_df.columns for col in ["Demand/Sales", "Demand", "Sales"]):
# #                     outbound_col = "Demand/Sales" if "Demand/Sales" in raw_df.columns else ("Demand" if "Demand" in raw_df.columns else "Sales")
# #                     df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df[outbound_col], "Purchase_Qty": raw_df["Receiving"]})
# #                 else:
# #                     st.error("❌ Column layout structure mismatch. Could not find Demand and Receiving columns.")

# #                 if df_mapped is not None:
# #                     df_mapped = df_mapped.groupby("Date").agg({"Demand_Qty": "sum", "Purchase_Qty": "sum"}).reset_index()
# #                     df_mapped = df_mapped.set_index("Date").resample("1D").asfreq()
# #                     df_mapped["Demand_Qty"] = df_mapped["Demand_Qty"].fillna(0.0)
# #                     df_mapped["Purchase_Qty"] = df_mapped["Purchase_Qty"].fillna(0.0)
# #                     df_mapped = df_mapped.reset_index()
# #                     data_loaded_successfully = True
                    
# #         except Exception as e:
# #             st.error(f"Error parsing file elements: {e}")

# #         # ONLY PROCEED IF DATA IS CLEANED
# #         if data_loaded_successfully and df_mapped is not None:
# #             full_df = df_mapped.copy() 
            
# #             absolute_min_date = full_df["Date"].min().date()
# #             absolute_max_date = full_df["Date"].max().date()
            
# #             file_state_key = f"last_file_{uploaded_file.name}_{uploaded_file.size}"
            
# #             if "current_file_token" not in st.session_state or st.session_state.current_file_token != file_state_key:
# #                 st.session_state.current_file_token = file_state_key
# #                 st.session_state.min_date_global = absolute_min_date
# #                 st.session_state.max_date_global = absolute_max_date
                
# #                 avg_daily_full = full_df["Demand_Qty"].mean()
# #                 if detected_sheet_opening_stock is not None:
# #                     default_start = int(detected_sheet_opening_stock)
# #                 else:
# #                     default_start = int(1.25 * (avg_daily_full * lead_time_days))
                
# #                 st.session_state.absolute_day1_stock = default_start
# #                 st.session_state.previous_start_date = absolute_min_date
# #                 st.session_state.previous_end_date = absolute_max_date
# #                 st.session_state.opening_stock_global = default_start
                
# #                 st.session_state.start_date_key = absolute_min_date
# #                 st.session_state.end_date_key = absolute_max_date
                
# #                 if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
# #                 if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite

# #             st.divider()
            
# #             st.markdown("### 📅 3. Select Analysis Period")
# #             st.markdown("Filter historical data. The starting inventory will automatically mathematically roll forward to match your selected Start Date.")
            
# #             def reset_dates():
# #                 st.session_state.start_date_key = st.session_state.min_date_global
# #                 st.session_state.end_date_key = st.session_state.max_date_global
# #                 st.session_state.previous_start_date = st.session_state.min_date_global
# #                 st.session_state.previous_end_date = st.session_state.max_date_global
# #                 st.session_state.opening_stock_global = st.session_state.absolute_day1_stock

# #             col_date1, col_date2, col_date3 = st.columns([2, 2, 1])
            
# #             with col_date1:
# #                 start_date = st.date_input("Starting Date", min_value=absolute_min_date, max_value=absolute_max_date, key="start_date_key")
# #             with col_date2:
# #                 end_date = st.date_input("Ending Date", min_value=absolute_min_date, max_value=absolute_max_date, key="end_date_key")
# #             with col_date3:
# #                 st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
# #                 st.button("🔄 Reset", on_click=reset_dates, use_container_width=True)
            
# #             if "previous_start_date" not in st.session_state:
# #                 st.session_state.previous_start_date = absolute_min_date
# #             if "previous_end_date" not in st.session_state:
# #                 st.session_state.previous_end_date = absolute_max_date

# #             date_changed = (start_date != st.session_state.previous_start_date) or (end_date != st.session_state.previous_end_date)

# #             if date_changed:
# #                 if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
# #                 if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite
                
# #                 if start_date != st.session_state.previous_start_date:
# #                     temp_balance = st.session_state.absolute_day1_stock
# #                     pre_period_df = full_df[full_df["Date"].dt.date < start_date]
                    
# #                     pre_d_arr = pre_period_df["Demand_Qty"].values
# #                     pre_p_arr = pre_period_df["Purchase_Qty"].values
# #                     for idx in range(len(pre_d_arr)):
# #                         temp_balance = max(0, temp_balance + pre_p_arr[idx] - pre_d_arr[idx])
                    
# #                     st.session_state.opening_stock_global = int(temp_balance)
                
# #                 st.session_state.previous_start_date = start_date
# #                 st.session_state.previous_end_date = end_date

# #             if start_date > end_date:
# #                 st.error("⚠️ The Starting Date must be before or equal to the Ending Date. Please adjust your selection.")
# #             else:
# #                 df = full_df[(full_df["Date"].dt.date >= start_date) & (full_df["Date"].dt.date <= end_date)].reset_index(drop=True)
                
# #                 if df.empty:
# #                     st.warning("No data available for the selected date range. Please widen your selection.")
# #                 else:
# #                     demand_arr_main = df["Demand_Qty"].values
# #                     purchase_arr_main = df["Purchase_Qty"].values
                    
# #                     actual_orders_placed = np.count_nonzero(purchase_arr_main)
# #                     actual_total_units_purchased = purchase_arr_main.sum()
# #                     total_demand = demand_arr_main.sum()
                    
# #                     avg_daily_demand_calc = demand_arr_main.mean()
# #                     std_daily_demand = demand_arr_main.std() if len(df) > 1 else 0
# #                     cov = std_daily_demand / max(0.1, avg_daily_demand_calc)

# #                     with st.expander("📈 View Historical Demand Trend & Growth Timeline", expanded=False):
# #                         rolling_days = st.slider("Select Rolling Average Window (Days)", min_value=1, max_value=90, value=15, step=1)
# #                         df[f"Rolling_Avg"] = df["Demand_Qty"].rolling(window=rolling_days, min_periods=1).mean()
                        
# #                         stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
# #                         if len(df) >= rolling_days * 2:
# #                             current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
# #                             previous_window_sum = df["Demand_Qty"].iloc[-(rolling_days*2):-rolling_days].sum()
# #                             if previous_window_sum > 0:
# #                                 trend_pct = ((current_window_sum - previous_window_sum) / previous_window_sum) * 100
# #                             else:
# #                                 trend_pct = 100.0 if current_window_sum > 0 else 0.0
# #                             stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta=f"{trend_pct:+.1f}% Growth", delta_color="normal")
# #                         elif len(df) >= rolling_days:
# #                             current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
# #                             stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta="Widen dates for trend", delta_color="off")
# #                         else:
# #                             stat_col1.metric(label="Trend", value="N/A", delta="Insufficient Data", delta_color="off")
                            
# #                         stat_col2.metric("Average Daily Demand", f"{avg_daily_demand_calc:.1f} units")
# #                         stat_col3.metric("Standard Deviation", f"{std_daily_demand:.1f} units")
# #                         stat_col4.metric("Volatility (CoV)", f"{cov:.2f}")
# #                         st.markdown("---")

# #                         demand_fig = go.Figure()
# #                         demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Demand_Qty"], mode='lines', name='Raw Daily Demand', line=dict(color='#B0C4DE', width=1.5), opacity=0.6))
# #                         demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Rolling_Avg"], mode='lines', name=f'{rolling_days}-Day Moving Avg', line=dict(color='#3d5a80', width=3)))
# #                         demand_fig.update_layout(template="plotly_white", yaxis_title="Units", xaxis_title="Date", margin=dict(t=20, b=20), height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
# #                         st.plotly_chart(demand_fig, use_container_width=True)

# #                     st.markdown("---")
# #                     st.markdown("##### 📦 Initial Warehouse Capital Balance")
# #                     opening_stock_override = st.number_input("Starting Balance for Selected Period", min_value=0, step=10, key="opening_stock_global", help="This value automatically updates mathematically based on the start date you select above, but you can override it manually.")

# #                     with st.expander("📋 View Complete Running Balance Table Snapshots", expanded=False):
# #                         st.markdown("An interactive historical stock card ledger driven directly by your initial opening stock parameter above.")
# #                         cl_open_list, cl_close_list, t_bal = np.zeros(len(df)), np.zeros(len(df)), opening_stock_override
                        
# #                         for i_run in range(len(df)):
# #                             cl_open_list[i_run] = t_bal
# #                             t_bal = max(0, t_bal + purchase_arr_main[i_run] - demand_arr_main[i_run])
# #                             cl_close_list[i_run] = t_bal
                            
# #                         full_stock_card_df = pd.DataFrame({
# #                             "Timeline Date": df["Date"].dt.strftime('%Y-%m-%d'),
# #                             "Opening Balance": cl_open_list.astype(int),
# #                             "Cleaned Demand Volume (Units)": demand_arr_main.astype(int),
# #                             "Consolidated Stock Received (Units)": purchase_arr_main.astype(int),
# #                             "Closing Balance": cl_close_list.astype(int)
# #                         })
                        
# #                         st.dataframe(full_stock_card_df, use_container_width=True, hide_index=True, column_config={"Opening Balance": st.column_config.NumberColumn(format="%d"), "Cleaned Demand Volume (Units)": st.column_config.NumberColumn(format="%d"), "Consolidated Stock Received (Units)": st.column_config.NumberColumn(format="%d"), "Closing Balance": st.column_config.NumberColumn(format="%d")})

# #                     # --- STEP 4: ADVANCED STATISTICAL FIT RUNNER ---
# #                     st.subheader("4. Statistical Risk & Distribution Engines")
                    
# #                     annual_demand = avg_daily_demand_calc * 365
# #                     annual_fixed_holding_per_unit = holding_fixed_daily * 365
# #                     unit_holding_cost = annual_fixed_holding_per_unit + (item_unit_cost * holding_var_pct)
                    
# #                     risk_horizon_days = lead_time_days if review_system == "Continuous Review (Q, R)" else (user_p_days + lead_time_days)
# #                     rolling_risk_demand = df["Demand_Qty"].rolling(window=int(risk_horizon_days)).sum().dropna().values
# #                     risk_mean = np.mean(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0
# #                     risk_std = np.std(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0

# #                     if len(rolling_risk_demand) > 0:
# #                         if np.max(rolling_risk_demand) <= 0:
# #                             best_fit_name = "Zero Demand Base"
# #                             raw_target_level = 0.0
# #                         else:
# #                             empirical_rop_raw = np.percentile(rolling_risk_demand, service_level * 100)
# #                             safe_demand = np.where(rolling_risk_demand <= 0, 1e-5, rolling_risk_demand)
                            
# #                             log_params = stats.lognorm.fit(safe_demand, floc=0)
# #                             gam_params = stats.gamma.fit(safe_demand, floc=0)

# #                             counts, bins = np.histogram(rolling_risk_demand, bins=20, density=True)
# #                             bin_centers = (bins[:-1] + bins[1:]) / 2
                            
# #                             rss_norm = np.sum((counts - stats.norm.pdf(bin_centers, loc=risk_mean, scale=risk_std)) ** 2)
# #                             rss_log = np.sum((counts - stats.lognorm.pdf(bin_centers, *log_params)) ** 2)
# #                             rss_gam = np.sum((counts - stats.gamma.pdf(bin_centers, *gam_params)) ** 2)

# #                             if cov > 0.75:
# #                                 best_fit_name = "Empirical (Data-Driven)"
# #                                 raw_target_level = empirical_rop_raw
# #                             else:
# #                                 errors = {"Normal": rss_norm, "Log-Normal": rss_log, "Gamma": rss_gam}
# #                                 best_fit_name = min(errors, key=errors.get)
# #                                 if best_fit_name == "Normal":
# #                                     raw_target_level = stats.norm.ppf(service_level, loc=risk_mean, scale=risk_std)
# #                                 elif best_fit_name == "Log-Normal":
# #                                     raw_target_level = stats.lognorm.ppf(service_level, *log_params)
# #                                 else:
# #                                     raw_target_level = stats.gamma.ppf(service_level, *gam_params)
# #                     else:
# #                         best_fit_name = "Default (Insufficient Data)"
# #                         raw_target_level = avg_daily_demand_calc * risk_horizon_days
                        
# #                     raw_optimal_q = np.sqrt((2 * annual_demand * ordering_cost) / max(0.01, unit_holding_cost))

# #                     if "q_audit_suite" not in st.session_state:
# #                         st.session_state.q_audit_suite = max(1, int(raw_optimal_q)) if review_system == "Continuous Review (Q, R)" else int(avg_daily_demand_calc * user_p_days)
# #                     if "rop_audit_suite" not in st.session_state:
# #                         st.session_state.rop_audit_suite = max(0, int(raw_target_level))

# #                     with st.expander("📊 View Cleaned Demand Distribution & Best-Fit Curve Metrics", expanded=False):
# #                         stat_col1, stat_col2, stat_col3 = st.columns(3)
# #                         stat_col1.metric("Average Daily Demand", f"{avg_daily_demand_calc:.2f} units")
# #                         stat_col2.metric("Coefficient of Variation (CV)", f"{cov:.2f}")
# #                         stat_col3.metric("Engine Selection", f"✨ {best_fit_name}")
# #                         st.markdown("---")
# #                         hist_fig = go.Figure()
# #                         hist_fig.add_trace(go.Histogram(x=df["Demand_Qty"], name="Historical Days", marker_color='#1F77B4', opacity=0.6))
# #                         hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Demand Quantity (Units / Day)", yaxis_title="Frequency", margin=dict(l=40, r=40, t=10, b=40), height=300)
# #                         st.plotly_chart(hist_fig, use_container_width=True)

# #                     # ==========================================
# #                     #      SECTION C: ERRATIC DEMAND PROFILER
# #                     # ==========================================
# #                     st.markdown("---")
# #                     st.subheader("5. 🌪️ Erratic Demand & Empirical ROP Profiler")
                    
# #                     with st.expander("Open Rolling Window & Empirical Profiler", expanded=False):
# #                         st.markdown(
# #                             "Standard safety stock math assumes demand follows a predictable bell curve. For erratic or lumpy demand, "
# #                             "that assumption breaks down. This tool mechanically profiles your exact historical risk by analyzing every "
# #                             "rolling vulnerability window in your dataset."
# #                         )
                        
# #                         tab_emp_cont, tab_emp_per = st.tabs(["📉 Continuous Review (ROP)", "⏳ Periodic Review (Target Level)"])
                        
# #                         with tab_emp_cont:
# #                             st.markdown("##### Continuous Review Risk Profiler")
# #                             st.write("In a Continuous system, you reorder immediately when you hit a threshold. Your vulnerability window is simply the **Lead Time**.")
                            
# #                             col_emp1, col_emp2, col_emp3 = st.columns(3)
                            
# #                             with col_emp1:
# #                                 emp_lt_window = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_lt_window")
# #                             with col_emp2:
# #                                 emp_test_rop = st.number_input("Test Reorder Point (ROP)", min_value=0, value=int(raw_target_level), step=10, key="emp_test_rop")
# #                             with col_emp3:
# #                                 emp_target_sl = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl")

# #                             rolling_demand_c = df["Demand_Qty"].rolling(window=emp_lt_window).sum().dropna()

# #                             if len(rolling_demand_c) > 0:
# #                                 windows_below_rop = np.sum(rolling_demand_c <= emp_test_rop)
# #                                 total_windows_c = len(rolling_demand_c)
# #                                 achieved_sl_c = (windows_below_rop / total_windows_c) * 100
# #                                 required_rop = np.percentile(rolling_demand_c, emp_target_sl)

# #                                 res_col1, res_col2 = st.columns(2)
# #                                 with res_col1:
# #                                     st.info(f"**Testing ROP of {emp_test_rop:,}:**\n\nOut of {total_windows_c:,} historical {emp_lt_window}-day windows, the total demand was successfully covered by {emp_test_rop:,} units exactly **{windows_below_rop:,} times**. This yields an empirical service level of **{achieved_sl_c:.1f}%**.")
# #                                 with res_col2:
# #                                     st.success(f"**Targeting {emp_target_sl}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl}% of all historical {emp_lt_window}-day scenarios, your ROP must be set to the empirical percentile: **{int(required_rop):,} units**.")

# #                                 emp_fig_c = go.Figure()
# #                                 emp_fig_c.add_trace(go.Histogram(x=rolling_demand_c, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {emp_lt_window}-Day Windows"))
# #                                 emp_fig_c.add_vline(x=emp_test_rop, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested ROP ({emp_test_rop})", annotation_position="top right")
# #                                 emp_fig_c.add_vline(x=required_rop, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Target ROP ({int(required_rop)})", annotation_position="top left")

# #                                 emp_fig_c.update_layout(
# #                                     title=f"Actual Demand Distribution Across All {emp_lt_window}-Day Windows",
# #                                     xaxis_title=f"Total Units Demanded in a {emp_lt_window}-Day Window",
# #                                     yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40)
# #                                 )
# #                                 st.plotly_chart(emp_fig_c, use_container_width=True)
# #                             else:
# #                                 st.warning(f"⚠️ Not enough data to calculate rolling windows for a {emp_lt_window}-day lead time.")

# #                         with tab_emp_per:
# #                             st.markdown("##### Periodic Review Risk Profiler")
# #                             st.write("In a Periodic system, you are blind between cycle counts. Your vulnerability window is the **Lead Time + the Review Period**.")
                            
# #                             default_p_val = int(user_p_days) if user_p_days > 1 else 14
# #                             default_t_val = int(raw_target_level + (avg_daily_demand_calc * default_p_val))
                            
# #                             p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                            
# #                             with p_col1:
# #                                 emp_p_days = st.number_input("Review Period (P)", min_value=1, value=default_p_val, step=1, key="emp_p_days")
# #                             with p_col2:
# #                                 emp_p_lt = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_p_lt")
# #                             with p_col3:
# #                                 emp_test_t = st.number_input("Test Target Level (T)", min_value=0, value=default_t_val, step=10, key="emp_test_t")
# #                             with p_col4:
# #                                 emp_target_sl_p = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl_p")

# #                             risk_window_days = emp_p_days + emp_p_lt
# #                             rolling_demand_p = df["Demand_Qty"].rolling(window=risk_window_days).sum().dropna()

# #                             if len(rolling_demand_p) > 0:
# #                                 windows_below_t = np.sum(rolling_demand_p <= emp_test_t)
# #                                 total_windows_p = len(rolling_demand_p)
# #                                 achieved_sl_p = (windows_below_t / total_windows_p) * 100
# #                                 required_t = np.percentile(rolling_demand_p, emp_target_sl_p)

# #                                 res_col3, res_col4 = st.columns(2)
# #                                 with res_col3:
# #                                     st.info(f"**Testing Target (T) of {emp_test_t:,}:**\n\nOut of {total_windows_p:,} historical {risk_window_days}-day windows (P+L), demand was successfully covered by {emp_test_t:,} units exactly **{windows_below_t:,} times**. Empirical SL: **{achieved_sl_p:.1f}%**.")
# #                                 with res_col4:
# #                                     st.success(f"**Targeting {emp_target_sl_p}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl_p}% of all historical {risk_window_days}-day scenarios, your Target Level must be: **{int(required_t):,} units**.")

# #                                 emp_fig_p = go.Figure()
# #                                 emp_fig_p.add_trace(go.Histogram(x=rolling_demand_p, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {risk_window_days}-Day Windows"))
# #                                 emp_fig_p.add_vline(x=emp_test_t, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested Target ({emp_test_t})", annotation_position="top right")
# #                                 emp_fig_p.add_vline(x=required_t, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Required Target ({int(required_t)})", annotation_position="top left")

# #                                 emp_fig_p.update_layout(
# #                                     title=f"Actual Demand Distribution Across All {risk_window_days}-Day (P+L) Windows",
# #                                     xaxis_title=f"Total Units Demanded in a {risk_window_days}-Day Window",
# #                                     yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40)
# #                                 )
# #                                 st.plotly_chart(emp_fig_p, use_container_width=True)
# #                             else:
# #                                 st.warning(f"⚠️ Not enough data to calculate rolling windows for a {risk_window_days}-day (P+L) window.")

# #                     # --- STEP 5: FINAL POLICY OPTIMIZATION TUNING CONFIG ---
# #                     st.markdown("---")
# #                     st.subheader("6. Final Policy Parameter Tuning")
# #                     adjust_col1, adjust_col2 = st.columns(2)
# #                     with adjust_col1:
# #                         if review_system == "Continuous Review (Q, R)":
# #                             final_q = st.number_input("Target Order Quantity (Q)", min_value=1, step=10, key="q_audit_suite")
# #                         else:
# #                             cycle_demand_baseline = int(avg_daily_demand_calc * user_p_days)
# #                             final_q = st.number_input("Average Target Batch Size (Q)", min_value=1, value=cycle_demand_baseline, step=10, disabled=True, key="q_audit_suite_disabled")
# #                     with adjust_col2:
# #                         final_buffer_target = st.number_input("Reorder Point (ROP) / Target Level (T)", min_value=0, step=10, key="rop_audit_suite")

# #                     if review_system == "Continuous Review (Q, R)":
# #                         st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Optimal Order Quantity (EOQ): **{int(raw_optimal_q):,}** units | Recommended Reorder Point (ROP): **{int(raw_target_level):,}** units.")
# #                     else:
# #                         st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Expected Cycle Batch Size: **{int(avg_daily_demand_calc * user_p_days):,}** units | Recommended Max Order Up-To Level (T): **{int(raw_target_level):,}** units.")

# #                     optimal_p_days = max(1, int((final_q / max(0.1, avg_daily_demand_calc)))) if review_system == "Continuous Review (Q, R)" else int(user_p_days)

# #                     # ==========================================
# #                     #      SECTION A: HISTORICAL BACKTEST
# #                     # ==========================================
# #                     st.markdown("---")
# #                     st.header("📊 Section A: Historical Backtest Audit")
# #                     st.markdown("This analysis compares your **Historical Actuals** against our **Recommended Optimized Policy** under identical historical demand constraints to reveal operational friction.")
                    
# #                     st.info(f"🔥 **Intelligent Warm Start Active:** To prevent unfair 'Cold Start' stockouts, the optimization engine's initial pipeline has been pre-seeded with your actual historical receipts for the first **{int(lead_time_days)}** days (Lead Time window).")
                    
# #                     inv_levels_act, lost_sales_act_arr, orders_placed_act_arr = fast_simulate_inventory(
# #                         demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, "Actual", 0, 0
# #                     )
                    
# #                     inv_levels_opt, lost_sales_opt_arr, orders_placed_opt_arr = fast_simulate_inventory(
# #                         demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, review_system, 
# #                         optimal_p_days if review_system != "Continuous Review (Q, R)" else final_q, final_buffer_target
# #                     )

# #                     # --- MASTER CALCULATION COMPILING ENGINE ---
# #                     lost_sales_qty_act = lost_sales_act_arr.sum()
# #                     stockout_days_act = np.count_nonzero(lost_sales_act_arr)
# #                     zero_stock_days_act = np.count_nonzero(inv_levels_act == 0)
                    
# #                     actual_max_inventory = np.max(inv_levels_act)
# #                     actual_min_inventory = np.min(inv_levels_act)
# #                     actual_avg_inventory = np.mean(inv_levels_act)
# #                     actual_fill_rate = max(0.0, 1.0 - (lost_sales_qty_act / max(1, total_demand)))
# #                     actual_cycle_time = 365 / actual_orders_placed if actual_orders_placed > 0 else 365.0
# #                     actual_avg_order_size = actual_total_units_purchased / actual_orders_placed if actual_orders_placed > 0 else 0.0

# #                     actual_total_ordering_cost = actual_orders_placed * ordering_cost
# #                     actual_total_holding_cost = actual_avg_inventory * unit_holding_cost
# #                     actual_lost_sales_financial = lost_sales_qty_act * lost_sales_penalty
# #                     actual_total_cost = actual_total_ordering_cost + actual_total_holding_cost + actual_lost_sales_financial

# #                     lost_sales_qty_opt = lost_sales_opt_arr.sum()
# #                     stockout_days_opt = np.count_nonzero(lost_sales_opt_arr)
# #                     zero_stock_days_opt = np.count_nonzero(inv_levels_opt == 0)
# #                     opt_orders_placed = np.count_nonzero(orders_placed_opt_arr)
# #                     policy_total_units_ordered = orders_placed_opt_arr.sum()

# #                     simmed_avg_opt_inv = np.mean(inv_levels_opt)
# #                     simmed_max_opt_inv = np.max(inv_levels_opt)
# #                     simmed_min_inventory = np.min(inv_levels_opt)
# #                     simmed_opt_fill_rate = max(0.0, 1.0 - (lost_sales_qty_opt / max(1, total_demand)))
# #                     policy_cycle_time = 365 / opt_orders_placed if opt_orders_placed > 0 else 365.0
# #                     policy_avg_order_size = policy_total_units_ordered / opt_orders_placed if opt_orders_placed > 0 else 0.0

# #                     optimal_ordering_cost = opt_orders_placed * ordering_cost
# #                     optimal_holding_cost = simmed_avg_opt_inv * unit_holding_cost
# #                     optimal_lost_sales_financial = lost_sales_qty_opt * lost_sales_penalty
# #                     optimal_total_cost = optimal_ordering_cost + optimal_holding_cost + optimal_lost_sales_financial
                    
# #                     act_max_wc = actual_max_inventory * item_unit_cost
# #                     act_avg_wc = actual_avg_inventory * item_unit_cost
# #                     act_min_wc = actual_min_inventory * item_unit_cost
                    
# #                     opt_max_wc = simmed_max_opt_inv * item_unit_cost
# #                     opt_avg_wc = simmed_avg_opt_inv * item_unit_cost
# #                     opt_min_wc = simmed_min_inventory * item_unit_cost

# #                     true_net_benefit = actual_total_cost - optimal_total_cost
                    
# #                     if true_net_benefit > 0:
# #                         st.success(f"### 🎯 The Efficiency Opportunity\nBy shifting to the recommended optimized policy, you would have recovered **${true_net_benefit:,.2f}** over this historical period.")
# #                     else:
# #                         st.error(f"⚠️ **Operational Margin Deficit Risk:** This setup increases operational overhead by **${abs(true_net_benefit):,.2f} / year** compared to actuals.")

# #                     # =========================================================
# #                     # --- EXECUTIVE KPI SCORECARD ---
# #                     # =========================================================
# #                     st.markdown("### 🏆 Executive Summary: Value Realization")
                    
# #                     cash_released = act_avg_wc - opt_avg_wc

# #                     kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# #                     with kpi_col1:
# #                         st.metric(label="Total Cost Saving", value=f"${true_net_benefit:,.0f}")
# #                     with kpi_col2:
# #                         st.metric(label="Optimized Fill Rate", value=f"{simmed_opt_fill_rate * 100:.1f}%")
# #                     with kpi_col3:
# #                         st.metric(label="Avg Working Capital (Opt)", value=f"${opt_avg_wc:,.0f}")
# #                         st.markdown(f"<div style='margin-top: -15px; font-size: 0.85rem; color: gray;'>Historical: ${act_avg_wc:,.0f}</div>", unsafe_allow_html=True)
# #                     with kpi_col4:
# #                         release_label = "Cash Released" if cash_released >= 0 else "Capital Added (Tied Up)"
# #                         st.metric(label=release_label, value=f"${abs(cash_released):,.0f}")

# #                     st.markdown("---")
                    
# #                     def render_clustered_matrix(title, metrics, act_vals, pol_vals, formats):
# #                         st.markdown(f"#### {title}")
# #                         abs_var = [a - p for a, p in zip(act_vals, pol_vals)]
# #                         pct_var = []
# #                         for a, p in zip(act_vals, pol_vals):
# #                             if a == 0: pct_var.append(0.0)
# #                             else: pct_var.append(((a - p) / a) * 100)
                            
# #                         m_df = pd.DataFrame({"Operational Attribute Pillar": metrics})
# #                         for idx in range(len(metrics)):
# #                             fmt = formats[idx]
# #                             if fmt == "currency":
# #                                 m_df.at[idx, "Historical Actuals"] = f"${act_vals[idx]:,.2f}"
# #                                 m_df.at[idx, "Optimized Policy"] = f"${pol_vals[idx]:,.2f}"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"${abs_var[idx]:,.2f}" if abs_var[idx] >= 0 else f"-${abs(abs_var[idx]):,.2f}"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
# #                             elif fmt == "pct":
# #                                 m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:.1f}%"
# #                                 m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:.1f}%"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+.1f}% pts"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pol_vals[idx] - act_vals[idx]:+.1f}% pts"
# #                             elif fmt == "days":
# #                                 m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:,.1f} days"
# #                                 m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:,.1f} days"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+,.1f} days"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
# #                             else:
# #                                 m_df.at[idx, "Historical Actuals"] = f"{int(act_vals[idx]):,}"
# #                                 m_df.at[idx, "Optimized Policy"] = f"{int(pol_vals[idx]):,}"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"{int(abs_var[idx]):+1,}"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"

# #                         def apply_matrix_styles(x):
# #                             colors = pd.DataFrame('', index=x.index, columns=x.columns)
# #                             fav = 'background-color: #1A3E2B; color: #81C784; font-weight: bold;'
# #                             unfav = 'background-color: #3E1A1A; color: #E57373;'
# #                             for i, metric in enumerate(metrics):
# #                                 v = abs_var[i]
# #                                 if title == "1. Financial Breakdown Matrix" or title == "3. Working Capital Release Matrix":
# #                                     if v > 0: colors.iloc[i, 3:] = fav
# #                                     elif v < 0: colors.iloc[i, 3:] = unfav
# #                                 elif title == "4. Stockout Risk & Vulnerability Matrix":
# #                                     if "Fill Rate" in metric:
# #                                         if pol_vals[i] > act_vals[i]: colors.iloc[i, 3:] = fav
# #                                         elif pol_vals[i] < act_vals[i]: colors.iloc[i, 3:] = unfav
# #                                     else:
# #                                         if v > 0: colors.iloc[i, 3:] = fav
# #                                         elif v < 0: colors.iloc[i, 3:] = unfav
# #                             return colors
# #                         st.dataframe(m_df.style.apply(apply_matrix_styles, axis=None), use_container_width=True, hide_index=True)

# #                     render_clustered_matrix("1. Financial Breakdown Matrix", ["Annual Ordering Fees ($)", "Annual Storage Carrying Cost ($)", "Financial Penalty from Stockouts ($)", "Total Policy Operating Cost ($)"], [actual_total_ordering_cost, actual_total_holding_cost, actual_lost_sales_financial, actual_total_cost], [optimal_ordering_cost, optimal_holding_cost, optimal_lost_sales_financial, optimal_total_cost], ["currency", "currency", "currency", "currency"])
# #                     render_clustered_matrix("2. Logistical Operations Footprint Matrix", ["Average Volume Kept On-Hand", "Maximum Storage Spike Level", "Total Orders Dispatched", "Average Logistics Cycle Time", "Average Order Shipment Size"], [actual_avg_inventory, actual_max_inventory, actual_orders_placed, actual_cycle_time, actual_avg_order_size], [simmed_avg_opt_inv, simmed_max_opt_inv, opt_orders_placed, policy_cycle_time, policy_avg_order_size], ["units", "units", "count", "days", "units"])
# #                     render_clustered_matrix("3. Working Capital Release Matrix", ["Peak Working Capital Tied Up ($)", "Average Working Capital Tied Up ($)", "Minimum Base Working Capital ($)"], [act_max_wc, act_avg_wc, act_min_wc], [opt_max_wc, opt_avg_wc, opt_min_wc], ["currency", "currency", "currency"])
# #                     render_clustered_matrix("4. Stockout Risk & Vulnerability Matrix", ["Absolute Minimum Buffer Stock", "Stockout Events (Unfulfilled Days)", "Total Unfulfilled Deficit Volume", "Days with Absolute Zero Closing Stock", "Achieved Order Fill Rate (%)"], [actual_min_inventory, stockout_days_act, lost_sales_qty_act, zero_stock_days_act, actual_fill_rate * 100], [simmed_min_inventory, stockout_days_opt, lost_sales_qty_opt, zero_stock_days_opt, simmed_opt_fill_rate * 100], ["units", "count", "units", "count", "pct"])

# #                     st.markdown("---")
# #                     st.markdown("### 📈 Tactical Operations Timeline Visualizations")
# #                     timeline_fig = go.Figure()
# #                     timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_act, name="Historical Actuals (Ledger)", line=dict(color='#B0C4DE', width=2), fill='tozeroy', fillcolor='rgba(176, 196, 222, 0.15)'))
# #                     timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_opt, name=f"Recommended Optimized Policy ({best_fit_name.split(' ')[0]})", line=dict(color='#1F77B4', width=2.5)))
# #                     timeline_fig.add_trace(go.Scatter(x=df["Date"], y=[max(0, raw_target_level - risk_mean)] * len(df), name="Calculated Safety Stock Floor", line=dict(color='#FF4B4B', width=1.5, dash='dot')))
# #                     timeline_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
# #                     st.plotly_chart(timeline_fig, use_container_width=True)


# #                     # ==========================================
# #                     #      SECTION B: COMPARATIVE ANALYSIS (CLEAN UI)
# #                     # ==========================================
# #                     st.markdown("---")
# #                     st.header("🔬 Section B: Multi-Scenario Comparative Analysis")
# #                     st.markdown(
# #                         "Leveraging the high-speed vectorized simulation engine, you can now backtest and compare up to 6 "
# #                         "different inventory policies simultaneously. By adjusting these mechanical levers, you can easily identify "
# #                         "operational blind spots without manually tracking the math."
# #                     )
                    
# #                     active_scenarios_list = []
                    
# #                     tab_cont, tab_per = st.tabs(["📉 Continuous Review (Q, R) Scenarios", "⏳ Periodic Review (P, T) Scenarios"])
                    
# #                     with tab_cont:
# #                         st.markdown(f"**Baseline Intelligence:** The data-driven optimal benchmark is **Q: {int(raw_optimal_q):,}** and **ROP: {int(raw_target_level):,}**.")
                        
# #                         c1, c2, c3 = st.columns(3)
                        
# #                         def create_cont_box(col, num, default_q, default_r):
# #                             with col:
# #                                 st.markdown(f"##### 🎛️ Scenario C{num}")
# #                                 run_c = st.toggle(f"Include C{num} in Chart", key=f"run_c{num}", value=(num==1))
# #                                 q_val = st.number_input("Order Qty (Q)", min_value=1, value=int(default_q), step=10, key=f"q_c{num}")
# #                                 r_val = st.number_input("Reorder Point (ROP)", min_value=0, value=int(default_r), step=10, key=f"r_c{num}")
                                
# #                                 if run_c:
# #                                     active_scenarios_list.append({
# #                                         "Case Name": f"C{num} (Q:{q_val}, R:{r_val})", 
# #                                         "Policy Type": "Continuous Review (Q, R)", 
# #                                         "P1": q_val, 
# #                                         "P2": r_val
# #                                     })

# #                         create_cont_box(c1, 1, raw_optimal_q, raw_target_level)
# #                         create_cont_box(c2, 2, raw_optimal_q * 1.5, raw_target_level)
# #                         create_cont_box(c3, 3, raw_optimal_q, raw_target_level * 1.2)

# #                     with tab_per:
# #                         st.markdown("**Baseline Intelligence:** Adjust the Review Period (P) below. The engine will dynamically calculate a safe expected Target Level (T) for that exact timeframe.")
                        
# #                         p1, p2, p3 = st.columns(3)
                        
# #                         def create_per_box(col, num, default_p):
# #                             with col:
# #                                 st.markdown(f"##### 🎛️ Scenario P{num}")
# #                                 run_p = st.toggle(f"Include P{num} in Chart", key=f"run_p{num}", value=False)
# #                                 p_val = st.number_input("Review Period (P Days)", min_value=1, value=int(default_p), step=1, key=f"p_p{num}")
                                
# #                                 target_guide = int(raw_target_level + (avg_daily_demand_calc * p_val))
                                
# #                                 t_val = st.number_input("Target Level (T)", min_value=0, value=target_guide, step=10, key=f"t_p{num}")
# #                                 st.caption(f"💡 *Engine recommended (T) for {p_val} days: **~{target_guide:,}***")
                                
# #                                 if run_p:
# #                                     active_scenarios_list.append({
# #                                         "Case Name": f"P{num} (P:{p_val}, T:{t_val})", 
# #                                         "Policy Type": "Periodic Review (P, T)", 
# #                                         "P1": p_val, 
# #                                         "P2": t_val
# #                                     })

# #                         create_per_box(p1, 1, 7)
# #                         create_per_box(p2, 2, 14)
# #                         create_per_box(p3, 3, 30)

# #                     st.markdown("---")
                    
# #                     if st.button("🚀 Compare Scenarios", type="primary", use_container_width=True):
# #                         if not active_scenarios_list:
# #                             st.warning("Please toggle 'Include' for at least one scenario above to generate the comparison.")
# #                         else:
# #                             summary_data = []
# #                             comp_fig = go.Figure()
                            
# #                             comp_fig.add_trace(go.Scatter(
# #                                 x=df["Date"], y=inv_levels_act, mode='lines', 
# #                                 name="Historical Actuals", line=dict(color='rgba(176, 196, 222, 0.4)', width=2, dash='dot')
# #                             ))

# #                             # Add baseline to the summary data
# #                             summary_data.append({
# #                                 "Scenario Blueprint": "📊 Historical Actuals (Baseline)",
# #                                 "Total Op Cost ($)": actual_total_cost,
# #                                 "Fill Rate (%)": actual_fill_rate * 100,
# #                                 "Avg Inv (Units)": actual_avg_inventory,
# #                                 "Avg Working Capital ($)": act_avg_wc,
# #                                 "Max Peak Capital ($)": act_max_wc
# #                             })

# #                             color_palette = px.colors.qualitative.D3
                            
# #                             for index, scenario in enumerate(active_scenarios_list):
# #                                 case_name = scenario["Case Name"]
# #                                 p_type = scenario["Policy Type"]
# #                                 val1 = scenario["P1"]
# #                                 val2 = scenario["P2"]
                                
# #                                 s_inv, s_lost, s_orders = fast_simulate_inventory(
# #                                     demand_arr_main, purchase_arr_main, opening_stock_override, 
# #                                     lead_time_days, p_type, val1, val2
# #                                 )
                                
# #                                 s_lost_sum = s_lost.sum()
# #                                 s_orders_count = np.count_nonzero(s_orders)
# #                                 s_avg_inv = np.mean(s_inv)
# #                                 s_max_inv = np.max(s_inv)
                                
# #                                 s_fill_rate = max(0.0, 1.0 - (s_lost_sum / max(1, total_demand)))
# #                                 s_total_cost = (s_orders_count * ordering_cost) + (s_avg_inv * unit_holding_cost) + (s_lost_sum * lost_sales_penalty)
                                
# #                                 summary_data.append({
# #                                     "Scenario Blueprint": case_name,
# #                                     "Total Op Cost ($)": s_total_cost,
# #                                     "Fill Rate (%)": s_fill_rate * 100,
# #                                     "Avg Inv (Units)": s_avg_inv,
# #                                     "Avg Working Capital ($)": s_avg_inv * item_unit_cost,
# #                                     "Max Peak Capital ($)": s_max_inv * item_unit_cost
# #                                 })
                                
# #                                 comp_fig.add_trace(go.Scatter(
# #                                     x=df["Date"], y=s_inv, mode='lines', 
# #                                     name=case_name, line=dict(color=color_palette[index % len(color_palette)], width=2.5)
# #                                 ))

# #                             st.markdown("##### 🏆 Comparative Outcomes Scorecard")
# #                             comp_df = pd.DataFrame(summary_data)
                            
# #                             def highlight_baseline(s):
# #                                 return ['background-color: rgba(176, 196, 222, 0.15)' if s['Scenario Blueprint'] == "📊 Historical Actuals (Baseline)" else '' for v in s]

# #                             st.dataframe(
# #                                 comp_df.style.apply(highlight_baseline, axis=1),
# #                                 use_container_width=True,
# #                                 hide_index=True,
# #                                 column_config={
# #                                     "Total Op Cost ($)": st.column_config.NumberColumn(format="$%.2f"),
# #                                     "Fill Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
# #                                     "Avg Inv (Units)": st.column_config.NumberColumn(format="%d"),
# #                                     "Avg Working Capital ($)": st.column_config.NumberColumn(format="$%.0f"),
# #                                     "Max Peak Capital ($)": st.column_config.NumberColumn(format="$%.0f")
# #                                 }
# #                             )
                            
# #                             st.markdown("##### 📈 Strategic Trajectory Matrix")
# #                             comp_fig.update_layout(
# #                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
# #                                 xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", 
# #                                 height=450, legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
# #                             )
# #                             st.plotly_chart(comp_fig, use_container_width=True)

# # with tab5:
# #     st.header("⚖️ Advanced Inventory Optimization Suite")
# #     st.markdown(
# #         "Analyze your inventory data through a twin-lens framework. First, review a historical backtest audit "
# #         "to identify legacy profit leaks."
# #     )
    
# #     # --- 🚀 THE VECTORIZED SIMULATION ENGINE (NOW WITH FIFO BATCH AGE TRACKING) ---
# #     def fast_simulate_inventory(demand_arr, purchase_arr, opening_stock, lead_time, policy_type, param1, param2):
# #         total_days = len(demand_arr)
# #         inv_levels = np.zeros(total_days)
# #         lost_sales = np.zeros(total_days)
# #         orders_placed = np.zeros(total_days)
        
# #         # FIFO Age Tracking Arrays
# #         avg_ages = np.zeros(total_days)
# #         max_ages = np.zeros(total_days)
# #         age_buckets = np.zeros((total_days, 4)) # [0-30, 31-60, 61-90, 90+]
        
# #         max_lt = int(lead_time)
# #         pipeline = np.zeros(total_days + max_lt + 1)
        
# #         if policy_type == "Actual":
# #             pipeline[:total_days] = purchase_arr
# #         else:
# #             warm_up = min(max_lt, total_days)
# #             pipeline[:warm_up] = purchase_arr[:warm_up]
            
# #         current_inv = opening_stock
        
# #         # FIFO Queue representing batches of inventory. Format: [arrival_day_index, remaining_quantity]
# #         # We assume initial opening stock arrived at day 0.
# #         fifo_queue = [[0, opening_stock]] if opening_stock > 0 else []
        
# #         for i in range(total_days):
# #             demand = demand_arr[i]
# #             arriving = pipeline[i]
            
# #             # 1. Receive new stock
# #             if arriving > 0:
# #                 fifo_queue.append([i, arriving])
# #                 current_inv += arriving
                
# #             # 2. Fulfill demand (Consume oldest batches first)
# #             demand_left = demand
# #             while demand_left > 0 and fifo_queue:
# #                 oldest_batch_qty = fifo_queue[0][1]
# #                 if oldest_batch_qty <= demand_left:
# #                     demand_left -= oldest_batch_qty
# #                     fifo_queue.pop(0) # Batch fully consumed
# #                 else:
# #                     fifo_queue[0][1] -= demand_left
# #                     demand_left = 0 # Demand fully satisfied
                    
# #             if demand_left > 0:
# #                 lost_sales[i] = demand_left
# #                 current_inv = 0
# #             else:
# #                 current_inv -= demand
                
# #             inv_levels[i] = current_inv
            
# #             # 3. Calculate Age Metrics for remaining inventory
# #             if fifo_queue:
# #                 total_qty = 0
# #                 sum_age_qty = 0
# #                 max_age_val = 0
                
# #                 for item in fifo_queue:
# #                     arr_day, qty = item
# #                     age = i - arr_day
# #                     total_qty += qty
# #                     sum_age_qty += (age * qty)
                    
# #                     if age > max_age_val:
# #                         max_age_val = age
                        
# #                     # Allocate to age buckets
# #                     if age <= 30:
# #                         age_buckets[i, 0] += qty
# #                     elif age <= 60:
# #                         age_buckets[i, 1] += qty
# #                     elif age <= 90:
# #                         age_buckets[i, 2] += qty
# #                     else:
# #                         age_buckets[i, 3] += qty
                        
# #                 if total_qty > 0:
# #                     avg_ages[i] = sum_age_qty / total_qty
# #                     max_ages[i] = max_age_val
            
# #             # 4. Replenishment Policy Logic
# #             if policy_type == "Continuous Review (Q, R)":
# #                 net_position = current_inv + np.sum(pipeline[i+1:])
# #                 if net_position <= param2:
# #                     pipeline[i + max_lt] += param1
# #                     orders_placed[i] = param1
# #             elif policy_type == "Periodic Review (P, T)":
# #                 if i % int(param1) == 0:
# #                     net_position = current_inv + np.sum(pipeline[i+1:])
# #                     order_qty = max(0, param2 - net_position)
# #                     if order_qty > 0:
# #                         pipeline[i + max_lt] += order_qty
# #                         orders_placed[i] = order_qty
                        
# #         return inv_levels, lost_sales, orders_placed, avg_ages, max_ages, age_buckets

# #     # --- STEP 1: INPUT PARAMETERS ---
# #     st.subheader("1. Parameters & Cost Drivers")
    
# #     col1, col2, col3 = st.columns(3)
    
# #     with col1:
# #         item_unit_cost = st.number_input("Item Unit Cost ($/Unit)", min_value=0.01, value=25.00, step=1.00, key="unit_cost_global")
# #         holding_fixed_daily = st.number_input("Fixed Holding Cost ($/Unit/Day)", min_value=0.0, value=0.0, step=0.01, key="fixed_hold_global")
        
# #     with col2:
# #         holding_var_pct = st.number_input("Variable Holding Cost (% of Item Cost/year)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="var_hold_global") / 100.0
# #         ordering_cost = st.number_input("Ordering Cost ($/order)", min_value=0.1, value=75.0, step=5.0, key="order_cost_global")
        
# #     with col3:
# #         lost_sales_penalty = st.number_input("Lost Sales Penalty ($/Unit Lost)", min_value=0.0, value=0.0, step=1.0, key="penalty_global")
# #         lead_time_days = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1, key="lt_global")

# #     st.markdown("---")
# #     col_sys1, col_sys2 = st.columns([1, 2])
# #     with col_sys1:
# #         review_system = st.radio("Inventory Review System Strategy", ["Continuous Review (Q, R)", "Periodic Review (P, T)"], key="review_system_global")
# #     with col_sys2:
# #         service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="service_level_global") / 100.0

# #     if review_system == "Periodic Review (P, T)":
# #         st.markdown("##### ⏳ Periodic Configuration")
# #         p_col1, _ = st.columns(2)
# #         with p_col1:
# #             user_p_days = st.number_input("Review Period Cycle (P in Days)", min_value=1, value=14, step=1, key="p_days_global")
# #     else:
# #         user_p_days = 1

# #     # --- STEP 2: MULTI-FORMAT DATA INGESTION ENGINE ---
# #     st.subheader("2. Upload Historical Invoices & Demand Data")
# #     uploaded_file = st.file_uploader(
# #         "Upload Inventory Ledger (Supports standard templates, raw ERP transactional logs, or stock card snapshots)", 
# #         type=["csv", "xlsx", "xls"], 
# #         key="uploader_global"
# #     )
    
# #     if uploaded_file is None:
# #         st.info("📥 Please upload your inventory ledger file (CSV or Excel) above to populate the suite modules.")
# #     else:
# #         detected_sheet_opening_stock = None
# #         data_loaded_successfully = False
# #         df_mapped = None
        
# #         try:
# #             if uploaded_file.name.endswith('.csv'):
# #                 raw_df = pd.read_csv(uploaded_file)
# #             else:
# #                 raw_df = pd.read_excel(uploaded_file)
                
# #             raw_df.columns = raw_df.columns.str.strip()
                
# #             if "Date" not in raw_df.columns:
# #                 st.error("❌ Missing required column: 'Date'.")
# #             else:
# #                 raw_df["Date"] = pd.to_datetime(raw_df["Date"])
# #                 raw_df = raw_df.sort_values(by="Date").reset_index(drop=True)
                
# #                 open_balance_headers = ["Opening Balance", "Opening", "Opening_Stock", "Opening Stock"]
# #                 for header in open_balance_headers:
# #                     if header in raw_df.columns:
# #                         detected_sheet_opening_stock = int(raw_df[header].iloc[0])
# #                         break
                
# #                 if "Demand_Qty" in raw_df.columns and "Purchase_Qty" in raw_df.columns:
# #                     df_mapped = raw_df[["Date", "Demand_Qty", "Purchase_Qty"]].copy()
# #                 elif "Demand" in raw_df.columns and "Stock Received" in raw_df.columns:
# #                     df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df["Demand"], "Purchase_Qty": raw_df["Stock Received"]})
# #                 elif ("Receiving" in raw_df.columns) and any(col in raw_df.columns for col in ["Demand/Sales", "Demand", "Sales"]):
# #                     outbound_col = "Demand/Sales" if "Demand/Sales" in raw_df.columns else ("Demand" if "Demand" in raw_df.columns else "Sales")
# #                     df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df[outbound_col], "Purchase_Qty": raw_df["Receiving"]})
# #                 else:
# #                     st.error("❌ Column layout structure mismatch. Could not find Demand and Receiving columns.")

# #                 if df_mapped is not None:
# #                     df_mapped = df_mapped.groupby("Date").agg({"Demand_Qty": "sum", "Purchase_Qty": "sum"}).reset_index()
# #                     df_mapped = df_mapped.set_index("Date").resample("1D").asfreq()
# #                     df_mapped["Demand_Qty"] = df_mapped["Demand_Qty"].fillna(0.0)
# #                     df_mapped["Purchase_Qty"] = df_mapped["Purchase_Qty"].fillna(0.0)
# #                     df_mapped = df_mapped.reset_index()
# #                     data_loaded_successfully = True
                    
# #         except Exception as e:
# #             st.error(f"Error parsing file elements: {e}")

# #         # ONLY PROCEED IF DATA IS CLEANED
# #         if data_loaded_successfully and df_mapped is not None:
# #             full_df = df_mapped.copy() 
            
# #             absolute_min_date = full_df["Date"].min().date()
# #             absolute_max_date = full_df["Date"].max().date()
            
# #             file_state_key = f"last_file_{uploaded_file.name}_{uploaded_file.size}"
            
# #             if "current_file_token" not in st.session_state or st.session_state.current_file_token != file_state_key:
# #                 st.session_state.current_file_token = file_state_key
# #                 st.session_state.min_date_global = absolute_min_date
# #                 st.session_state.max_date_global = absolute_max_date
                
# #                 avg_daily_full = full_df["Demand_Qty"].mean()
# #                 if detected_sheet_opening_stock is not None:
# #                     default_start = int(detected_sheet_opening_stock)
# #                 else:
# #                     default_start = int(1.25 * (avg_daily_full * lead_time_days))
                
# #                 st.session_state.absolute_day1_stock = default_start
# #                 st.session_state.previous_start_date = absolute_min_date
# #                 st.session_state.previous_end_date = absolute_max_date
# #                 st.session_state.opening_stock_global = default_start
                
# #                 st.session_state.start_date_key = absolute_min_date
# #                 st.session_state.end_date_key = absolute_max_date
                
# #                 if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
# #                 if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite

# #             st.divider()
            
# #             st.markdown("### 📅 3. Select Analysis Period")
# #             st.markdown("Filter historical data. The starting inventory will automatically mathematically roll forward to match your selected Start Date.")
            
# #             def reset_dates():
# #                 st.session_state.start_date_key = st.session_state.min_date_global
# #                 st.session_state.end_date_key = st.session_state.max_date_global
# #                 st.session_state.previous_start_date = st.session_state.min_date_global
# #                 st.session_state.previous_end_date = st.session_state.max_date_global
# #                 st.session_state.opening_stock_global = st.session_state.absolute_day1_stock

# #             col_date1, col_date2, col_date3 = st.columns([2, 2, 1])
            
# #             with col_date1:
# #                 start_date = st.date_input("Starting Date", min_value=absolute_min_date, max_value=absolute_max_date, key="start_date_key")
# #             with col_date2:
# #                 end_date = st.date_input("Ending Date", min_value=absolute_min_date, max_value=absolute_max_date, key="end_date_key")
# #             with col_date3:
# #                 st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
# #                 st.button("🔄 Reset", on_click=reset_dates, use_container_width=True)
            
# #             if "previous_start_date" not in st.session_state:
# #                 st.session_state.previous_start_date = absolute_min_date
# #             if "previous_end_date" not in st.session_state:
# #                 st.session_state.previous_end_date = absolute_max_date

# #             date_changed = (start_date != st.session_state.previous_start_date) or (end_date != st.session_state.previous_end_date)

# #             if date_changed:
# #                 if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
# #                 if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite
                
# #                 if start_date != st.session_state.previous_start_date:
# #                     temp_balance = st.session_state.absolute_day1_stock
# #                     pre_period_df = full_df[full_df["Date"].dt.date < start_date]
                    
# #                     pre_d_arr = pre_period_df["Demand_Qty"].values
# #                     pre_p_arr = pre_period_df["Purchase_Qty"].values
# #                     for idx in range(len(pre_d_arr)):
# #                         temp_balance = max(0, temp_balance + pre_p_arr[idx] - pre_d_arr[idx])
                    
# #                     st.session_state.opening_stock_global = int(temp_balance)
                
# #                 st.session_state.previous_start_date = start_date
# #                 st.session_state.previous_end_date = end_date

# #             if start_date > end_date:
# #                 st.error("⚠️ The Starting Date must be before or equal to the Ending Date. Please adjust your selection.")
# #             else:
# #                 df = full_df[(full_df["Date"].dt.date >= start_date) & (full_df["Date"].dt.date <= end_date)].reset_index(drop=True)
                
# #                 if df.empty:
# #                     st.warning("No data available for the selected date range. Please widen your selection.")
# #                 else:
# #                     demand_arr_main = df["Demand_Qty"].values
# #                     purchase_arr_main = df["Purchase_Qty"].values
                    
# #                     actual_orders_placed = np.count_nonzero(purchase_arr_main)
# #                     actual_total_units_purchased = purchase_arr_main.sum()
# #                     total_demand = demand_arr_main.sum()
                    
# #                     avg_daily_demand_calc = demand_arr_main.mean()
# #                     std_daily_demand = demand_arr_main.std() if len(df) > 1 else 0
# #                     cov = std_daily_demand / max(0.1, avg_daily_demand_calc)

# #                     with st.expander("📈 View Historical Demand Trend & Growth Timeline", expanded=False):
# #                         rolling_days = st.slider("Select Rolling Average Window (Days)", min_value=1, max_value=90, value=15, step=1)
# #                         df[f"Rolling_Avg"] = df["Demand_Qty"].rolling(window=rolling_days, min_periods=1).mean()
                        
# #                         stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
# #                         if len(df) >= rolling_days * 2:
# #                             current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
# #                             previous_window_sum = df["Demand_Qty"].iloc[-(rolling_days*2):-rolling_days].sum()
# #                             if previous_window_sum > 0:
# #                                 trend_pct = ((current_window_sum - previous_window_sum) / previous_window_sum) * 100
# #                             else:
# #                                 trend_pct = 100.0 if current_window_sum > 0 else 0.0
# #                             stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta=f"{trend_pct:+.1f}% Growth", delta_color="normal")
# #                         elif len(df) >= rolling_days:
# #                             current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
# #                             stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta="Widen dates for trend", delta_color="off")
# #                         else:
# #                             stat_col1.metric(label="Trend", value="N/A", delta="Insufficient Data", delta_color="off")
                            
# #                         stat_col2.metric("Average Daily Demand", f"{avg_daily_demand_calc:.1f} units")
# #                         stat_col3.metric("Standard Deviation", f"{std_daily_demand:.1f} units")
# #                         stat_col4.metric("Volatility (CoV)", f"{cov:.2f}")
# #                         st.markdown("---")

# #                         demand_fig = go.Figure()
# #                         demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Demand_Qty"], mode='lines', name='Raw Daily Demand', line=dict(color='#B0C4DE', width=1.5), opacity=0.6))
# #                         demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Rolling_Avg"], mode='lines', name=f'{rolling_days}-Day Moving Avg', line=dict(color='#3d5a80', width=3)))
# #                         demand_fig.update_layout(template="plotly_white", yaxis_title="Units", xaxis_title="Date", margin=dict(t=20, b=20), height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
# #                         st.plotly_chart(demand_fig, use_container_width=True)

# #                     st.markdown("---")
# #                     st.markdown("##### 📦 Initial Warehouse Capital Balance")
# #                     opening_stock_override = st.number_input("Starting Balance for Selected Period", min_value=0, step=10, key="opening_stock_global", help="This value automatically updates mathematically based on the start date you select above, but you can override it manually.")

# #                     with st.expander("📋 View Complete Running Balance Table Snapshots", expanded=False):
# #                         st.markdown("An interactive historical stock card ledger driven directly by your initial opening stock parameter above.")
# #                         cl_open_list, cl_close_list, t_bal = np.zeros(len(df)), np.zeros(len(df)), opening_stock_override
                        
# #                         for i_run in range(len(df)):
# #                             cl_open_list[i_run] = t_bal
# #                             t_bal = max(0, t_bal + purchase_arr_main[i_run] - demand_arr_main[i_run])
# #                             cl_close_list[i_run] = t_bal
                            
# #                         full_stock_card_df = pd.DataFrame({
# #                             "Timeline Date": df["Date"].dt.strftime('%Y-%m-%d'),
# #                             "Opening Balance": cl_open_list.astype(int),
# #                             "Cleaned Demand Volume (Units)": demand_arr_main.astype(int),
# #                             "Consolidated Stock Received (Units)": purchase_arr_main.astype(int),
# #                             "Closing Balance": cl_close_list.astype(int)
# #                         })
                        
# #                         st.dataframe(full_stock_card_df, use_container_width=True, hide_index=True, column_config={"Opening Balance": st.column_config.NumberColumn(format="%d"), "Cleaned Demand Volume (Units)": st.column_config.NumberColumn(format="%d"), "Consolidated Stock Received (Units)": st.column_config.NumberColumn(format="%d"), "Closing Balance": st.column_config.NumberColumn(format="%d")})

# #                     # --- STEP 4: ADVANCED STATISTICAL FIT RUNNER ---
# #                     st.subheader("4. Statistical Risk & Distribution Engines")
                    
# #                     annual_demand = avg_daily_demand_calc * 365
# #                     annual_fixed_holding_per_unit = holding_fixed_daily * 365
# #                     unit_holding_cost = annual_fixed_holding_per_unit + (item_unit_cost * holding_var_pct)
                    
# #                     risk_horizon_days = lead_time_days if review_system == "Continuous Review (Q, R)" else (user_p_days + lead_time_days)
# #                     rolling_risk_demand = df["Demand_Qty"].rolling(window=int(risk_horizon_days)).sum().dropna().values
# #                     risk_mean = np.mean(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0
# #                     risk_std = np.std(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0

# #                     if len(rolling_risk_demand) > 0:
# #                         if np.max(rolling_risk_demand) <= 0:
# #                             best_fit_name = "Zero Demand Base"
# #                             raw_target_level = 0.0
# #                         else:
# #                             empirical_rop_raw = np.percentile(rolling_risk_demand, service_level * 100)
# #                             safe_demand = np.where(rolling_risk_demand <= 0, 1e-5, rolling_risk_demand)
                            
# #                             log_params = stats.lognorm.fit(safe_demand, floc=0)
# #                             gam_params = stats.gamma.fit(safe_demand, floc=0)

# #                             counts, bins = np.histogram(rolling_risk_demand, bins=20, density=True)
# #                             bin_centers = (bins[:-1] + bins[1:]) / 2
                            
# #                             rss_norm = np.sum((counts - stats.norm.pdf(bin_centers, loc=risk_mean, scale=risk_std)) ** 2)
# #                             rss_log = np.sum((counts - stats.lognorm.pdf(bin_centers, *log_params)) ** 2)
# #                             rss_gam = np.sum((counts - stats.gamma.pdf(bin_centers, *gam_params)) ** 2)

# #                             if cov > 0.75:
# #                                 best_fit_name = "Empirical (Data-Driven)"
# #                                 raw_target_level = empirical_rop_raw
# #                             else:
# #                                 errors = {"Normal": rss_norm, "Log-Normal": rss_log, "Gamma": rss_gam}
# #                                 best_fit_name = min(errors, key=errors.get)
# #                                 if best_fit_name == "Normal":
# #                                     raw_target_level = stats.norm.ppf(service_level, loc=risk_mean, scale=risk_std)
# #                                 elif best_fit_name == "Log-Normal":
# #                                     raw_target_level = stats.lognorm.ppf(service_level, *log_params)
# #                                 else:
# #                                     raw_target_level = stats.gamma.ppf(service_level, *gam_params)
# #                     else:
# #                         best_fit_name = "Default (Insufficient Data)"
# #                         raw_target_level = avg_daily_demand_calc * risk_horizon_days
                        
# #                     raw_optimal_q = np.sqrt((2 * annual_demand * ordering_cost) / max(0.01, unit_holding_cost))

# #                     if "q_audit_suite" not in st.session_state:
# #                         st.session_state.q_audit_suite = max(1, int(raw_optimal_q)) if review_system == "Continuous Review (Q, R)" else int(avg_daily_demand_calc * user_p_days)
# #                     if "rop_audit_suite" not in st.session_state:
# #                         st.session_state.rop_audit_suite = max(0, int(raw_target_level))

# #                     with st.expander("📊 View Cleaned Demand Distribution & Best-Fit Curve Metrics", expanded=False):
# #                         stat_col1, stat_col2, stat_col3 = st.columns(3)
# #                         stat_col1.metric("Average Daily Demand", f"{avg_daily_demand_calc:.2f} units")
# #                         stat_col2.metric("Coefficient of Variation (CV)", f"{cov:.2f}")
# #                         stat_col3.metric("Engine Selection", f"✨ {best_fit_name}")
# #                         st.markdown("---")
# #                         hist_fig = go.Figure()
# #                         hist_fig.add_trace(go.Histogram(x=df["Demand_Qty"], name="Historical Days", marker_color='#1F77B4', opacity=0.6))
# #                         hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Demand Quantity (Units / Day)", yaxis_title="Frequency", margin=dict(l=40, r=40, t=10, b=40), height=300)
# #                         st.plotly_chart(hist_fig, use_container_width=True)

# #                     # ==========================================
# #                     #      SECTION C: ERRATIC DEMAND PROFILER
# #                     # ==========================================
# #                     st.markdown("---")
# #                     st.subheader("5. 🌪️ Erratic Demand & Empirical ROP Profiler")
                    
# #                     with st.expander("Open Rolling Window & Empirical Profiler", expanded=False):
# #                         st.markdown(
# #                             "Standard safety stock math assumes demand follows a predictable bell curve. For erratic or lumpy demand, "
# #                             "that assumption breaks down. This tool mechanically profiles your exact historical risk by analyzing every "
# #                             "rolling vulnerability window in your dataset."
# #                         )
                        
# #                         tab_emp_cont, tab_emp_per = st.tabs(["📉 Continuous Review (ROP)", "⏳ Periodic Review (Target Level)"])
                        
# #                         with tab_emp_cont:
# #                             st.markdown("##### Continuous Review Risk Profiler")
# #                             st.write("In a Continuous system, you reorder immediately when you hit a threshold. Your vulnerability window is simply the **Lead Time**.")
                            
# #                             col_emp1, col_emp2, col_emp3 = st.columns(3)
                            
# #                             with col_emp1:
# #                                 emp_lt_window = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_lt_window")
# #                             with col_emp2:
# #                                 emp_test_rop = st.number_input("Test Reorder Point (ROP)", min_value=0, value=int(raw_target_level), step=10, key="emp_test_rop")
# #                             with col_emp3:
# #                                 emp_target_sl = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl")

# #                             rolling_demand_c = df["Demand_Qty"].rolling(window=emp_lt_window).sum().dropna()

# #                             if len(rolling_demand_c) > 0:
# #                                 windows_below_rop = np.sum(rolling_demand_c <= emp_test_rop)
# #                                 total_windows_c = len(rolling_demand_c)
# #                                 achieved_sl_c = (windows_below_rop / total_windows_c) * 100
# #                                 required_rop = np.percentile(rolling_demand_c, emp_target_sl)

# #                                 res_col1, res_col2 = st.columns(2)
# #                                 with res_col1:
# #                                     st.info(f"**Testing ROP of {emp_test_rop:,}:**\n\nOut of {total_windows_c:,} historical {emp_lt_window}-day windows, the total demand was successfully covered by {emp_test_rop:,} units exactly **{windows_below_rop:,} times**. This yields an empirical service level of **{achieved_sl_c:.1f}%**.")
# #                                 with res_col2:
# #                                     st.success(f"**Targeting {emp_target_sl}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl}% of all historical {emp_lt_window}-day scenarios, your ROP must be set to the empirical percentile: **{int(required_rop):,} units**.")

# #                                 emp_fig_c = go.Figure()
# #                                 emp_fig_c.add_trace(go.Histogram(x=rolling_demand_c, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {emp_lt_window}-Day Windows"))
# #                                 emp_fig_c.add_vline(x=emp_test_rop, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested ROP ({emp_test_rop})", annotation_position="top right")
# #                                 emp_fig_c.add_vline(x=required_rop, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Target ROP ({int(required_rop)})", annotation_position="top left")

# #                                 emp_fig_c.update_layout(
# #                                     title=f"Actual Demand Distribution Across All {emp_lt_window}-Day Windows",
# #                                     xaxis_title=f"Total Units Demanded in a {emp_lt_window}-Day Window",
# #                                     yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40)
# #                                 )
# #                                 st.plotly_chart(emp_fig_c, use_container_width=True)
# #                             else:
# #                                 st.warning(f"⚠️ Not enough data to calculate rolling windows for a {emp_lt_window}-day lead time.")

# #                         with tab_emp_per:
# #                             st.markdown("##### Periodic Review Risk Profiler")
# #                             st.write("In a Periodic system, you are blind between cycle counts. Your vulnerability window is the **Lead Time + the Review Period**.")
                            
# #                             default_p_val = int(user_p_days) if user_p_days > 1 else 14
# #                             default_t_val = int(raw_target_level + (avg_daily_demand_calc * default_p_val))
                            
# #                             p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                            
# #                             with p_col1:
# #                                 emp_p_days = st.number_input("Review Period (P)", min_value=1, value=default_p_val, step=1, key="emp_p_days")
# #                             with p_col2:
# #                                 emp_p_lt = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_p_lt")
# #                             with p_col3:
# #                                 emp_test_t = st.number_input("Test Target Level (T)", min_value=0, value=default_t_val, step=10, key="emp_test_t")
# #                             with p_col4:
# #                                 emp_target_sl_p = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl_p")

# #                             risk_window_days = emp_p_days + emp_p_lt
# #                             rolling_demand_p = df["Demand_Qty"].rolling(window=risk_window_days).sum().dropna()

# #                             if len(rolling_demand_p) > 0:
# #                                 windows_below_t = np.sum(rolling_demand_p <= emp_test_t)
# #                                 total_windows_p = len(rolling_demand_p)
# #                                 achieved_sl_p = (windows_below_t / total_windows_p) * 100
# #                                 required_t = np.percentile(rolling_demand_p, emp_target_sl_p)

# #                                 res_col3, res_col4 = st.columns(2)
# #                                 with res_col3:
# #                                     st.info(f"**Testing Target (T) of {emp_test_t:,}:**\n\nOut of {total_windows_p:,} historical {risk_window_days}-day windows (P+L), demand was successfully covered by {emp_test_t:,} units exactly **{windows_below_t:,} times**. Empirical SL: **{achieved_sl_p:.1f}%**.")
# #                                 with res_col4:
# #                                     st.success(f"**Targeting {emp_target_sl_p}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl_p}% of all historical {risk_window_days}-day scenarios, your Target Level must be: **{int(required_t):,} units**.")

# #                                 emp_fig_p = go.Figure()
# #                                 emp_fig_p.add_trace(go.Histogram(x=rolling_demand_p, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {risk_window_days}-Day Windows"))
# #                                 emp_fig_p.add_vline(x=emp_test_t, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested Target ({emp_test_t})", annotation_position="top right")
# #                                 emp_fig_p.add_vline(x=required_t, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Required Target ({int(required_t)})", annotation_position="top left")

# #                                 emp_fig_p.update_layout(
# #                                     title=f"Actual Demand Distribution Across All {risk_window_days}-Day (P+L) Windows",
# #                                     xaxis_title=f"Total Units Demanded in a {risk_window_days}-Day Window",
# #                                     yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40)
# #                                 )
# #                                 st.plotly_chart(emp_fig_p, use_container_width=True)
# #                             else:
# #                                 st.warning(f"⚠️ Not enough data to calculate rolling windows for a {risk_window_days}-day (P+L) window.")

# #                     # --- STEP 5: FINAL POLICY OPTIMIZATION TUNING CONFIG ---
# #                     st.markdown("---")
# #                     st.subheader("6. Final Policy Parameter Tuning")
# #                     adjust_col1, adjust_col2 = st.columns(2)
# #                     with adjust_col1:
# #                         if review_system == "Continuous Review (Q, R)":
# #                             final_q = st.number_input("Target Order Quantity (Q)", min_value=1, step=10, key="q_audit_suite")
# #                         else:
# #                             cycle_demand_baseline = int(avg_daily_demand_calc * user_p_days)
# #                             final_q = st.number_input("Average Target Batch Size (Q)", min_value=1, value=cycle_demand_baseline, step=10, disabled=True, key="q_audit_suite_disabled")
# #                     with adjust_col2:
# #                         final_buffer_target = st.number_input("Reorder Point (ROP) / Target Level (T)", min_value=0, step=10, key="rop_audit_suite")

# #                     if review_system == "Continuous Review (Q, R)":
# #                         st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Optimal Order Quantity (EOQ): **{int(raw_optimal_q):,}** units | Recommended Reorder Point (ROP): **{int(raw_target_level):,}** units.")
# #                     else:
# #                         st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Expected Cycle Batch Size: **{int(avg_daily_demand_calc * user_p_days):,}** units | Recommended Max Order Up-To Level (T): **{int(raw_target_level):,}** units.")

# #                     optimal_p_days = max(1, int((final_q / max(0.1, avg_daily_demand_calc)))) if review_system == "Continuous Review (Q, R)" else int(user_p_days)

# #                     # ==========================================
# #                     #      SECTION A: HISTORICAL BACKTEST
# #                     # ==========================================
# #                     st.markdown("---")
# #                     st.header("📊 Section A: Historical Backtest Audit")
# #                     st.markdown("This analysis compares your **Historical Actuals** against our **Recommended Optimized Policy** under identical historical demand constraints to reveal operational friction.")
                    
# #                     st.info(f"🔥 **Intelligent Warm Start Active:** To prevent unfair 'Cold Start' stockouts, the optimization engine's initial pipeline has been pre-seeded with your actual historical receipts for the first **{int(lead_time_days)}** days (Lead Time window).")
                    
# #                     # RUN HISTORICAL ACTUALS WITH FIFO AGE TRACKING
# #                     inv_levels_act, lost_sales_act_arr, orders_placed_act_arr, avg_age_act, max_age_act, buckets_act = fast_simulate_inventory(
# #                         demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, "Actual", 0, 0
# #                     )
                    
# #                     # RUN OPTIMIZED POLICY WITH FIFO AGE TRACKING
# #                     inv_levels_opt, lost_sales_opt_arr, orders_placed_opt_arr, avg_age_opt, max_age_opt, buckets_opt = fast_simulate_inventory(
# #                         demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, review_system, 
# #                         optimal_p_days if review_system != "Continuous Review (Q, R)" else final_q, final_buffer_target
# #                     )

# #                     # --- MASTER CALCULATION COMPILING ENGINE ---
# #                     lost_sales_qty_act = lost_sales_act_arr.sum()
# #                     stockout_days_act = np.count_nonzero(lost_sales_act_arr)
# #                     zero_stock_days_act = np.count_nonzero(inv_levels_act == 0)
                    
# #                     actual_max_inventory = np.max(inv_levels_act)
# #                     actual_min_inventory = np.min(inv_levels_act)
# #                     actual_avg_inventory = np.mean(inv_levels_act)
# #                     actual_fill_rate = max(0.0, 1.0 - (lost_sales_qty_act / max(1, total_demand)))
# #                     actual_cycle_time = 365 / actual_orders_placed if actual_orders_placed > 0 else 365.0
# #                     actual_avg_order_size = actual_total_units_purchased / actual_orders_placed if actual_orders_placed > 0 else 0.0

# #                     actual_total_ordering_cost = actual_orders_placed * ordering_cost
# #                     actual_total_holding_cost = actual_avg_inventory * unit_holding_cost
# #                     actual_lost_sales_financial = lost_sales_qty_act * lost_sales_penalty
# #                     actual_total_cost = actual_total_ordering_cost + actual_total_holding_cost + actual_lost_sales_financial
                    
# #                     actual_overall_avg_age = np.mean(avg_age_act)
# #                     actual_overall_max_age = np.max(max_age_act)

# #                     lost_sales_qty_opt = lost_sales_opt_arr.sum()
# #                     stockout_days_opt = np.count_nonzero(lost_sales_opt_arr)
# #                     zero_stock_days_opt = np.count_nonzero(inv_levels_opt == 0)
# #                     opt_orders_placed = np.count_nonzero(orders_placed_opt_arr)
# #                     policy_total_units_ordered = orders_placed_opt_arr.sum()

# #                     simmed_avg_opt_inv = np.mean(inv_levels_opt)
# #                     simmed_max_opt_inv = np.max(inv_levels_opt)
# #                     simmed_min_inventory = np.min(inv_levels_opt)
# #                     simmed_opt_fill_rate = max(0.0, 1.0 - (lost_sales_qty_opt / max(1, total_demand)))
# #                     policy_cycle_time = 365 / opt_orders_placed if opt_orders_placed > 0 else 365.0
# #                     policy_avg_order_size = policy_total_units_ordered / opt_orders_placed if opt_orders_placed > 0 else 0.0

# #                     optimal_ordering_cost = opt_orders_placed * ordering_cost
# #                     optimal_holding_cost = simmed_avg_opt_inv * unit_holding_cost
# #                     optimal_lost_sales_financial = lost_sales_qty_opt * lost_sales_penalty
# #                     optimal_total_cost = optimal_ordering_cost + optimal_holding_cost + optimal_lost_sales_financial
                    
# #                     opt_overall_avg_age = np.mean(avg_age_opt)
# #                     opt_overall_max_age = np.max(max_age_opt)
                    
# #                     act_max_wc = actual_max_inventory * item_unit_cost
# #                     act_avg_wc = actual_avg_inventory * item_unit_cost
# #                     act_min_wc = actual_min_inventory * item_unit_cost
                    
# #                     opt_max_wc = simmed_max_opt_inv * item_unit_cost
# #                     opt_avg_wc = simmed_avg_opt_inv * item_unit_cost
# #                     opt_min_wc = simmed_min_inventory * item_unit_cost

# #                     true_net_benefit = actual_total_cost - optimal_total_cost
                    
# #                     if true_net_benefit > 0:
# #                         st.success(f"### 🎯 The Efficiency Opportunity\nBy shifting to the recommended optimized policy, you would have recovered **${true_net_benefit:,.2f}** over this historical period.")
# #                     else:
# #                         st.error(f"⚠️ **Operational Margin Deficit Risk:** This setup increases operational overhead by **${abs(true_net_benefit):,.2f} / year** compared to actuals.")

# #                     # =========================================================
# #                     # --- EXECUTIVE KPI SCORECARD ---
# #                     # =========================================================
# #                     st.markdown("### 🏆 Executive Summary: Value Realization")
                    
# #                     cash_released = act_avg_wc - opt_avg_wc

# #                     kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# #                     with kpi_col1:
# #                         st.metric(label="Total Cost Saving", value=f"${true_net_benefit:,.0f}")
# #                     with kpi_col2:
# #                         st.metric(label="Optimized Fill Rate", value=f"{simmed_opt_fill_rate * 100:.1f}%")
# #                     with kpi_col3:
# #                         st.metric(label="Avg Working Capital (Opt)", value=f"${opt_avg_wc:,.0f}")
# #                         st.markdown(f"<div style='margin-top: -15px; font-size: 0.85rem; color: gray;'>Historical: ${act_avg_wc:,.0f}</div>", unsafe_allow_html=True)
# #                     with kpi_col4:
# #                         release_label = "Cash Released" if cash_released >= 0 else "Capital Added (Tied Up)"
# #                         st.metric(label=release_label, value=f"${abs(cash_released):,.0f}")

# #                     st.markdown("---")
                    
# #                     def render_clustered_matrix(title, metrics, act_vals, pol_vals, formats):
# #                         st.markdown(f"#### {title}")
# #                         abs_var = [a - p for a, p in zip(act_vals, pol_vals)]
# #                         pct_var = []
# #                         for a, p in zip(act_vals, pol_vals):
# #                             if a == 0: pct_var.append(0.0)
# #                             else: pct_var.append(((a - p) / a) * 100)
                            
# #                         m_df = pd.DataFrame({"Operational Attribute Pillar": metrics})
# #                         for idx in range(len(metrics)):
# #                             fmt = formats[idx]
# #                             if fmt == "currency":
# #                                 m_df.at[idx, "Historical Actuals"] = f"${act_vals[idx]:,.2f}"
# #                                 m_df.at[idx, "Optimized Policy"] = f"${pol_vals[idx]:,.2f}"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"${abs_var[idx]:,.2f}" if abs_var[idx] >= 0 else f"-${abs(abs_var[idx]):,.2f}"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
# #                             elif fmt == "pct":
# #                                 m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:.1f}%"
# #                                 m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:.1f}%"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+.1f}% pts"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pol_vals[idx] - act_vals[idx]:+.1f}% pts"
# #                             elif fmt == "days":
# #                                 m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:,.1f} days"
# #                                 m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:,.1f} days"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+,.1f} days"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
# #                             else:
# #                                 m_df.at[idx, "Historical Actuals"] = f"{int(act_vals[idx]):,}"
# #                                 m_df.at[idx, "Optimized Policy"] = f"{int(pol_vals[idx]):,}"
# #                                 m_df.at[idx, "Net Delta Variance"] = f"{int(abs_var[idx]):+1,}"
# #                                 m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"

# #                         def apply_matrix_styles(x):
# #                             colors = pd.DataFrame('', index=x.index, columns=x.columns)
# #                             fav = 'background-color: #1A3E2B; color: #81C784; font-weight: bold;'
# #                             unfav = 'background-color: #3E1A1A; color: #E57373;'
# #                             for i, metric in enumerate(metrics):
# #                                 v = abs_var[i]
# #                                 if title == "1. Financial Breakdown Matrix" or title == "3. Working Capital Release Matrix":
# #                                     if v > 0: colors.iloc[i, 3:] = fav
# #                                     elif v < 0: colors.iloc[i, 3:] = unfav
# #                                 elif title == "4. Stockout Risk & Vulnerability Matrix":
# #                                     if "Fill Rate" in metric:
# #                                         if pol_vals[i] > act_vals[i]: colors.iloc[i, 3:] = fav
# #                                         elif pol_vals[i] < act_vals[i]: colors.iloc[i, 3:] = unfav
# #                                     else:
# #                                         if v > 0: colors.iloc[i, 3:] = fav
# #                                         elif v < 0: colors.iloc[i, 3:] = unfav
# #                                 elif title == "5. Inventory Age & Freshness Matrix (FIFO)":
# #                                     if v > 0: colors.iloc[i, 3:] = fav   # Positive delta means policy age is lower (which is favorable/greener)
# #                                     elif v < 0: colors.iloc[i, 3:] = unfav
# #                             return colors
# #                         st.dataframe(m_df.style.apply(apply_matrix_styles, axis=None), use_container_width=True, hide_index=True)

# #                     render_clustered_matrix("1. Financial Breakdown Matrix", ["Annual Ordering Fees ($)", "Annual Storage Carrying Cost ($)", "Financial Penalty from Stockouts ($)", "Total Policy Operating Cost ($)"], [actual_total_ordering_cost, actual_total_holding_cost, actual_lost_sales_financial, actual_total_cost], [optimal_ordering_cost, optimal_holding_cost, optimal_lost_sales_financial, optimal_total_cost], ["currency", "currency", "currency", "currency"])
# #                     render_clustered_matrix("2. Logistical Operations Footprint Matrix", ["Average Volume Kept On-Hand", "Maximum Storage Spike Level", "Total Orders Dispatched", "Average Logistics Cycle Time", "Average Order Shipment Size"], [actual_avg_inventory, actual_max_inventory, actual_orders_placed, actual_cycle_time, actual_avg_order_size], [simmed_avg_opt_inv, simmed_max_opt_inv, opt_orders_placed, policy_cycle_time, policy_avg_order_size], ["units", "units", "count", "days", "units"])
# #                     render_clustered_matrix("3. Working Capital Release Matrix", ["Peak Working Capital Tied Up ($)", "Average Working Capital Tied Up ($)", "Minimum Base Working Capital ($)"], [act_max_wc, act_avg_wc, act_min_wc], [opt_max_wc, opt_avg_wc, opt_min_wc], ["currency", "currency", "currency"])
# #                     render_clustered_matrix("4. Stockout Risk & Vulnerability Matrix", ["Absolute Minimum Buffer Stock", "Stockout Events (Unfulfilled Days)", "Total Unfulfilled Deficit Volume", "Days with Absolute Zero Closing Stock", "Achieved Order Fill Rate (%)"], [actual_min_inventory, stockout_days_act, lost_sales_qty_act, zero_stock_days_act, actual_fill_rate * 100], [simmed_min_inventory, stockout_days_opt, lost_sales_qty_opt, zero_stock_days_opt, simmed_opt_fill_rate * 100], ["units", "count", "units", "count", "pct"])
# #                     render_clustered_matrix("5. Inventory Age & Freshness Matrix (FIFO)", ["Overall Average Inventory Age (Days)", "Maximum Peak Inventory Age (Days)"], [actual_overall_avg_age, actual_overall_max_age], [opt_overall_avg_age, opt_overall_max_age], ["days", "days"])

# #                     st.markdown("---")
# #                     st.markdown("### 📈 Tactical Operations Timeline Visualizations")
# #                     timeline_fig = go.Figure()
# #                     timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_act, name="Historical Actuals (Ledger)", line=dict(color='#B0C4DE', width=2), fill='tozeroy', fillcolor='rgba(176, 196, 222, 0.15)'))
# #                     timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_opt, name=f"Recommended Optimized Policy ({best_fit_name.split(' ')[0]})", line=dict(color='#1F77B4', width=2.5)))
# #                     timeline_fig.add_trace(go.Scatter(x=df["Date"], y=[max(0, raw_target_level - risk_mean)] * len(df), name="Calculated Safety Stock Floor", line=dict(color='#FF4B4B', width=1.5, dash='dot')))
# #                     timeline_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
# #                     st.plotly_chart(timeline_fig, use_container_width=True)

# #                     st.markdown("---")
# #                     st.markdown("### 🕰️ Inventory Age Profile (FIFO Stacked)")
# #                     st.markdown("Visualize what fraction of your total inventory sitting in the warehouse belongs to distinct aging brackets over time.")
                    
# #                     bucket_labels = ["0-30 Days", "31-60 Days", "61-90 Days", "90+ Days Old"]
# #                     bucket_colors = ["#2CA02C", "#FF7F0E", "#D62728", "#9467BD"] 
                    
# #                     age_tab1, age_tab2 = st.tabs(["📉 Historical Actuals Age Profile", "⚙️ Optimized Policy Age Profile"])
                    
# #                     with age_tab1:
# #                         fig_act_age = go.Figure()
# #                         for j in range(4):
# #                             fig_act_age.add_trace(go.Scatter(
# #                                 x=df["Date"], y=buckets_act[:, j], name=bucket_labels[j],
# #                                 mode='lines', stackgroup='one', line=dict(width=0.5, color=bucket_colors[j])
# #                             ))
# #                         fig_act_age.update_layout(template="plotly_white", yaxis_title="Units In Stock", xaxis_title="Date", height=350)
# #                         st.plotly_chart(fig_act_age, use_container_width=True)
                        
# #                     with age_tab2:
# #                         fig_opt_age = go.Figure()
# #                         for j in range(4):
# #                             fig_opt_age.add_trace(go.Scatter(
# #                                 x=df["Date"], y=buckets_opt[:, j], name=bucket_labels[j],
# #                                 mode='lines', stackgroup='one', line=dict(width=0.5, color=bucket_colors[j])
# #                             ))
# #                         fig_opt_age.update_layout(template="plotly_white", yaxis_title="Units In Stock", xaxis_title="Date", height=350)
# #                         st.plotly_chart(fig_opt_age, use_container_width=True)

# #                     # ==========================================
# #                     #      SECTION B: COMPARATIVE ANALYSIS
# #                     # ==========================================
# #                     st.markdown("---")
# #                     st.header("🔬 Section B: Multi-Scenario Comparative Analysis")
# #                     st.markdown(
# #                         "Leveraging the high-speed vectorized simulation engine, you can now backtest and compare up to 6 "
# #                         "different inventory policies simultaneously. By adjusting these mechanical levers, you can easily identify "
# #                         "operational blind spots without manually tracking the math."
# #                     )
                    
# #                     active_scenarios_list = []
                    
# #                     tab_cont, tab_per = st.tabs(["📉 Continuous Review (Q, R) Scenarios", "⏳ Periodic Review (P, T) Scenarios"])
                    
# #                     with tab_cont:
# #                         st.markdown(f"**Baseline Intelligence:** The data-driven optimal benchmark is **Q: {int(raw_optimal_q):,}** and **ROP: {int(raw_target_level):,}**.")
                        
# #                         c1, c2, c3 = st.columns(3)
                        
# #                         def create_cont_box(col, num, default_q, default_r):
# #                             with col:
# #                                 st.markdown(f"##### 🎛️ Scenario C{num}")
# #                                 run_c = st.toggle(f"Include C{num} in Chart", key=f"run_c{num}", value=(num==1))
# #                                 q_val = st.number_input("Order Qty (Q)", min_value=1, value=int(default_q), step=10, key=f"q_c{num}")
# #                                 r_val = st.number_input("Reorder Point (ROP)", min_value=0, value=int(default_r), step=10, key=f"r_c{num}")
                                
# #                                 if run_c:
# #                                     active_scenarios_list.append({
# #                                         "Case Name": f"C{num} (Q:{q_val}, R:{r_val})", 
# #                                         "Policy Type": "Continuous Review (Q, R)", 
# #                                         "P1": q_val, 
# #                                         "P2": r_val
# #                                     })

# #                         create_cont_box(c1, 1, raw_optimal_q, raw_target_level)
# #                         create_cont_box(c2, 2, raw_optimal_q * 1.5, raw_target_level)
# #                         create_cont_box(c3, 3, raw_optimal_q, raw_target_level * 1.2)

# #                     with tab_per:
# #                         st.markdown("**Baseline Intelligence:** Adjust the Review Period (P) below. The engine will dynamically calculate a safe expected Target Level (T) for that exact timeframe.")
                        
# #                         p1, p2, p3 = st.columns(3)
                        
# #                         def create_per_box(col, num, default_p):
# #                             with col:
# #                                 st.markdown(f"##### 🎛️ Scenario P{num}")
# #                                 run_p = st.toggle(f"Include P{num} in Chart", key=f"run_p{num}", value=False)
# #                                 p_val = st.number_input("Review Period (P Days)", min_value=1, value=int(default_p), step=1, key=f"p_p{num}")
                                
# #                                 target_guide = int(raw_target_level + (avg_daily_demand_calc * p_val))
                                
# #                                 t_val = st.number_input("Target Level (T)", min_value=0, value=target_guide, step=10, key=f"t_p{num}")
# #                                 st.caption(f"💡 *Engine recommended (T) for {p_val} days: **~{target_guide:,}***")
                                
# #                                 if run_p:
# #                                     active_scenarios_list.append({
# #                                         "Case Name": f"P{num} (P:{p_val}, T:{t_val})", 
# #                                         "Policy Type": "Periodic Review (P, T)", 
# #                                         "P1": p_val, 
# #                                         "P2": t_val
# #                                     })

# #                         create_per_box(p1, 1, 7)
# #                         create_per_box(p2, 2, 14)
# #                         create_per_box(p3, 3, 30)

# #                     st.markdown("---")
                    
# #                     if st.button("🚀 Compare Scenarios", type="primary", use_container_width=True):
# #                         if not active_scenarios_list:
# #                             st.warning("Please toggle 'Include' for at least one scenario above to generate the comparison.")
# #                         else:
# #                             summary_data = []
# #                             comp_fig = go.Figure()
                            
# #                             comp_fig.add_trace(go.Scatter(
# #                                 x=df["Date"], y=inv_levels_act, mode='lines', 
# #                                 name="Historical Actuals", line=dict(color='rgba(176, 196, 222, 0.4)', width=2, dash='dot')
# #                             ))

# #                             summary_data.append({
# #                                 "Scenario Blueprint": "📊 Historical Actuals (Baseline)",
# #                                 "Total Op Cost ($)": actual_total_cost,
# #                                 "Fill Rate (%)": actual_fill_rate * 100,
# #                                 "Avg Inv (Units)": actual_avg_inventory,
# #                                 "Avg Working Capital ($)": act_avg_wc,
# #                                 "Max Peak Capital ($)": act_max_wc,
# #                                 "Avg Age (Days)": np.mean(avg_age_act),
# #                                 "Peak Age (Days)": np.max(max_age_act)
# #                             })

# #                             color_palette = px.colors.qualitative.D3
                            
# #                             for index, scenario in enumerate(active_scenarios_list):
# #                                 case_name = scenario["Case Name"]
# #                                 p_type = scenario["Policy Type"]
# #                                 val1 = scenario["P1"]
# #                                 val2 = scenario["P2"]
                                
# #                                 s_inv, s_lost, s_orders, s_avg_age, s_max_age, s_buckets = fast_simulate_inventory(
# #                                     demand_arr_main, purchase_arr_main, opening_stock_override, 
# #                                     lead_time_days, p_type, val1, val2
# #                                 )
                                
# #                                 s_lost_sum = s_lost.sum()
# #                                 s_orders_count = np.count_nonzero(s_orders)
# #                                 s_avg_inv = np.mean(s_inv)
# #                                 s_max_inv = np.max(s_inv)
                                
# #                                 s_fill_rate = max(0.0, 1.0 - (s_lost_sum / max(1, total_demand)))
# #                                 s_total_cost = (s_orders_count * ordering_cost) + (s_avg_inv * unit_holding_cost) + (s_lost_sum * lost_sales_penalty)
                                
# #                                 summary_data.append({
# #                                     "Scenario Blueprint": case_name,
# #                                     "Total Op Cost ($)": s_total_cost,
# #                                     "Fill Rate (%)": s_fill_rate * 100,
# #                                     "Avg Inv (Units)": s_avg_inv,
# #                                     "Avg Working Capital ($)": s_avg_inv * item_unit_cost,
# #                                     "Max Peak Capital ($)": s_max_inv * item_unit_cost,
# #                                     "Avg Age (Days)": np.mean(s_avg_age),
# #                                     "Peak Age (Days)": np.max(s_max_age)
# #                                 })
                                
# #                                 comp_fig.add_trace(go.Scatter(
# #                                     x=df["Date"], y=s_inv, mode='lines', 
# #                                     name=case_name, line=dict(color=color_palette[index % len(color_palette)], width=2.5)
# #                                 ))

# #                             st.markdown("##### 🏆 Comparative Outcomes Scorecard")
# #                             comp_df = pd.DataFrame(summary_data)
                            
# #                             def highlight_baseline(s):
# #                                 return ['background-color: rgba(176, 196, 222, 0.15)' if s['Scenario Blueprint'] == "📊 Historical Actuals (Baseline)" else '' for v in s]

# #                             st.dataframe(
# #                                 comp_df.style.apply(highlight_baseline, axis=1),
# #                                 use_container_width=True,
# #                                 hide_index=True,
# #                                 column_config={
# #                                     "Total Op Cost ($)": st.column_config.NumberColumn(format="$%.2f"),
# #                                     "Fill Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
# #                                     "Avg Inv (Units)": st.column_config.NumberColumn(format="%d"),
# #                                     "Avg Working Capital ($)": st.column_config.NumberColumn(format="$%.0f"),
# #                                     "Max Peak Capital ($)": st.column_config.NumberColumn(format="$%.0f"),
# #                                     "Avg Age (Days)": st.column_config.NumberColumn(format="%.1f"),
# #                                     "Peak Age (Days)": st.column_config.NumberColumn(format="%.0f")
# #                                 }
# #                             )
                            
# #                             st.markdown("##### 📈 Strategic Trajectory Matrix")
# #                             comp_fig.update_layout(
# #                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
# #                                 xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", 
# #                                 height=450, legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
# #                             )
# #                             st.plotly_chart(comp_fig, use_container_width=True)




# with tab5:
#         st.header("⚖️ Advanced Inventory Optimization Suite")
#         st.markdown(
#             "Analyze your inventory data through a twin-lens framework. First, review a historical backtest audit "
#             "to identify legacy profit leaks."
#         )
        
#         # --- 🚀 THE VECTORIZED SIMULATION ENGINE (DYNAMIC FIFO AGE BUCKETS) ---
#         def fast_simulate_inventory(demand_arr, purchase_arr, opening_stock, lead_time, policy_type, param1, param2, age_bucket_edges=[30, 60, 90]):
#             total_days = len(demand_arr)
#             inv_levels = np.zeros(total_days)
#             lost_sales = np.zeros(total_days)
#             orders_placed = np.zeros(total_days)
            
#             # Split positive receipts from negative returns to apply write-offs universally
#             positive_receipts = np.where(purchase_arr > 0, purchase_arr, 0)
#             negative_returns = np.where(purchase_arr < 0, np.abs(purchase_arr), 0)
            
#             # FIFO Age Tracking Arrays
#             avg_ages = np.zeros(total_days)
#             max_ages = np.zeros(total_days)
            
#             num_buckets = len(age_bucket_edges) + 1
#             age_buckets = np.zeros((total_days, num_buckets)) 
            
#             max_lt = int(lead_time)
#             pipeline = np.zeros(total_days + max_lt + 1)
            
#             if policy_type == "Actual":
#                 pipeline[:total_days] = positive_receipts
#             else:
#                 warm_up = min(max_lt, total_days)
#                 pipeline[:warm_up] = positive_receipts[:warm_up]
                
#             current_inv = opening_stock
#             fifo_queue = [[0, opening_stock]] if opening_stock > 0 else []
            
#             for i in range(total_days):
#                 demand = demand_arr[i]
#                 arriving = pipeline[i]
#                 daily_return = negative_returns[i]
                
#                 # 1. Process Positive Arriving Stock
#                 if arriving > 0:
#                     fifo_queue.append([i, arriving])
#                     current_inv += arriving
                    
#                 # 2. Mechanically Deduct Returns / Expiries / Write-offs
#                 if daily_return > 0:
#                     current_inv = max(0, current_inv - daily_return)
#                     # Deduct from FIFO queue to clear out the oldest physical stock
#                     while daily_return > 0 and fifo_queue:
#                         oldest_batch_qty = fifo_queue[0][1]
#                         if oldest_batch_qty <= daily_return:
#                             daily_return -= oldest_batch_qty
#                             fifo_queue.pop(0)
#                         else:
#                             fifo_queue[0][1] -= daily_return
#                             daily_return = 0
                            
#                 # 3. Fulfill Valid Customer Demand
#                 demand_left = demand
#                 while demand_left > 0 and fifo_queue:
#                     oldest_batch_qty = fifo_queue[0][1]
#                     if oldest_batch_qty <= demand_left:
#                         demand_left -= oldest_batch_qty
#                         fifo_queue.pop(0) 
#                     else:
#                         fifo_queue[0][1] -= demand_left
#                         demand_left = 0 
                        
#                 if demand_left > 0:
#                     lost_sales[i] = demand_left
#                     current_inv = 0
#                 else:
#                     current_inv -= demand
                    
#                 inv_levels[i] = current_inv
                
#                 # Calculate Age Metrics & Dynamic Buckets
#                 if fifo_queue:
#                     total_qty = 0
#                     sum_age_qty = 0
#                     max_age_val = 0
                    
#                     for item in fifo_queue:
#                         arr_day, qty = item
#                         age = i - arr_day
#                         total_qty += qty
#                         sum_age_qty += (age * qty)
                        
#                         if age > max_age_val:
#                             max_age_val = age
                            
#                         # Dynamic bucket sorting
#                         idx = 0
#                         while idx < len(age_bucket_edges) and age > age_bucket_edges[idx]:
#                             idx += 1
#                         age_buckets[i, idx] += qty
                            
#                     if total_qty > 0:
#                         avg_ages[i] = sum_age_qty / total_qty
#                         max_ages[i] = max_age_val
                
#                 if policy_type == "Continuous Review (Q, R)":
#                     net_position = current_inv + np.sum(pipeline[i+1:])
#                     if net_position <= param2:
#                         pipeline[i + max_lt] += param1
#                         orders_placed[i] = param1
#                 elif policy_type == "Periodic Review (P, T)":
#                     if i % int(param1) == 0:
#                         net_position = current_inv + np.sum(pipeline[i+1:])
#                         order_qty = max(0, param2 - net_position)
#                         if order_qty > 0:
#                             pipeline[i + max_lt] += order_qty
#                             orders_placed[i] = order_qty
                            
#             return inv_levels, lost_sales, orders_placed, avg_ages, max_ages, age_buckets

#         # --- STEP 1: INPUT PARAMETERS ---
#         st.subheader("1. Parameters & Cost Drivers")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             item_unit_cost = st.number_input("Item Unit Cost ($/Unit)", min_value=0.01, value=25.00, step=1.00, key="unit_cost_global")
#             holding_fixed_daily = st.number_input("Fixed Holding Cost ($/Unit/Day)", min_value=0.0, value=0.0, step=0.01, key="fixed_hold_global")
            
#         with col2:
#             holding_var_pct = st.number_input("Variable Holding Cost (% of Item Cost/year)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="var_hold_global") / 100.0
#             ordering_cost = st.number_input("Ordering Cost ($/order)", min_value=0.1, value=75.0, step=5.0, key="order_cost_global")
            
#         with col3:
#             lost_sales_penalty = st.number_input("Lost Sales Penalty ($/Unit Lost)", min_value=0.0, value=0.0, step=1.0, key="penalty_global")
#             lead_time_days = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1, key="lt_global")

#         st.markdown("---")
#         col_sys1, col_sys2 = st.columns([1, 2])
#         with col_sys1:
#             review_system = st.radio("Inventory Review System Strategy", ["Continuous Review (Q, R)", "Periodic Review (P, T)"], key="review_system_global")
#         with col_sys2:
#             service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="service_level_global") / 100.0

#         if review_system == "Periodic Review (P, T)":
#             st.markdown("##### ⏳ Periodic Configuration")
#             p_col1, _ = st.columns(2)
#             with p_col1:
#                 user_p_days = st.number_input("Review Period Cycle (P in Days)", min_value=1, value=14, step=1, key="p_days_global")
#         else:
#             user_p_days = 1

#         # --- STEP 2: MULTI-FORMAT DATA INGESTION ENGINE ---
#         st.subheader("2. Upload Historical Invoices & Demand Data")
#         uploaded_file = st.file_uploader(
#             "Upload Inventory Ledger (Supports standard templates, raw ERP transactional logs, or stock card snapshots)", 
#             type=["csv", "xlsx", "xls"], 
#             key="uploader_global"
#         )
        
#         if uploaded_file is None:
#             st.info("📥 Please upload your inventory ledger file (CSV or Excel) above to populate the suite modules.")
#         else:
#             detected_sheet_opening_stock = None
#             data_loaded_successfully = False
#             df_mapped = None
            
#             try:
#                 if uploaded_file.name.endswith('.csv'):
#                     raw_df = pd.read_csv(uploaded_file)
#                 else:
#                     raw_df = pd.read_excel(uploaded_file)
                    
#                 raw_df.columns = raw_df.columns.str.strip()
                    
#                 if "Date" not in raw_df.columns:
#                     st.error("❌ Missing required column: 'Date'.")
#                 else:
#                     raw_df["Date"] = pd.to_datetime(raw_df["Date"])
#                     raw_df = raw_df.sort_values(by="Date").reset_index(drop=True)
                    
#                     open_balance_headers = ["Opening Balance", "Opening", "Opening_Stock", "Opening Stock"]
#                     for header in open_balance_headers:
#                         if header in raw_df.columns:
#                             detected_sheet_opening_stock = int(raw_df[header].iloc[0])
#                             break
                    
#                     if "Demand_Qty" in raw_df.columns and "Purchase_Qty" in raw_df.columns:
#                         df_mapped = raw_df[["Date", "Demand_Qty", "Purchase_Qty"]].copy()
#                     elif "Demand" in raw_df.columns and "Stock Received" in raw_df.columns:
#                         df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df["Demand"], "Purchase_Qty": raw_df["Stock Received"]})
#                     elif ("Receiving" in raw_df.columns) and any(col in raw_df.columns for col in ["Demand/Sales", "Demand", "Sales"]):
#                         outbound_col = "Demand/Sales" if "Demand/Sales" in raw_df.columns else ("Demand" if "Demand" in raw_df.columns else "Sales")
#                         df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df[outbound_col], "Purchase_Qty": raw_df["Receiving"]})
#                     else:
#                         st.error("❌ Column layout structure mismatch. Could not find Demand and Receiving columns.")

#                     if df_mapped is not None:
#                         df_mapped = df_mapped.groupby("Date").agg({"Demand_Qty": "sum", "Purchase_Qty": "sum"}).reset_index()
#                         df_mapped = df_mapped.set_index("Date").resample("1D").asfreq()
#                         df_mapped["Demand_Qty"] = df_mapped["Demand_Qty"].fillna(0.0)
#                         df_mapped["Purchase_Qty"] = df_mapped["Purchase_Qty"].fillna(0.0)
#                         df_mapped = df_mapped.reset_index()
#                         data_loaded_successfully = True
                        
#             except Exception as e:
#                 st.error(f"Error parsing file elements: {e}")

#             # ONLY PROCEED IF DATA IS CLEANED
#             if data_loaded_successfully and df_mapped is not None:
#                 full_df = df_mapped.copy() 
                
#                 absolute_min_date = full_df["Date"].min().date()
#                 absolute_max_date = full_df["Date"].max().date()
                
#                 file_state_key = f"last_file_{uploaded_file.name}_{uploaded_file.size}"
                
#                 if "current_file_token" not in st.session_state or st.session_state.current_file_token != file_state_key:
#                     st.session_state.current_file_token = file_state_key
#                     st.session_state.min_date_global = absolute_min_date
#                     st.session_state.max_date_global = absolute_max_date
                    
#                     avg_daily_full = full_df["Demand_Qty"].mean()
#                     if detected_sheet_opening_stock is not None:
#                         default_start = int(detected_sheet_opening_stock)
#                     else:
#                         default_start = int(1.25 * (avg_daily_full * lead_time_days))
                    
#                     st.session_state.absolute_day1_stock = default_start
#                     st.session_state.previous_start_date = absolute_min_date
#                     st.session_state.previous_end_date = absolute_max_date
#                     st.session_state.opening_stock_global = default_start
                    
#                     st.session_state.start_date_key = absolute_min_date
#                     st.session_state.end_date_key = absolute_max_date
                    
#                     if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
#                     if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite

#                 st.divider()
                
#                 st.markdown("### 📅 3. Select Analysis Period")
#                 st.markdown("Filter historical data. The starting inventory will automatically mathematically roll forward to match your selected Start Date.")
                
#                 def reset_dates():
#                     st.session_state.start_date_key = st.session_state.min_date_global
#                     st.session_state.end_date_key = st.session_state.max_date_global
#                     st.session_state.previous_start_date = st.session_state.min_date_global
#                     st.session_state.previous_end_date = st.session_state.max_date_global
#                     st.session_state.opening_stock_global = st.session_state.absolute_day1_stock

#                 col_date1, col_date2, col_date3 = st.columns([2, 2, 1])
                
#                 with col_date1:
#                     start_date = st.date_input("Starting Date", min_value=absolute_min_date, max_value=absolute_max_date, key="start_date_key")
#                 with col_date2:
#                     end_date = st.date_input("Ending Date", min_value=absolute_min_date, max_value=absolute_max_date, key="end_date_key")
#                 with col_date3:
#                     st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
#                     st.button("🔄 Reset", on_click=reset_dates, use_container_width=True)
                
#                 if "previous_start_date" not in st.session_state:
#                     st.session_state.previous_start_date = absolute_min_date
#                 if "previous_end_date" not in st.session_state:
#                     st.session_state.previous_end_date = absolute_max_date

#                 date_changed = (start_date != st.session_state.previous_start_date) or (end_date != st.session_state.previous_end_date)

#                 if date_changed:
#                     if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
#                     if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite
                    
#                     if start_date != st.session_state.previous_start_date:
#                         temp_balance = st.session_state.absolute_day1_stock
#                         pre_period_df = full_df[full_df["Date"].dt.date < start_date]
                        
#                         pre_d_arr = pre_period_df["Demand_Qty"].values
#                         pre_p_arr = pre_period_df["Purchase_Qty"].values
#                         for idx in range(len(pre_d_arr)):
#                             temp_balance = max(0, temp_balance + pre_p_arr[idx] - pre_d_arr[idx])
                        
#                         st.session_state.opening_stock_global = int(temp_balance)
                    
#                     st.session_state.previous_start_date = start_date
#                     st.session_state.previous_end_date = end_date

#                 if start_date > end_date:
#                     st.error("⚠️ The Starting Date must be before or equal to the Ending Date. Please adjust your selection.")
#                 else:
#                     df = full_df[(full_df["Date"].dt.date >= start_date) & (full_df["Date"].dt.date <= end_date)].reset_index(drop=True)
                    
#                     if df.empty:
#                         st.warning("No data available for the selected date range. Please widen your selection.")
#                     else:
#                         demand_arr_main = df["Demand_Qty"].values
#                         purchase_arr_main = df["Purchase_Qty"].values
                        
#                         # Filter for actual positive purchase orders to calculate correct baseline KPIs
#                         actual_orders_placed = np.count_nonzero(purchase_arr_main[purchase_arr_main > 0])
#                         actual_total_units_purchased = purchase_arr_main[purchase_arr_main > 0].sum()
#                         total_demand = demand_arr_main.sum()
                        
#                         avg_daily_demand_calc = demand_arr_main.mean()
#                         std_daily_demand = demand_arr_main.std() if len(df) > 1 else 0
#                         cov = std_daily_demand / max(0.1, avg_daily_demand_calc)

#                     with st.expander("📈 View Historical Demand Trend & Growth Timeline", expanded=False):
#                         rolling_days = st.slider("Select Rolling Average Window (Days)", min_value=1, max_value=90, value=15, step=1)
#                         df[f"Rolling_Avg"] = df["Demand_Qty"].rolling(window=rolling_days, min_periods=1).mean()
                        
#                         stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
#                         if len(df) >= rolling_days * 2:
#                             current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
#                             previous_window_sum = df["Demand_Qty"].iloc[-(rolling_days*2):-rolling_days].sum()
#                             if previous_window_sum > 0:
#                                 trend_pct = ((current_window_sum - previous_window_sum) / previous_window_sum) * 100
#                             else:
#                                 trend_pct = 100.0 if current_window_sum > 0 else 0.0
#                             stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta=f"{trend_pct:+.1f}% Growth", delta_color="normal")
#                         elif len(df) >= rolling_days:
#                             current_window_sum = df["Demand_Qty"].iloc[-rolling_days:].sum()
#                             stat_col1.metric(label=f"Trend (Last {rolling_days} Days)", value=f"{current_window_sum:,.0f} units", delta="Widen dates for trend", delta_color="off")
#                         else:
#                             stat_col1.metric(label="Trend", value="N/A", delta="Insufficient Data", delta_color="off")
                            
#                         stat_col2.metric("Average Daily Demand", f"{avg_daily_demand_calc:.1f} units")
#                         stat_col3.metric("Standard Deviation", f"{std_daily_demand:.1f} units")
#                         stat_col4.metric("Volatility (CoV)", f"{cov:.2f}")
#                         st.markdown("---")

#                         demand_fig = go.Figure()
#                         demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Demand_Qty"], mode='lines', name='Raw Daily Demand', line=dict(color='#B0C4DE', width=1.5), opacity=0.6))
#                         demand_fig.add_trace(go.Scatter(x=df["Date"], y=df["Rolling_Avg"], mode='lines', name=f'{rolling_days}-Day Moving Avg', line=dict(color='#3d5a80', width=3)))
#                         demand_fig.update_layout(template="plotly_white", yaxis_title="Units", xaxis_title="Date", margin=dict(t=20, b=20), height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
#                         st.plotly_chart(demand_fig, use_container_width=True)

#                     st.markdown("---")
#                     st.markdown("##### 📦 Initial Warehouse Capital Balance")
#                     opening_stock_override = st.number_input("Starting Balance for Selected Period", min_value=0, step=10, key="opening_stock_global", help="This value automatically updates mathematically based on the start date you select above, but you can override it manually.")

#                     with st.expander("📋 View Complete Running Balance Table Snapshots", expanded=False):
#                         st.markdown("An interactive historical stock card ledger driven directly by your initial opening stock parameter above.")
#                         cl_open_list, cl_close_list, t_bal = np.zeros(len(df)), np.zeros(len(df)), opening_stock_override
                        
#                         for i_run in range(len(df)):
#                             cl_open_list[i_run] = t_bal
#                             t_bal = max(0, t_bal + purchase_arr_main[i_run] - demand_arr_main[i_run])
#                             cl_close_list[i_run] = t_bal
                            
#                         full_stock_card_df = pd.DataFrame({
#                             "Timeline Date": df["Date"].dt.strftime('%Y-%m-%d'),
#                             "Opening Balance": cl_open_list.astype(int),
#                             "Cleaned Demand Volume (Units)": demand_arr_main.astype(int),
#                             "Consolidated Stock Received (Units)": purchase_arr_main.astype(int),
#                             "Closing Balance": cl_close_list.astype(int)
#                         })
                        
#                         st.dataframe(full_stock_card_df, use_container_width=True, hide_index=True, column_config={"Opening Balance": st.column_config.NumberColumn(format="%d"), "Cleaned Demand Volume (Units)": st.column_config.NumberColumn(format="%d"), "Consolidated Stock Received (Units)": st.column_config.NumberColumn(format="%d"), "Closing Balance": st.column_config.NumberColumn(format="%d")})

#                     st.subheader("4. Statistical Risk & Distribution Engines")
                    
#                     annual_demand = avg_daily_demand_calc * 365
#                     annual_fixed_holding_per_unit = holding_fixed_daily * 365
#                     unit_holding_cost = annual_fixed_holding_per_unit + (item_unit_cost * holding_var_pct)
                    
#                     risk_horizon_days = lead_time_days if review_system == "Continuous Review (Q, R)" else (user_p_days + lead_time_days)
#                     rolling_risk_demand = df["Demand_Qty"].rolling(window=int(risk_horizon_days)).sum().dropna().values
#                     risk_mean = np.mean(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0
#                     risk_std = np.std(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0

#                     if len(rolling_risk_demand) > 0:
#                         if np.max(rolling_risk_demand) <= 0:
#                             best_fit_name = "Zero Demand Base"
#                             raw_target_level = 0.0
#                         else:
#                             empirical_rop_raw = np.percentile(rolling_risk_demand, service_level * 100)
#                             safe_demand = np.where(rolling_risk_demand <= 0, 1e-5, rolling_risk_demand)
                            
#                             log_params = stats.lognorm.fit(safe_demand, floc=0)
#                             gam_params = stats.gamma.fit(safe_demand, floc=0)

#                             counts, bins = np.histogram(rolling_risk_demand, bins=20, density=True)
#                             bin_centers = (bins[:-1] + bins[1:]) / 2
                            
#                             rss_norm = np.sum((counts - stats.norm.pdf(bin_centers, loc=risk_mean, scale=risk_std)) ** 2)
#                             rss_log = np.sum((counts - stats.lognorm.pdf(bin_centers, *log_params)) ** 2)
#                             rss_gam = np.sum((counts - stats.gamma.pdf(bin_centers, *gam_params)) ** 2)

#                             if cov > 0.75:
#                                 best_fit_name = "Empirical (Data-Driven)"
#                                 raw_target_level = empirical_rop_raw
#                             else:
#                                 errors = {"Normal": rss_norm, "Log-Normal": rss_log, "Gamma": rss_gam}
#                                 best_fit_name = min(errors, key=errors.get)
#                                 if best_fit_name == "Normal":
#                                     raw_target_level = stats.norm.ppf(service_level, loc=risk_mean, scale=risk_std)
#                                 elif best_fit_name == "Log-Normal":
#                                     raw_target_level = stats.lognorm.ppf(service_level, *log_params)
#                                 else:
#                                     raw_target_level = stats.gamma.ppf(service_level, *gam_params)
#                     else:
#                         best_fit_name = "Default (Insufficient Data)"
#                         raw_target_level = avg_daily_demand_calc * risk_horizon_days
                        
#                     raw_optimal_q = np.sqrt((2 * annual_demand * ordering_cost) / max(0.01, unit_holding_cost))

#                     if "q_audit_suite" not in st.session_state:
#                         st.session_state.q_audit_suite = max(1, int(raw_optimal_q)) if review_system == "Continuous Review (Q, R)" else int(avg_daily_demand_calc * user_p_days)
#                     if "rop_audit_suite" not in st.session_state:
#                         st.session_state.rop_audit_suite = max(0, int(raw_target_level))

#                     with st.expander("📊 View Cleaned Demand Distribution & Best-Fit Curve Metrics", expanded=False):
#                         stat_col1, stat_col2, stat_col3 = st.columns(3)
#                         stat_col1.metric("Average Daily Demand", f"{avg_daily_demand_calc:.2f} units")
#                         stat_col2.metric("Coefficient of Variation (CV)", f"{cov:.2f}")
#                         stat_col3.metric("Engine Selection", f"✨ {best_fit_name}")
#                         st.markdown("---")
#                         hist_fig = go.Figure()
#                         hist_fig.add_trace(go.Histogram(x=df["Demand_Qty"], name="Historical Days", marker_color='#1F77B4', opacity=0.6))
#                         hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Demand Quantity (Units / Day)", yaxis_title="Frequency", margin=dict(l=40, r=40, t=10, b=40), height=300)
#                         st.plotly_chart(hist_fig, use_container_width=True)

#                     st.markdown("---")
#                     st.subheader("5. 🌪️ Erratic Demand & Empirical ROP Profiler")
                    
#                     with st.expander("Open Rolling Window & Empirical Profiler", expanded=False):
#                         st.markdown(
#                             "Standard safety stock math assumes demand follows a predictable bell curve. For erratic or lumpy demand, "
#                             "that assumption breaks down. This tool mechanically profiles your exact historical risk by analyzing every "
#                             "rolling vulnerability window in your dataset."
#                         )
                        
#                         tab_emp_cont, tab_emp_per = st.tabs(["📉 Continuous Review (ROP)", "⏳ Periodic Review (Target Level)"])
                        
#                         with tab_emp_cont:
#                             st.markdown("##### Continuous Review Risk Profiler")
#                             col_emp1, col_emp2, col_emp3 = st.columns(3)
#                             with col_emp1:
#                                 emp_lt_window = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_lt_window")
#                             with col_emp2:
#                                 emp_test_rop = st.number_input("Test Reorder Point (ROP)", min_value=0, value=int(raw_target_level), step=10, key="emp_test_rop")
#                             with col_emp3:
#                                 emp_target_sl = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl")

#                             rolling_demand_c = df["Demand_Qty"].rolling(window=emp_lt_window).sum().dropna()

#                             if len(rolling_demand_c) > 0:
#                                 windows_below_rop = np.sum(rolling_demand_c <= emp_test_rop)
#                                 total_windows_c = len(rolling_demand_c)
#                                 achieved_sl_c = (windows_below_rop / total_windows_c) * 100
#                                 required_rop = np.percentile(rolling_demand_c, emp_target_sl)

#                                 res_col1, res_col2 = st.columns(2)
#                                 with res_col1:
#                                     st.info(f"**Testing ROP of {emp_test_rop:,}:**\n\nOut of {total_windows_c:,} historical {emp_lt_window}-day windows, the total demand was successfully covered by {emp_test_rop:,} units exactly **{windows_below_rop:,} times**. This yields an empirical service level of **{achieved_sl_c:.1f}%**.")
#                                 with res_col2:
#                                     st.success(f"**Targeting {emp_target_sl}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl}% of all historical {emp_lt_window}-day scenarios, your ROP must be set to the empirical percentile: **{int(required_rop):,} units**.")

#                                 emp_fig_c = go.Figure()
#                                 emp_fig_c.add_trace(go.Histogram(x=rolling_demand_c, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {emp_lt_window}-Day Windows"))
#                                 emp_fig_c.add_vline(x=emp_test_rop, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested ROP ({emp_test_rop})", annotation_position="top right")
#                                 emp_fig_c.add_vline(x=required_rop, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Target ROP ({int(required_rop)})", annotation_position="top left")

#                                 emp_fig_c.update_layout(title=f"Actual Demand Distribution Across All {emp_lt_window}-Day Windows", xaxis_title=f"Total Units Demanded in a {emp_lt_window}-Day Window", yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40))
#                                 st.plotly_chart(emp_fig_c, use_container_width=True)
#                             else:
#                                 st.warning(f"⚠️ Not enough data to calculate rolling windows for a {emp_lt_window}-day lead time.")

#                         with tab_emp_per:
#                             st.markdown("##### Periodic Review Risk Profiler")
#                             default_p_val = int(user_p_days) if user_p_days > 1 else 14
#                             default_t_val = int(raw_target_level + (avg_daily_demand_calc * default_p_val))
                            
#                             p_col1, p_col2, p_col3, p_col4 = st.columns(4)
#                             with p_col1:
#                                 emp_p_days = st.number_input("Review Period (P)", min_value=1, value=default_p_val, step=1, key="emp_p_days")
#                             with p_col2:
#                                 emp_p_lt = st.number_input("Lead Time (Days)", min_value=1, value=int(lead_time_days), step=1, key="emp_p_lt")
#                             with p_col3:
#                                 emp_test_t = st.number_input("Test Target Level (T)", min_value=0, value=default_t_val, step=10, key="emp_test_t")
#                             with p_col4:
#                                 emp_target_sl_p = st.number_input("Target Service Level (%)", min_value=1.0, max_value=99.9, value=95.0, step=0.5, key="emp_target_sl_p")

#                             risk_window_days = emp_p_days + emp_p_lt
#                             rolling_demand_p = df["Demand_Qty"].rolling(window=risk_window_days).sum().dropna()

#                             if len(rolling_demand_p) > 0:
#                                 windows_below_t = np.sum(rolling_demand_p <= emp_test_t)
#                                 total_windows_p = len(rolling_demand_p)
#                                 achieved_sl_p = (windows_below_t / total_windows_p) * 100
#                                 required_t = np.percentile(rolling_demand_p, emp_target_sl_p)

#                                 res_col3, res_col4 = st.columns(2)
#                                 with res_col3:
#                                     st.info(f"**Testing Target (T) of {emp_test_t:,}:**\n\nOut of {total_windows_p:,} historical {risk_window_days}-day windows (P+L), demand was successfully covered by {emp_test_t:,} units exactly **{windows_below_t:,} times**. Empirical SL: **{achieved_sl_p:.1f}%**.")
#                                 with res_col4:
#                                     st.success(f"**Targeting {emp_target_sl_p}% Service Level:**\n\nTo mechanically guarantee that you don't stock out in {emp_target_sl_p}% of all historical {risk_window_days}-day scenarios, your Target Level must be: **{int(required_t):,} units**.")

#                                 emp_fig_p = go.Figure()
#                                 emp_fig_p.add_trace(go.Histogram(x=rolling_demand_p, nbinsx=40, marker_color='#B0C4DE', name=f"Historical {risk_window_days}-Day Windows"))
#                                 emp_fig_p.add_vline(x=emp_test_t, line_width=2, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Tested Target ({emp_test_t})", annotation_position="top right")
#                                 emp_fig_p.add_vline(x=required_t, line_width=2, line_dash="dash", line_color="#1F77B4", annotation_text=f"Required Target ({int(required_t)})", annotation_position="top left")

#                                 emp_fig_p.update_layout(title=f"Actual Demand Distribution Across All {risk_window_days}-Day (P+L) Windows", xaxis_title=f"Total Units Demanded in a {risk_window_days}-Day Window", yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=40))
#                                 st.plotly_chart(emp_fig_p, use_container_width=True)
#                             else:
#                                 st.warning(f"⚠️ Not enough data to calculate rolling windows for a {risk_window_days}-day (P+L) window.")

#                     st.markdown("---")
#                     st.subheader("6. Policy Control & Aging Parameters")
                    
#                     adjust_col1, adjust_col2 = st.columns(2)
#                     with adjust_col1:
#                         if review_system == "Continuous Review (Q, R)":
#                             final_q = st.number_input("Target Order Quantity (Q)", min_value=1, step=10, key="q_audit_suite")
#                         else:
#                             cycle_demand_baseline = max(1, int(avg_daily_demand_calc * user_p_days))
#                             final_q = st.number_input("Average Target Batch Size (Q)", min_value=1, value=cycle_demand_baseline, step=10, disabled=True, key="q_audit_suite_disabled")
#                     with adjust_col2:
#                         final_buffer_target = st.number_input("Reorder Point (ROP) / Target Level (T)", min_value=0, step=10, key="rop_audit_suite")

#                     if review_system == "Continuous Review (Q, R)":
#                         st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Optimal Order Quantity (EOQ): **{int(raw_optimal_q):,}** units | Recommended Reorder Point (ROP): **{int(raw_target_level):,}** units.")
#                     else:
#                         st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Expected Cycle Batch Size: **{int(avg_daily_demand_calc * user_p_days):,}** units | Recommended Max Order Up-To Level (T): **{int(raw_target_level):,}** units.")

#                     optimal_p_days = max(1, int((final_q / max(0.1, avg_daily_demand_calc)))) if review_system == "Continuous Review (Q, R)" else int(user_p_days)

#                     # --- UI CONTAINERS TO ENFORCE RENDERING ORDER ---
#                     header_container = st.container()
#                     kpi_container = st.container()
#                     matrix_container = st.container()
#                     timeline_container = st.container()
#                     age_input_container = st.container()
#                     age_profile_container = st.container()
#                     drilldown_container = st.container()

#                     # 1. Capture the age bucket definitions (renders physically in the 5th container, evaluates here logically)
#                     with age_input_container:
#                         st.markdown("---")
#                         st.markdown("### 🕰️ Inventory Age Profile (FIFO Stacked)")
#                         st.markdown("Visualize what fraction of your total inventory sitting in the warehouse belongs to distinct aging brackets over time.")
#                         bucket_input = st.text_input("Define custom inventory age thresholds in days (comma-separated, e.g., '15, 30, 45, 60, 90')", value="30, 60, 90")
                        
#                     try:
#                         custom_edges = sorted(list(set([int(x.strip()) for x in bucket_input.split(',') if x.strip().isdigit()])))
#                         if not custom_edges: custom_edges = [30, 60, 90]
#                     except:
#                         custom_edges = [30, 60, 90]

#                     bucket_labels = []
#                     prev_edge = 0
#                     for edge in custom_edges:
#                         bucket_labels.append(f"{prev_edge}-{edge} Days")
#                         prev_edge = edge + 1
#                     bucket_labels.append(f"{prev_edge}+ Days")

#                     num_buckets = len(bucket_labels)
                    
#                     if num_buckets == 1:
#                         bucket_colors = ["#8EC9FF"]
#                     else:
#                         bucket_colors = px.colors.sample_colorscale(
#                             [
#                                 [0.0, "#8EC9FF"],
#                                 [0.4, "#1E88E5"],
#                                 [0.7, "#FFA726"],
#                                 [1.0, "#E53935"]
#                             ],
#                             [i/(num_buckets-1) for i in range(num_buckets)]
#                         )

#                     # 2. RUN SIMULATIONS (Hidden from UI, populates downstream components)
#                     with header_container:
#                         st.markdown("---")
#                         st.header("📊 Section A: Historical Backtest Audit")
#                         st.markdown("This analysis compares your **Historical Actuals** against our **Recommended Optimized Policy** under identical historical demand constraints to reveal operational friction.")

#                     # Actuals
#                     inv_levels_act, lost_sales_act_arr, orders_placed_act_arr, avg_age_act, max_age_act, buckets_act = fast_simulate_inventory(
#                         demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, "Actual", 0, 0, custom_edges
#                     )
                    
#                     # Optimized
#                     inv_levels_opt, lost_sales_opt_arr, orders_placed_opt_arr, avg_age_opt, max_age_opt, buckets_opt = fast_simulate_inventory(
#                         demand_arr_main, purchase_arr_main, opening_stock_override, lead_time_days, review_system, 
#                         optimal_p_days if review_system != "Continuous Review (Q, R)" else final_q, final_buffer_target, custom_edges
#                     )

#                     # --- CALCULATE METRICS ---
#                     lost_sales_qty_act = lost_sales_act_arr.sum()
#                     stockout_days_act = np.count_nonzero(lost_sales_act_arr)
#                     zero_stock_days_act = np.count_nonzero(inv_levels_act == 0)
                    
#                     actual_max_inventory = np.max(inv_levels_act)
#                     actual_min_inventory = np.min(inv_levels_act)
#                     actual_avg_inventory = np.mean(inv_levels_act)
#                     actual_fill_rate = max(0.0, 1.0 - (lost_sales_qty_act / max(1, total_demand)))
#                     actual_cycle_time = 365 / actual_orders_placed if actual_orders_placed > 0 else 365.0
#                     actual_avg_order_size = actual_total_units_purchased / actual_orders_placed if actual_orders_placed > 0 else 0.0

#                     actual_total_ordering_cost = actual_orders_placed * ordering_cost
#                     actual_total_holding_cost = actual_avg_inventory * unit_holding_cost
#                     actual_lost_sales_financial = lost_sales_qty_act * lost_sales_penalty
#                     actual_total_cost = actual_total_ordering_cost + actual_total_holding_cost + actual_lost_sales_financial
#                     actual_overall_avg_age = np.mean(avg_age_act)
#                     actual_overall_max_age = np.max(max_age_act)

#                     lost_sales_qty_opt = lost_sales_opt_arr.sum()
#                     stockout_days_opt = np.count_nonzero(lost_sales_opt_arr)
#                     zero_stock_days_opt = np.count_nonzero(inv_levels_opt == 0)
#                     opt_orders_placed = np.count_nonzero(orders_placed_opt_arr)
#                     policy_total_units_ordered = orders_placed_opt_arr.sum()

#                     simmed_avg_opt_inv = np.mean(inv_levels_opt)
#                     simmed_max_opt_inv = np.max(inv_levels_opt)
#                     simmed_min_inventory = np.min(inv_levels_opt)
#                     simmed_opt_fill_rate = max(0.0, 1.0 - (lost_sales_qty_opt / max(1, total_demand)))
#                     policy_cycle_time = 365 / opt_orders_placed if opt_orders_placed > 0 else 365.0
#                     policy_avg_order_size = policy_total_units_ordered / opt_orders_placed if opt_orders_placed > 0 else 0.0

#                     optimal_ordering_cost = opt_orders_placed * ordering_cost
#                     optimal_holding_cost = simmed_avg_opt_inv * unit_holding_cost
#                     optimal_lost_sales_financial = lost_sales_qty_opt * lost_sales_penalty
#                     optimal_total_cost = optimal_ordering_cost + optimal_holding_cost + optimal_lost_sales_financial
#                     opt_overall_avg_age = np.mean(avg_age_opt)
#                     opt_overall_max_age = np.max(max_age_opt)
                    
#                     act_max_wc = actual_max_inventory * item_unit_cost
#                     act_avg_wc = actual_avg_inventory * item_unit_cost
#                     act_min_wc = actual_min_inventory * item_unit_cost
                    
#                     opt_max_wc = simmed_max_opt_inv * item_unit_cost
#                     opt_avg_wc = simmed_avg_opt_inv * item_unit_cost
#                     opt_min_wc = simmed_min_inventory * item_unit_cost

#                     true_net_benefit = actual_total_cost - optimal_total_cost

#                     # 3. POPULATE UI CONTAINERS
#                     with kpi_container:
#                         if true_net_benefit > 0:
#                             st.success(f"### 🎯 The Efficiency Opportunity\nBy shifting to the recommended optimized policy, you would have recovered **${true_net_benefit:,.2f}** over this historical period.")
#                         else:
#                             st.error(f"⚠️ **Operational Margin Deficit Risk:** This setup increases operational overhead by **${abs(true_net_benefit):,.2f} / year** compared to actuals.")

#                         st.markdown("### 🏆 Executive Summary: Value Realization")
#                         cash_released = act_avg_wc - opt_avg_wc

#                         kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
#                         with kpi_col1: st.metric(label="Total Cost Saving", value=f"${true_net_benefit:,.0f}")
#                         with kpi_col2: st.metric(label="Optimized Fill Rate", value=f"{simmed_opt_fill_rate * 100:.1f}%")
#                         with kpi_col3:
#                             st.metric(label="Avg Working Capital (Opt)", value=f"${opt_avg_wc:,.0f}")
#                             st.markdown(f"<div style='margin-top: -15px; font-size: 0.85rem; color: gray;'>Historical: ${act_avg_wc:,.0f}</div>", unsafe_allow_html=True)
#                         with kpi_col4:
#                             release_label = "Cash Released" if cash_released >= 0 else "Capital Added (Tied Up)"
#                             # st.metric(label=release_label, value=f"${abs(cash_released):,.0f}")
#                             cash_released_pct = (cash_released / act_avg_wc) * 100 if act_avg_wc > 0 else 0.0
#                             st.metric(label=release_label, value=f"${abs(cash_released):,.0f}", delta=f"{cash_released_pct:+.1f}%")
#                         st.markdown("---")

#                     with matrix_container:
#                         def render_clustered_matrix(title, metrics, act_vals, pol_vals, formats):
#                             st.markdown(f"#### {title}")
#                             abs_var = [a - p for a, p in zip(act_vals, pol_vals)]
#                             pct_var = []
#                             for a, p in zip(act_vals, pol_vals):
#                                 if a == 0: pct_var.append(0.0)
#                                 else: pct_var.append(((a - p) / a) * 100)
                                
#                             m_df = pd.DataFrame({"Operational Attribute Pillar": metrics})
#                             for idx in range(len(metrics)):
#                                 fmt = formats[idx]
#                                 if fmt == "currency":
#                                     m_df.at[idx, "Historical Actuals"] = f"${act_vals[idx]:,.2f}"
#                                     m_df.at[idx, "Optimized Policy"] = f"${pol_vals[idx]:,.2f}"
#                                     m_df.at[idx, "Net Delta Variance"] = f"${abs_var[idx]:,.2f}" if abs_var[idx] >= 0 else f"-${abs(abs_var[idx]):,.2f}"
#                                     m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
#                                 elif fmt == "pct":
#                                     m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:.1f}%"
#                                     m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:.1f}%"
#                                     m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+.1f}% pts"
#                                     m_df.at[idx, "% Impact Efficiency"] = f"{pol_vals[idx] - act_vals[idx]:+.1f}% pts"
#                                 elif fmt == "days":
#                                     m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:,.1f} days"
#                                     m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:,.1f} days"
#                                     m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+,.1f} days"
#                                     m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
#                                 else:
#                                     m_df.at[idx, "Historical Actuals"] = f"{int(act_vals[idx]):,}"
#                                     m_df.at[idx, "Optimized Policy"] = f"{int(pol_vals[idx]):,}"
#                                     m_df.at[idx, "Net Delta Variance"] = f"{int(abs_var[idx]):+1,}"
#                                     m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"

#                             def apply_matrix_styles(x):
#                                 colors = pd.DataFrame('', index=x.index, columns=x.columns)
#                                 fav = 'background-color: #1A3E2B; color: #81C784; font-weight: bold;'
#                                 unfav = 'background-color: #3E1A1A; color: #E57373;'
#                                 for i, metric in enumerate(metrics):
#                                     v = abs_var[i]
#                                     if title == "1. Financial Breakdown Matrix" or title == "3. Working Capital Release Matrix":
#                                         if v > 0: colors.iloc[i, 3:] = fav
#                                         elif v < 0: colors.iloc[i, 3:] = unfav
#                                     elif title == "4. Stockout Risk & Vulnerability Matrix":
#                                         if "Fill Rate" in metric:
#                                             if pol_vals[i] > act_vals[i]: colors.iloc[i, 3:] = fav
#                                             elif pol_vals[i] < act_vals[i]: colors.iloc[i, 3:] = unfav
#                                         else:
#                                             if v > 0: colors.iloc[i, 3:] = fav
#                                             elif v < 0: colors.iloc[i, 3:] = unfav
#                                     elif title == "5. Inventory Age & Freshness Matrix (FIFO)":
#                                         if v > 0: colors.iloc[i, 3:] = fav   
#                                         elif v < 0: colors.iloc[i, 3:] = unfav
#                                 return colors
#                             st.dataframe(m_df.style.apply(apply_matrix_styles, axis=None), use_container_width=True, hide_index=True)

#                         render_clustered_matrix("1. Financial Breakdown Matrix", ["Annual Ordering Fees ($)", "Annual Storage Carrying Cost ($)", "Financial Penalty from Stockouts ($)", "Total Policy Operating Cost ($)"], [actual_total_ordering_cost, actual_total_holding_cost, actual_lost_sales_financial, actual_total_cost], [optimal_ordering_cost, optimal_holding_cost, optimal_lost_sales_financial, optimal_total_cost], ["currency", "currency", "currency", "currency"])
#                         render_clustered_matrix("2. Logistical Operations Footprint Matrix", ["Average Volume Kept On-Hand", "Maximum Storage Spike Level", "Total Orders Dispatched", "Average Logistics Cycle Time", "Average Order Shipment Size"], [actual_avg_inventory, actual_max_inventory, actual_orders_placed, actual_cycle_time, actual_avg_order_size], [simmed_avg_opt_inv, simmed_max_opt_inv, opt_orders_placed, policy_cycle_time, policy_avg_order_size], ["units", "units", "count", "days", "units"])
#                         render_clustered_matrix("3. Working Capital Release Matrix", ["Peak Working Capital Tied Up ($)", "Average Working Capital Tied Up ($)", "Minimum Base Working Capital ($)"], [act_max_wc, act_avg_wc, act_min_wc], [opt_max_wc, opt_avg_wc, opt_min_wc], ["currency", "currency", "currency"])
#                         render_clustered_matrix("4. Stockout Risk & Vulnerability Matrix", ["Absolute Minimum Buffer Stock", "Stockout Events (Unfulfilled Days)", "Total Unfulfilled Deficit Volume", "Days with Absolute Zero Closing Stock", "Achieved Order Fill Rate (%)"], [actual_min_inventory, stockout_days_act, lost_sales_qty_act, zero_stock_days_act, actual_fill_rate * 100], [simmed_min_inventory, stockout_days_opt, lost_sales_qty_opt, zero_stock_days_opt, simmed_opt_fill_rate * 100], ["units", "count", "units", "count", "pct"])
#                         render_clustered_matrix("5. Inventory Age & Freshness Matrix (FIFO)", ["Overall Average Inventory Age (Days)", "Maximum Peak Inventory Age (Days)"], [actual_overall_avg_age, actual_overall_max_age], [opt_overall_avg_age, opt_overall_max_age], ["days", "days"])

#                     with timeline_container:
#                         st.markdown("---")
#                         st.markdown("### 📈 Tactical Operations Timeline Visualizations")
#                         timeline_fig = go.Figure()
#                         timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_act, name="Historical Actuals (Ledger)", line=dict(color='#B0C4DE', width=2), fill='tozeroy', fillcolor='rgba(176, 196, 222, 0.15)'))
#                         timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_opt, name=f"Recommended Optimized Policy ({best_fit_name.split(' ')[0]})", line=dict(color='#1F77B4', width=2.5)))
#                         timeline_fig.add_trace(go.Scatter(x=df["Date"], y=[max(0, raw_target_level - risk_mean)] * len(df), name="Calculated Safety Stock Floor", line=dict(color='#FF4B4B', width=1.5, dash='dot')))
#                         timeline_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
#                         st.plotly_chart(timeline_fig, use_container_width=True)

#                     with age_profile_container:
#                         age_tab1, age_tab2 = st.tabs(["📉 Historical Actuals Age Profile", "⚙️ Optimized Policy Age Profile"])
                        
#                         with age_tab1:
#                             fig_act_age = go.Figure()
#                             for j in range(len(bucket_labels)):
#                                 fig_act_age.add_trace(go.Scatter(
#                                     x=df["Date"], y=buckets_act[:, j], name=bucket_labels[j],
#                                     mode='lines', stackgroup='one', line=dict(width=0.5, color=bucket_colors[j])
#                                 ))
#                             fig_act_age.update_layout(template="plotly_white", yaxis_title="Units In Stock", xaxis_title="Date", height=350)
#                             st.plotly_chart(fig_act_age, use_container_width=True)
                            
#                         with age_tab2:
#                             fig_opt_age = go.Figure()
#                             for j in range(len(bucket_labels)):
#                                 fig_opt_age.add_trace(go.Scatter(
#                                     x=df["Date"], y=buckets_opt[:, j], name=bucket_labels[j],
#                                     mode='lines', stackgroup='one', line=dict(width=0.5, color=bucket_colors[j])
#                                 ))
#                             fig_opt_age.update_layout(template="plotly_white", yaxis_title="Units In Stock", xaxis_title="Date", height=350)
#                             st.plotly_chart(fig_opt_age, use_container_width=True)

#                     with drilldown_container:
#                         st.markdown("---")
#                         st.markdown("### 🔍 Point-in-Time Inventory Age Drilldown")
#                         drilldown_date = st.date_input("Select specific date to inspect inventory age distribution", value=end_date, min_value=start_date, max_value=end_date)
                        
#                         matched_row = df[df["Date"].dt.date == drilldown_date]
#                         if not matched_row.empty:
#                             idx_drill = matched_row.index[0]
#                             act_dist = buckets_act[idx_drill]
#                             opt_dist = buckets_opt[idx_drill]
                            
#                             drill_df = pd.DataFrame({
#                                 "Age Bracket": bucket_labels,
#                                 "Historical Actuals (Units)": act_dist.astype(int),
#                                 "Optimized Policy (Units)": opt_dist.astype(int)
#                             })
                            
#                             col_chart, col_table = st.columns([2, 1])
#                             with col_chart:
#                                 drill_fig = go.Figure()
#                                 drill_fig.add_trace(go.Bar(x=bucket_labels, y=act_dist, name="Actuals", marker_color="#B0C4DE"))
#                                 drill_fig.add_trace(go.Bar(x=bucket_labels, y=opt_dist, name="Optimized", marker_color="#1F77B4"))
#                                 drill_fig.update_layout(barmode='group', title=f"Age Distribution on {drilldown_date}", yaxis_title="Units", template="plotly_white", margin=dict(t=40, b=20))
#                                 st.plotly_chart(drill_fig, use_container_width=True)
                                
#                             with col_table:
#                                 st.markdown(f"**Exact Stock Counts:**")
#                                 st.dataframe(drill_df, hide_index=True, use_container_width=True)

#                     # ==========================================
#                     #     SECTION B: COMPARATIVE ANALYSIS
#                     # ==========================================
#                     st.markdown("---")
#                     st.header("🔬 Section B: Multi-Scenario Comparative Analysis")
#                     st.markdown(
#                         "Leveraging the high-speed vectorized simulation engine, you can now backtest and compare up to 6 "
#                         "different inventory policies simultaneously. By adjusting these mechanical levers, you can easily identify "
#                         "operational blind spots without manually tracking the math."
#                     )
                    
#                     active_scenarios_list = []
                    
#                     tab_cont, tab_per = st.tabs(["📉 Continuous Review (Q, R) Scenarios", "⏳ Periodic Review (P, T) Scenarios"])
                    
#                     with tab_cont:
#                         st.markdown(f"**Baseline Intelligence:** The data-driven optimal benchmark is **Q: {int(raw_optimal_q):,}** and **ROP: {int(raw_target_level):,}**.")
                        
#                         c1, c2, c3 = st.columns(3)
                        
#                         def create_cont_box(col, num, default_q, default_r):
#                             with col:
#                                 st.markdown(f"##### 🎛️ Scenario C{num}")
#                                 run_c = st.toggle(f"Include C{num} in Chart", key=f"run_c{num}", value=(num==1))
#                                 q_val = st.number_input("Order Qty (Q)", min_value=1, value=int(default_q), step=10, key=f"q_c{num}")
#                                 r_val = st.number_input("Reorder Point (ROP)", min_value=0, value=int(default_r), step=10, key=f"r_c{num}")
                                
#                                 if run_c:
#                                     active_scenarios_list.append({
#                                         "Case Name": f"C{num} (Q:{q_val}, R:{r_val})", 
#                                         "Policy Type": "Continuous Review (Q, R)", 
#                                         "P1": q_val, 
#                                         "P2": r_val
#                                     })

#                         create_cont_box(c1, 1, raw_optimal_q, raw_target_level)
#                         create_cont_box(c2, 2, raw_optimal_q * 1.5, raw_target_level)
#                         create_cont_box(c3, 3, raw_optimal_q, raw_target_level * 1.2)

#                     with tab_per:
#                         st.markdown("**Baseline Intelligence:** Adjust the Review Period (P) below. The engine will dynamically calculate a safe expected Target Level (T) for that exact timeframe.")
                        
#                         p1, p2, p3 = st.columns(3)
                        
#                         def create_per_box(col, num, default_p):
#                             with col:
#                                 st.markdown(f"##### 🎛️ Scenario P{num}")
#                                 run_p = st.toggle(f"Include P{num} in Chart", key=f"run_p{num}", value=False)
#                                 p_val = st.number_input("Review Period (P Days)", min_value=1, value=int(default_p), step=1, key=f"p_p{num}")
                                
#                                 target_guide = int(raw_target_level + (avg_daily_demand_calc * p_val))
                                
#                                 t_val = st.number_input("Target Level (T)", min_value=0, value=target_guide, step=10, key=f"t_p{num}")
#                                 st.caption(f"💡 *Engine recommended (T) for {p_val} days: **~{target_guide:,}***")
                                
#                                 if run_p:
#                                     active_scenarios_list.append({
#                                         "Case Name": f"P{num} (P:{p_val}, T:{t_val})", 
#                                         "Policy Type": "Periodic Review (P, T)", 
#                                         "P1": p_val, 
#                                         "P2": t_val
#                                     })

#                         create_per_box(p1, 1, 7)
#                         create_per_box(p2, 2, 14)
#                         create_per_box(p3, 3, 30)

#                     st.markdown("---")
                    
#                     if st.button("🚀 Compare Scenarios", type="primary", use_container_width=True):
#                         if not active_scenarios_list:
#                             st.warning("Please toggle 'Include' for at least one scenario above to generate the comparison.")
#                         else:
#                             summary_data = []
#                             comp_fig = go.Figure()
                            
#                             comp_fig.add_trace(go.Scatter(
#                                 x=df["Date"], y=inv_levels_act, mode='lines', 
#                                 name="Historical Actuals", line=dict(color='rgba(176, 196, 222, 0.4)', width=2, dash='dot')
#                             ))

#                             summary_data.append({
#                                 "Scenario Blueprint": "📊 Historical Actuals (Baseline)",
#                                 "Total Op Cost ($)": actual_total_cost,
#                                 "Fill Rate (%)": actual_fill_rate * 100,
#                                 "Avg Inv (Units)": actual_avg_inventory,
#                                 "Avg Working Capital ($)": act_avg_wc,
#                                 "Max Peak Capital ($)": act_max_wc,
#                                 "Avg Age (Days)": np.mean(avg_age_act),
#                                 "Peak Age (Days)": np.max(max_age_act)
#                             })

#                             line_colors = px.colors.qualitative.D3
                            
#                             for index, scenario in enumerate(active_scenarios_list):
#                                 case_name = scenario["Case Name"]
#                                 p_type = scenario["Policy Type"]
#                                 val1 = scenario["P1"]
#                                 val2 = scenario["P2"]
                                
#                                 s_inv, s_lost, s_orders, s_avg_age, s_max_age, s_buckets = fast_simulate_inventory(
#                                     demand_arr_main, purchase_arr_main, opening_stock_override, 
#                                     lead_time_days, p_type, val1, val2, custom_edges
#                                 )
                                
#                                 s_lost_sum = s_lost.sum()
#                                 s_orders_count = np.count_nonzero(s_orders)
#                                 s_avg_inv = np.mean(s_inv)
#                                 s_max_inv = np.max(s_inv)
                                
#                                 s_fill_rate = max(0.0, 1.0 - (s_lost_sum / max(1, total_demand)))
#                                 s_total_cost = (s_orders_count * ordering_cost) + (s_avg_inv * unit_holding_cost) + (s_lost_sum * lost_sales_penalty)
                                
#                                 summary_data.append({
#                                     "Scenario Blueprint": case_name,
#                                     "Total Op Cost ($)": s_total_cost,
#                                     "Fill Rate (%)": s_fill_rate * 100,
#                                     "Avg Inv (Units)": s_avg_inv,
#                                     "Avg Working Capital ($)": s_avg_inv * item_unit_cost,
#                                     "Max Peak Capital ($)": s_max_inv * item_unit_cost,
#                                     "Avg Age (Days)": np.mean(s_avg_age),
#                                     "Peak Age (Days)": np.max(s_max_age)
#                                 })
                                
#                                 comp_fig.add_trace(go.Scatter(
#                                     x=df["Date"], y=s_inv, mode='lines', 
#                                     name=case_name, line=dict(color=line_colors[index % len(line_colors)], width=2.5)
#                                 ))

#                             st.markdown("##### 🏆 Comparative Outcomes Scorecard")
#                             comp_df = pd.DataFrame(summary_data)
                            
#                             def highlight_baseline(s):
#                                 return ['background-color: rgba(176, 196, 222, 0.15)' if s['Scenario Blueprint'] == "📊 Historical Actuals (Baseline)" else '' for v in s]

#                             st.dataframe(
#                                 comp_df.style.apply(highlight_baseline, axis=1),
#                                 use_container_width=True, hide_index=True,
#                                 column_config={
#                                     "Total Op Cost ($)": st.column_config.NumberColumn(format="$%.2f"),
#                                     "Fill Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
#                                     "Avg Inv (Units)": st.column_config.NumberColumn(format="%d"),
#                                     "Avg Working Capital ($)": st.column_config.NumberColumn(format="$%.0f"),
#                                     "Max Peak Capital ($)": st.column_config.NumberColumn(format="$%.0f"),
#                                     "Avg Age (Days)": st.column_config.NumberColumn(format="%.1f"),
#                                     "Peak Age (Days)": st.column_config.NumberColumn(format="%.0f")
#                                 }
#                             )
                            
#                             st.markdown("##### 📈 Strategic Trajectory Matrix")
#                             comp_fig.update_layout(
#                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
#                                 xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", 
#                                 height=450, legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
#                             )
#                             st.plotly_chart(comp_fig, use_container_width=True)
# #Multi SKU Simulation
# with tab6:
#     st.header("Multi-SKU Dynamic Inventory Simulator")

#     # Helper function integrated inside the tab scope
#     def calculate_z_score(service_level_pct):
#         p = service_level_pct / 100.0
#         if p >= 0.9999: return 3.719 # Cap for extreme values
#         if p <= 0.5: return 0.0
        
#         # Rational approximation for standard normal inverse
#         t = np.sqrt(-2.0 * np.log(1.0 - p))
#         num = 2.515517 + 0.802853 * t + 0.010328 * t**2
#         den = 1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
#         z = t - (num / den)
#         return z

#     # Global Parameters
#     colA, colB, colC = st.columns(3)
#     with colA:
#         horizon = st.slider("Simulation Horizon (Days)", min_value=30, max_value=365, value=90, key="tab6_horizon_slider")
#     with colB:
#         service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1, key="tab6_service_slider", help="Drives Safety Stock calculations")
#     with colC:
#         warmup_days = st.slider("Warm-up Period (Days)", min_value=0, max_value=horizon-5, value=min(30, horizon//3), help="Metrics are calculated only after this day.", key="tab6_warmup_slider")
    
#     Z_score = calculate_z_score(service_level)
    
#     st.markdown("### 1. Define SKU Policies")
#     sku_data_list = []

#     # 1A. Continuous Review Section
#     with st.expander("🔄 Continuous Review (s, Q) - SKUs 01 to 05", expanded=True):
#         hc = st.columns([1, 0.8, 0.8, 0.8, 0.6, 0.8, 0.7, 0.6, 0.8, 0.8, 1])
#         hc[0].markdown("**SKU**")
#         hc[1].markdown("**Cost**")
#         hc[2].markdown("**Dmd**")
#         hc[3].markdown("**Std**")
#         hc[4].markdown("**LT**")
#         hc[5].markdown("**Ord(₹)**")
#         hc[6].markdown("**Hld%**")
#         hc[7].markdown("**Off**")
#         hc[8].markdown("**s** (Edit)")
#         hc[9].markdown("**Q** (Edit)")
#         hc[10].markdown("**Open Bal**")

#         cont_defaults = {1: (True, 500.0, 10.0, 2.5, 3, 1000.0, 20.0, 0), 2: (True, 150.0, 25.0, 6.0, 2, 500.0, 15.0, 10)}
        
#         for i in range(1, 6):
#             if i in cont_defaults:
#                 d_inc, d_c, d_d, d_std, d_lt, d_ord, d_hld, d_off = cont_defaults[i]
#             else:
#                 d_inc, d_c, d_d, d_std, d_lt, d_ord, d_hld, d_off = False, 100.0, 10.0, 2.0, 5, 500.0, 20.0, 0
            
#             cols = st.columns([1, 0.8, 0.8, 0.8, 0.6, 0.8, 0.7, 0.6, 0.8, 0.8, 1])
#             with cols[0]: include = st.checkbox(f"SKU-{i:02d}", value=d_inc, key=f"c_inc_{i}")
#             with cols[1]: cost = st.number_input("Cost", min_value=0.01, value=float(d_c), key=f"c_c_{i}", label_visibility="collapsed")
#             with cols[2]: demand = st.number_input("Demand", min_value=0.0, value=float(d_d), key=f"c_d_{i}", label_visibility="collapsed")
#             with cols[3]: std_dev = st.number_input("Std Dev", min_value=0.0, value=float(d_std), key=f"c_std_{i}", label_visibility="collapsed")
#             with cols[4]: lt = st.number_input("LT", min_value=0, value=int(d_lt), key=f"c_lt_{i}", label_visibility="collapsed")
#             with cols[5]: ord_cost = st.number_input("Order", min_value=0.0, value=float(d_ord), key=f"c_ord_{i}", label_visibility="collapsed")
#             with cols[6]: hld_pct = st.number_input("Hold", min_value=0.01, value=float(d_hld), key=f"c_hld_{i}", label_visibility="collapsed")
#             with cols[7]: offset = st.number_input("Offset", min_value=0, value=int(d_off), key=f"c_off_{i}", label_visibility="collapsed")
            
#             s_calc = (demand * lt) + (Z_score * std_dev * np.sqrt(lt))
#             hld_cost_annual = cost * (hld_pct / 100.0)
#             Q_calc = np.sqrt((2 * demand * 365 * ord_cost) / hld_cost_annual) if hld_cost_annual > 0 else 0
            
#             dyn_c_key = f"c{i}_d{demand}_std{std_dev}_lt{lt}_ord{ord_cost}_hld{hld_pct}_z{Z_score}"

#             with cols[8]: s_val = st.number_input("s", value=float(s_calc), min_value=0.0, key=f"s_{dyn_c_key}", label_visibility="collapsed")
#             with cols[9]: Q_val = st.number_input("Q", value=float(Q_calc), min_value=0.0, key=f"q_{dyn_c_key}", label_visibility="collapsed")
            
#             open_bal_default = s_val * 1.25
#             with cols[10]: open_bal = st.number_input("Open Bal", value=float(open_bal_default), min_value=0.0, key=f"op_{dyn_c_key}_{s_val}", label_visibility="collapsed")

#             if include:
#                 sku_data_list.append({
#                     "SKU": f"SKU-{i:02d}", "Unit Cost (₹)": cost, "Daily Demand": demand, "Std Dev": std_dev,
#                     "Lead Time (Days)": lt, "Initial Inventory": open_bal, "Policy Type": "Continuous (s, Q)",
#                     "Param 1 (s or R)": s_val, "Param 2 (Q or S)": Q_val, "Start Offset (Days)": offset,
#                     "Order Cost (₹)": ord_cost, "Holding Cost (%)": hld_pct
#                 })

#     # 1B. Periodic Review Section
#     with st.expander("📅 Periodic Review (R, S) - SKUs 06 to 10", expanded=True):
#         hp = st.columns([1, 0.8, 0.8, 0.8, 0.6, 0.8, 0.7, 0.7, 0.6, 0.9, 1])
#         hp[0].markdown("**SKU**")
#         hp[1].markdown("**Cost**")
#         hp[2].markdown("**Dmd**")
#         hp[3].markdown("**Std**")
#         hp[4].markdown("**LT**")
#         hp[5].markdown("**Ord(₹)**")
#         hp[6].markdown("**Hld%**")
#         hp[7].markdown("**R(Days)**")
#         hp[8].markdown("**Off**")
#         hp[9].markdown("**S** (Edit)")
#         hp[10].markdown("**Open Bal**")

#         per_defaults = {6: (True, 1200.0, 5.0, 1.5, 5, 2000.0, 25.0, 7, 5)}
        
#         for i in range(6, 11):
#             if i in per_defaults:
#                 d_inc, d_c, d_d, d_std, d_lt, d_ord, d_hld, d_r, d_off = per_defaults[i]
#             else:
#                 d_inc, d_c, d_d, d_std, d_lt, d_ord, d_hld, d_r, d_off = False, 100.0, 10.0, 2.0, 5, 500.0, 20.0, 7, 0
            
#             cols = st.columns([1, 0.8, 0.8, 0.8, 0.6, 0.8, 0.7, 0.7, 0.6, 0.9, 1])
#             with cols[0]: include = st.checkbox(f"SKU-{i:02d}", value=d_inc, key=f"p_inc_{i}")
#             with cols[1]: cost = st.number_input("Cost", min_value=0.01, value=float(d_c), key=f"p_c_{i}", label_visibility="collapsed")
#             with cols[2]: demand = st.number_input("Demand", min_value=0.0, value=float(d_d), key=f"p_d_{i}", label_visibility="collapsed")
#             with cols[3]: std_dev = st.number_input("Std Dev", min_value=0.0, value=float(d_std), key=f"p_std_{i}", label_visibility="collapsed")
#             with cols[4]: lt = st.number_input("LT", min_value=0, value=int(d_lt), key=f"p_lt_{i}", label_visibility="collapsed")
#             with cols[5]: ord_cost = st.number_input("Order", min_value=0.0, value=float(d_ord), key=f"p_ord_{i}", label_visibility="collapsed")
#             with cols[6]: hld_pct = st.number_input("Hold", min_value=0.01, value=float(d_hld), key=f"p_hld_{i}", label_visibility="collapsed")
#             with cols[7]: R_days = st.number_input("R", min_value=1, value=int(d_r), key=f"p_r_{i}", label_visibility="collapsed")
#             with cols[8]: offset = st.number_input("Offset", min_value=0, value=int(d_off), key=f"p_off_{i}", label_visibility="collapsed")
            
#             exposure_period = lt + R_days
#             S_calc = (demand * exposure_period) + (Z_score * std_dev * np.sqrt(exposure_period))
            
#             dyn_p_key = f"p{i}_d{demand}_std{std_dev}_lt{lt}_r{R_days}_z{Z_score}"

#             with cols[9]: S_val = st.number_input("S", value=float(S_calc), min_value=0.0, key=f"s_{dyn_p_key}", label_visibility="collapsed")
            
#             open_bal_default = S_val 
#             with cols[10]: open_bal = st.number_input("Open Bal", value=float(open_bal_default), min_value=0.0, key=f"op_{dyn_p_key}_{S_val}", label_visibility="collapsed")

#             if include:
#                 sku_data_list.append({
#                     "SKU": f"SKU-{i:02d}", "Unit Cost (₹)": cost, "Daily Demand": demand, "Std Dev": std_dev,
#                     "Lead Time (Days)": lt, "Initial Inventory": open_bal, "Policy Type": "Periodic (R, S)",
#                     "Param 1 (s or R)": R_days, "Param 2 (Q or S)": S_val, "Start Offset (Days)": offset,
#                     "Order Cost (₹)": ord_cost, "Holding Cost (%)": hld_pct
#                 })

#     active_df = pd.DataFrame(sku_data_list)

#     @st.cache_data 
#     def run_vectorized_simulation(df, days, warmup):
#         N = len(df)
#         if N == 0:
#             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

#         initial_inv = df["Initial Inventory"].values.astype(float)
#         mean_demand = df["Daily Demand"].values.astype(float)
#         std_dev = df["Std Dev"].values.astype(float)
#         lt = df["Lead Time (Days)"].values.astype(int) 
#         offset = df["Start Offset (Days)"].values.astype(int) 
#         costs = df["Unit Cost (₹)"].values.astype(float)
#         ord_costs_arr = df["Order Cost (₹)"].values.astype(float)
#         hld_pct_arr = df["Holding Cost (%)"].values.astype(float)
        
#         is_cont = (df["Policy Type"] == "Continuous (s, Q)").values
#         is_periodic = (df["Policy Type"] == "Periodic (R, S)").values
#         p1 = df["Param 1 (s or R)"].values.astype(float)
#         p2 = df["Param 2 (Q or S)"].values.astype(float)

#         on_hand = initial_inv.copy().astype(float)
#         max_lt = 0 if len(lt) == 0 else np.max(lt)
        
#         arrivals = np.zeros((N, days + max_lt + 1), dtype=float) 
#         history = np.zeros((N, days), dtype=float)
        
#         daily_demand_matrix = np.zeros((N, days), dtype=float)
#         lost_sales_matrix = np.zeros((N, days), dtype=float)
#         orders_placed_matrix = np.zeros((N, days), dtype=int)

#         for t in range(days):
#             on_hand += arrivals[:, t]
#             active = t >= offset
#             stochastic_demand = np.maximum(0, np.random.normal(loc=mean_demand, scale=std_dev)) * active
            
#             daily_demand_matrix[:, t] = stochastic_demand
#             on_hand -= stochastic_demand
            
#             lost_sales_matrix[:, t] = np.maximum(0, -on_hand)
#             on_hand = np.maximum(0, on_hand)

#             pipeline = np.sum(arrivals[:, t+1:], axis=1)
#             inv_position = on_hand + pipeline

#             trigger_cont = active & is_cont & (inv_position <= p1)
#             R_safe = np.where(p1 > 0, p1, 1).astype(int)
#             is_review_day = (t - offset) % R_safe == 0
#             trigger_per = active & is_periodic & is_review_day & (inv_position < p2)

#             order_qty = np.zeros(N, dtype=float)
#             order_qty[trigger_cont] = p2[trigger_cont]
#             order_qty[trigger_per] = p2[trigger_per] - inv_position[trigger_per]

#             valid_orders = order_qty > 0
#             orders_placed_matrix[:, t] = valid_orders.astype(int)

#             if np.any(valid_orders):
#                 skus_to_order = np.where(valid_orders)[0]
#                 for i in skus_to_order:
#                     arrival_day = t + lt[i]
#                     if arrival_day < arrivals.shape[1]:
#                         arrivals[i, arrival_day] += order_qty[i]

#             history[:, t] = on_hand

#         history_df = pd.DataFrame(history.T, columns=df["SKU"])
        
#         eval_days = days - warmup
#         if eval_days <= 0:
#             eval_days = 1 

#         steady_history = history[:, warmup:]
#         steady_demand = daily_demand_matrix[:, warmup:]
#         steady_lost_sales = lost_sales_matrix[:, warmup:]
#         steady_orders = np.sum(orders_placed_matrix[:, warmup:], axis=1)
        
#         daily_capital = np.sum(history * costs[:, np.newaxis], axis=0)
#         capital_df = pd.DataFrame(daily_capital, columns=["Total Working Capital (₹)"])

#         # Post-warmup metrics evaluation
#         avg_inv = steady_history.mean(axis=1)
#         peak_inv_units = steady_history.max(axis=1)
#         peak_inv_capital = (steady_history * costs[:, np.newaxis]).max(axis=1)
        
#         total_order_costs = steady_orders * ord_costs_arr
#         total_holding_costs = avg_inv * costs * (hld_pct_arr / 100.0) * (eval_days / 365.0)
#         total_opex = total_order_costs + total_holding_costs

#         total_steady_demand = np.sum(steady_demand, axis=1)
#         total_steady_lost = np.sum(steady_lost_sales, axis=1)
        
#         fill_rate = np.where(total_steady_demand > 0, 
#                              (total_steady_demand - total_steady_lost) / total_steady_demand, 
#                              1.0)

#         metrics_df = pd.DataFrame({
#             "SKU": df["SKU"],
#             "Avg Physical Inventory": avg_inv,
#             "Peak Physical Inventory": peak_inv_units,
#             "Avg Capital Tied Up (₹)": avg_inv * costs,
#             "Peak Capital Tied Up (₹)": peak_inv_capital,
#             "Fill Rate (%)": fill_rate * 100,
#             "Total Orders": steady_orders,
#             "Ordering Cost (₹)": total_order_costs,
#             "Holding Cost (₹)": total_holding_costs,
#             "Total OPEX (₹)": total_opex
#         })
        
#         cost_breakdown = pd.DataFrame({
#             "SKU": df["SKU"].tolist() * 2,
#             "Cost Type": ["Ordering Cost"] * N + ["Holding Cost"] * N,
#             "Cost (₹)": total_order_costs.tolist() + total_holding_costs.tolist()
#         })

#         return history_df, metrics_df, capital_df, cost_breakdown, eval_days

#     # 3. Output Displays
#     if not active_df.empty:
#         history_df, metrics_df, capital_df, cost_breakdown, eval_days = run_vectorized_simulation(active_df, horizon, warmup_days)

#         st.markdown("### 2. Portfolio KPIs (Post Warm-up)")
        
#         # Split into two rows for cleaner layout on varied screen sizes
#         kpi1, kpi2, kpi3 = st.columns(3)
#         kpi4, kpi5, kpi6 = st.columns(3)
        
#         total_avg_inv = metrics_df["Avg Physical Inventory"].sum()
#         steady_inv_sum = history_df.iloc[warmup_days:].sum(axis=1)
#         peak_portfolio_inv = steady_inv_sum.max() if not steady_inv_sum.empty else 0

#         total_capital = metrics_df["Avg Capital Tied Up (₹)"].sum()
#         steady_cap_sum = capital_df.iloc[warmup_days:]["Total Working Capital (₹)"]
#         peak_portfolio_cap = steady_cap_sum.max() if not steady_cap_sum.empty else 0

#         total_opex_all = metrics_df["Total OPEX (₹)"].sum()
#         avg_fill_rate = metrics_df["Fill Rate (%)"].mean()
        
#         kpi1.metric("Total Avg Inventory", f"{total_avg_inv:,.0f} Units")
#         kpi2.metric("Portfolio Peak (Physical)", f"{peak_portfolio_inv:,.0f} Units")
#         kpi3.metric("Avg Fill Rate", f"{avg_fill_rate:.1f}%")
        
#         kpi4.metric("Total Avg Capital", f"₹{total_capital:,.2f}")
#         kpi5.metric("Portfolio Peak (Value)", f"₹{peak_portfolio_cap:,.2f}")
#         kpi6.metric(f"OPEX ({eval_days} Days)", f"₹{total_opex_all:,.2f}")

#         st.markdown("### 3. Projected On-Hand Inventory")
#         inv_melt = history_df.reset_index().rename(columns={"index": "Day"}).melt("Day", var_name="SKU", value_name="Inventory")
        
#         inv_chart = alt.Chart(inv_melt).mark_line().encode(
#             x=alt.X("Day:Q", axis=alt.Axis(grid=False)),
#             y=alt.Y("Inventory:Q", title="On-Hand Inventory", axis=alt.Axis(gridColor="#f0f0f0")),
#             color=alt.Color("SKU:N", scale=alt.Scale(scheme="blues"))
#         ).properties(height=350)
        
#         warmup_rule = alt.Chart(pd.DataFrame({'Day': [warmup_days]})).mark_rule(
#             color='red', strokeDash=[5, 5]
#         ).encode(x='Day:Q')
        
#         final_inv_chart = (inv_chart + warmup_rule).configure_view(strokeWidth=0)
#         st.altair_chart(final_inv_chart, use_container_width=True, theme=None)
        
#         st.markdown("### 4. Cost Tradeoff Analysis (OPEX)")
#         cost_chart = alt.Chart(cost_breakdown).mark_bar().encode(
#             x=alt.X("SKU:N", axis=alt.Axis(labelAngle=0)),
#             y=alt.Y("Cost (₹):Q", title="Total OPEX (₹)", axis=alt.Axis(gridColor="#f0f0f0")),
#             color=alt.Color("Cost Type:N", scale=alt.Scale(range=["#005b96", "#b3cde0"]))
#         ).properties(height=350).configure_view(strokeWidth=0)
#         st.altair_chart(cost_chart, use_container_width=True, theme=None)

#         st.markdown("### 5. Total Working Capital Movement")
#         cap_chart = alt.Chart(capital_df.reset_index().rename(columns={"index": "Day"})).mark_line(color="#005b96", strokeWidth=2).encode(
#             x=alt.X("Day:Q", axis=alt.Axis(grid=False)),
#             y=alt.Y("Total Working Capital (₹):Q", title="Capital Tied Up (₹)", axis=alt.Axis(gridColor="#f0f0f0"))
#         ).properties(height=300).configure_view(strokeWidth=0)
#         st.altair_chart(cap_chart, use_container_width=True, theme=None)

#         st.markdown("### 6. Detailed Financial Impact")
#         st.dataframe(
#             metrics_df.style.format({
#                 "Avg Physical Inventory": "{:,.1f}", 
#                 "Peak Physical Inventory": "{:,.1f}", 
#                 "Avg Capital Tied Up (₹)": "₹{:,.2f}",
#                 "Peak Capital Tied Up (₹)": "₹{:,.2f}",
#                 "Fill Rate (%)": "{:.2f}%",
#                 "Ordering Cost (₹)": "₹{:,.2f}",
#                 "Holding Cost (₹)": "₹{:,.2f}",
#                 "Total OPEX (₹)": "₹{:,.2f}"
#             }), 
#             use_container_width=True, 
#             hide_index=True
#         )

#     else:
#         st.info("Please include at least one SKU to run the simulation.")
