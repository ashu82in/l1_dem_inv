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

# --- Session State Initialization ---
if 'next_clicked' not in st.session_state:
    st.session_state.next_clicked = False
if 'seed_counter' not in st.session_state:
    st.session_state.seed_counter = 42

st.set_page_config(page_title="Supply Chain Analytics Platform", layout="wide")

st.title("🚀 Supply Chain Analytics Platform")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Average Demand", "📊 Demand Histogram", "📈 Demand Forecasting", "Demand Simulator Game", "Age Analysis", "Inventory Audit"])

with tab1:
    st.header("The Basic Thumb Rule Used For Inventory Planning")
    # st.markdown("""
    # **The Concept:** Demonstrating how static, average-based demand strategies systematically introduce internal sabotage. 
    # While an average looks clean over a 300-day window, daily variability will trigger stockouts during finite replenishment cycles.
    # """)
    
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
            if st.button("🔄 Regenerate Demand"):
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
            
            # Expanded layout matrix (changed to 4 columns to fit the new metric)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Days with Valid Window", f"{total_valid_days} Days")
            with m2:
                # Calculate absolute peak gap to show if the strategy safely absorbed it
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

            # Row-by-row functional mapper for color injection logic
            def calculate_status(row):
                forward_demand = row[f"Demand Next {rolling_window} Days"]
                if pd.isna(forward_demand):
                    return ""
                
                net_value = int(row["Inventory Level Provided"] - forward_demand)
                if net_value >= 0:
                    return f'<span style="color: #2e7d32; font-weight: bold;">🟢 Surplus (+{net_value})</span>'
                else:
                    return f'<span style="color: #d32f2f; font-weight: bold;">🔴 Deficit ({net_value})</span>'

            # Build and finalize table display dataframe
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

        # Final Summary Executive Alerts
        if total_deficits > 0:
            st.error(f"❌ **Internal Sabotage Confirmed:** Volatility breached your static 'Average' allocation baseline strategy on **{total_deficits} separate window cycles** ({pct_deficits:.1f}% risk rate).")
        else:
            st.success(f"✅ **Strategic Parameter Verified.** Under these isolated settings, the current allocation buffer safely absorbed the simulated variance across all window blocks.")



with tab2:
    st.header("Demand Histogram Analyzer")
    
    # --- 1. Data Configuration ---
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
                    
                    if 'Demand' in df_upload.columns:
                        df = df_upload[['Demand']].dropna().copy()
                        df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
                        df = df.dropna()
                        st.success("✅ File successfully uploaded and parsed!")
                    else:
                        st.error("❌ Invalid Format: Your file must contain a column named exactly **'Demand'**.")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
                    
        with up_col2:
            st.markdown("#### 📋 Download Template")
            st.caption("Please match your data format to this template. The sheet must include a column header named **Demand**.")
            
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

    # --- Collapsible Raw Data Table ---
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

    # --- 2. Advanced Analysis (Thresholds & Percentiles) ---
    if df is not None:
        st.divider()
        st.subheader("2. Probability & Coverage Analysis")
        
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

        # --- 3. Visual Distribution & Tables Below ---
        st.divider()
        st.subheader("3. Visual Distribution")
        
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
        
        st.divider()
        
        table_col1, table_col2 = st.columns([1, 1])

        with table_col1:
            st.markdown("#### 📋 Statistical Summary")
            summary_stats = df['Demand'].describe().to_frame().T
            st.dataframe(summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']], use_container_width=True)

        with table_col2:
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

        # --- 4. Coefficient of Variation (CoV) Analysis ---
        st.divider()
        st.subheader("📊 Demand Volatility Analysis (CoV)")
        
        cov_col1, cov_col2 = st.columns([1, 2])
        
        with cov_col1:
            st.markdown("#### Formula")
            st.latex(r"CoV = \frac{\sigma}{\mu}")
            st.caption(r"Where $\sigma$ = Standard Deviation and $\mu$ = Mean")
            
        with cov_col2:
            # Extract statistics directly from data stream
            mean_val = float(df['Demand'].mean())
            std_val = float(df['Demand'].std())
            
            # Defensive check for edge case where mean is zero
            cov_val = (std_val / mean_val) if mean_val > 0 else 0.0
            
            # Determine demand volatility profile category
            if cov_val <= 0.10:
                status_text = "🟢 Ultra-Stable / Constant"
                explanation = "Highly repetitive and predictable demand. Use automated just-in-time (JIT) scheduling or lean kanbans. Minimize safety stock to release working capital."
                alert_type = "success"
            elif cov_val <= 0.25:
                status_text = "🟢 Stable / Predictable"
                explanation = "Normal variation patterns present. Standard statistical forecasting and fixed reorder points will yield high accuracy with minimal safety stock buffers."
                alert_type = "success"
            elif cov_val <= 0.50:
                status_text = "🟡 Moderate Volatility"
                explanation = "Demand exhibits noticeable fluctuations. Requires proactive demand sensing and traditional statistical safety stocks to counter stockout risks."
                alert_type = "warning"
            elif cov_val <= 1.00:
                status_text = "🟠 High Volatility"
                explanation = "Highly variable demand spikes. Avoid automated ordering systems without collaborative forecasting inputs. Expect to maintain higher, dynamic safety stock thresholds."
                alert_type = "warning"
            else:
                status_text = "🔴 Erratic / Lumpy / Sporadic"
                explanation = "Highly unpredictable or intermittent demand. Traditional safety stock formulas do not work well here. Consider move-to-order (MTO) execution or project-based buffers."
                alert_type = "error"
                
            # Render key calculation metrics inside columns
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Mean ($\mu$)", f"{mean_val:.2f}")
            with m_col2:
                st.metric("Std Dev ($\sigma$)", f"{std_val:.2f}")
            with m_col3:
                st.metric("Calculated CoV", f"{cov_val:.3f}")
                
            # Render descriptive behavioral classification banner
            st.markdown(f"### Profile: {status_text}")
            st.info(explanation)

# Placeholder layouts for future tabs
with tab3:
    st.header("🧬 Stage 3: The Probability Truth")
    st.markdown("Analyze historical patterns, simulate growth, and project future demand with AI uncertainty bands.")
    
    # --- 1. DATA SOURCE & SAMPLE DOWNLOAD ---
    col_header, col_download = st.columns([2, 1])
    with col_header:
        data_mode = st.radio("Data Mode:", ("Simulation", "Upload Data"), horizontal=True, key="mode_t3")
    
    with col_download:
        # Create a sample template with growth and seasonality
        sample_dates = pd.date_range(start="2024-01-01", periods=365, freq='D')
        t_sample = np.arange(365)
        sample_y = 500 + (15 * t_sample/30) + (50 * np.sin(2 * np.pi * t_sample / 30)) + np.random.normal(0, 30, 365)
        sample_df = pd.DataFrame({'Date': sample_dates, 'Demand': np.maximum(0, sample_y).astype(int)})
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            sample_df.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 Download Workshop Template.xlsx",
            data=buffer.getvalue(),
            file_name="demand_template.xlsx",
            mime="application/vnd.ms-excel"
        )

    df_truth = None

    if data_mode == "Simulation":
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                level = st.number_input("Base Level (Start)", value=100.0)
                growth = st.number_input("Annual Growth %", value=15.0) # Increased for visible 'cone'
            with c2:
                base_amp = st.number_input("Amplitude", value=400.0)
                target_cov = st.number_input("Target CoV", value=0.15)
            with c3:
                surcharge = st.slider("Peak Surcharge %", 0, 100, 30)
            with c4:
                forecast_days = st.number_input("Forecast Horizon (Days)", value=365)

        # SIMULATION GENERATION
        dates = pd.date_range(start="2023-01-01", periods=730, freq='D')
        t = np.arange(len(dates))
        growth_factor = (1 + growth/100) ** (t / 365)
        seasonal_wave = np.sin(2 * np.pi * t / 365.25)
        
        baseline_calc = level + base_amp
        y_vals = (baseline_calc + (seasonal_wave * (base_amp * 0.5))) * growth_factor
        y_vals += np.random.normal(0, (base_amp * target_cov), len(dates))
        
        df_truth = pd.DataFrame({'ds': dates, 'y': np.maximum(0, y_vals)})
        
        # Labeling and Surcharge
        high_t = baseline_calc * 1.25
        low_t = baseline_calc * 0.75
        df_truth.loc[df_truth['y'] > high_t, 'y'] *= (1 + surcharge/100)
        df_truth['Seasonality'] = df_truth['y'].apply(lambda x: 'High' if x > high_t else ('Low' if x < low_t else 'Normal'))

    else:
        uploaded_file = st.file_uploader("Upload xlsx", type=["xlsx"], key="up_t3")
        forecast_days = st.number_input("Forecast Horizon (Days)", value=180)
        if uploaded_file:
            df_truth = pd.read_excel(uploaded_file).rename(columns={'Date':'ds', 'Demand':'y'})
            df_truth['ds'] = pd.to_datetime(df_truth['ds'])
            q1, q3 = df_truth['y'].quantile([0.25, 0.75])
            df_truth['Seasonality'] = df_truth['y'].apply(lambda x: 'High' if x > q3 else ('Low' if x < q1 else 'Normal'))

    # --- 2. THE SEASONAL METRICS MATRIX ---
    if df_truth is not None:
        st.divider()
        st.subheader("📊 The Seasonal Metrics Matrix")
        
        # Calculate stats
        matrix = df_truth.groupby('Seasonality')['y'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        matrix['CoV'] = (matrix['std'] / matrix['mean']).round(3)
        matrix.columns = ['Season', 'Avg Demand', 'Std Dev', 'Min', 'Max', 'Days Count', 'CoV']
        
        # Display styled table
        st.dataframe(
            matrix.style.background_gradient(subset=['CoV'], cmap='RdYlGn_r').format(precision=2),
            use_container_width=True, hide_index=True
        )

        # --- 3. DUAL HISTOGRAMS ---
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(px.histogram(df_truth, x="y", title="A. General Distribution", template="plotly_dark", nbins=40), use_container_width=True)
        with col_r:
            fig_s = px.histogram(df_truth, x="y", color="Seasonality", title="B. Seasonal Breakdown", template="plotly_dark", barmode='overlay',
                                 color_discrete_map={"Normal": "#5B84B1", "High": "#FC766A", "Low": "#71918d"})
            st.plotly_chart(fig_s, use_container_width=True)

        # --- 4. PROPHET FORECAST (With Broadening Uncertainty) ---
        st.divider()
        st.subheader("🔮 AI Prophet Forecast (Trend & Uncertainty)")
        
        with st.spinner("AI training and calculating risk bands..."):
            # Changepoint_prior_scale=0.5 makes the uncertainty cone widen significantly
            m = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.5)
            m.fit(df_truth)
            
            future = m.make_future_dataframe(periods=int(forecast_days))
            forecast = m.predict(future)
            
            fig_f = go.Figure()

            # The broadening Uncertainty Ribbon
            fig_f.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(100, 100, 100, 0.3)', line=dict(color='rgba(255,255,255,0)'),
                name='Uncertainty Interval (95%)'
            ))

            # AI Prediction Line
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], line=dict(color='#4F8BF9', width=3), name='AI Forecast'))

            # Actual Historical Points
            fig_f.add_trace(go.Scatter(x=df_truth['ds'], y=df_truth['y'], mode='markers', marker=dict(color='white', size=2), name='Actual Data'))

            fig_f.update_layout(
                template="plotly_dark",
                title=f"Prophet Projection for {forecast_days} Days",
                yaxis=dict(range=[0, forecast['yhat_upper'].max() * 1.1]), # Force Y-axis to start at 0
                xaxis_title="Date", yaxis_title="Demand Quantity",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_f, use_container_width=True)
            
        with st.expander("📂 View Raw Data Table"):
            st.dataframe(df_truth, use_container_width=True)

with tab4:
    st.header("🎯 Tab 4: Safety Stock Simulation Game")
    st.markdown("""
    **The Challenge:** Balance inventory holding costs against shortages. 
    Toggle between a **Continuous Review (ROP)** or **Periodic Review (Order-Up-To)** system.
    """)

    # =========================================================================
    # GLOBAL VISUAL STYLE LAYOUT CONFIGURATION
    # =========================================================================
    shared_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="sans-serif"), 
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.07)", zeroline=False, linecolor="rgba(255, 255, 255, 0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.07)", zeroline=False, linecolor="rgba(255, 255, 255, 0.15)")
    )

    # =========================================================================
    # SECTION 1: DATA CONFIGURATION
    # =========================================================================
    st.markdown("### 1. General Configuration")
    
    col_policy, col_dist = st.columns(2)
    with col_policy:
        review_policy = st.selectbox("Inventory Review Policy", ["Continuous Review (ROP)", "Periodic Review (Interval)"], index=0, key="t4_review_policy")
    with col_dist:
        dist_type = st.selectbox("Distribution Type", ["Uniform", "Normal"], index=0, key="t4_dist_type")
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        avg_demand = st.number_input("Average Daily Demand", min_value=1, value=50, step=5, key="t4_avg_dem")
        
    with col_cfg2:
        if dist_type == "Uniform":
            variation = st.number_input("Daily Variation (± From Average)", min_value=0, value=25, step=5, key="t4_variation")
            low_bound = max(0, avg_demand - variation)
            high_bound = avg_demand + variation
            std_dev = 0
        else:
            std_dev = st.number_input("Daily Std Dev (σ)", min_value=0.0, value=10.0, step=1.0, key="t4_std_dev")
            low_bound, high_bound, variation = 0, 0, 0

    if dist_type == "Uniform":
        st.markdown(f"🔹 *Daily Range: **{low_bound}** to **{high_bound}** units*")

    st.markdown("### 2. Backlog & Stockout Policy")
    col_back1, col_back2 = st.columns(2)
    with col_back1:
        allow_backlog = st.radio("Can unfulfilled demand be backlogged for later?", ["Yes", "No"], index=0, key="t4_allow_backlog")
    with col_back2:
        if allow_backlog == "Yes":
            backlog_limit = st.number_input("Maximum Backlog Capacity Limit (0 for Unlimited)", min_value=0, value=100, step=10, key="t4_backlog_limit")
        else:
            st.markdown("<div style='padding-top: 30px; color: gray; font-size: 14px;'>❌ Lost Sales Policy Active: Inventory cannot drop below 0.</div>", unsafe_allow_html=True)
            backlog_limit = 0

    # =========================================================================
    # SECTION 2: DYNAMIC CALCULATOR ENGINE (Q vs P SYSTEM MATH)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📈 3. Mathematical Policy Suggestions")
    
    col_rop1, col_rop2 = st.columns([1.2, 1.8])
    
    with col_rop1:
        st.markdown("**Target Risk Parameters**")
        lead_time = st.number_input("Supplier Lead Time (Days)", min_value=1, value=3, step=1, key="t4_lt")
        
        if review_policy == "Periodic Review (Interval)":
            review_period = st.number_input("Review Interval / Period (Days)", min_value=1, value=5, step=1, key="t4_rp_days")
            risk_window = review_period + lead_time
            label_text = f"Review Period + Lead Time ({review_period} + {lead_time})"
        else:
            risk_window = lead_time
            label_text = f"Lead Time Only ({lead_time} Days)"
            
        target_service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="t4_tsl")
        
        # Calculate Scaled Baseline Demand over the vulnerability window
        scaled_avg_demand = avg_demand * risk_window
        
        if dist_type == "Normal":
            scaled_std_dev = std_dev * np.sqrt(risk_window)
            z_score = stats.norm.ppf(target_service_level / 100.0) if scaled_std_dev > 0 else 0
            safety_stock = int(np.ceil(z_score * scaled_std_dev))
            suggested_target = int(np.ceil(scaled_avg_demand + safety_stock))
        else:
            sim_rng = np.random.RandomState(42)
            if variation == 0:
                lt_samples = np.full(10000, scaled_avg_demand)
            else:
                lt_samples = np.sum(sim_rng.randint(low_bound, high_bound + 1, size=(risk_window, 10000)), axis=0)
            
            suggested_target = int(np.percentile(lt_samples, target_service_level))
            safety_stock = max(0, suggested_target - scaled_avg_demand)

        st.markdown("#### **Calculation Results**")
        st.metric(f"Expected Demand Over {risk_window} Days", f"{int(scaled_avg_demand)} units")
        st.metric("Required Safety Stock Buffer", f"{int(safety_stock)} units")
        
        if review_policy == "Periodic Review (Interval)":
            st.metric("Suggested Target Stock (Max T)", f"{suggested_target} units")
            prepopulated_val = suggested_target
        else:
            st.metric("Suggested Reorder Point (ROP)", f"{suggested_target} units")
            prepopulated_val = suggested_target

    with col_rop2:
        st.markdown(f"**Total Demand Distribution Across Risk Window ({label_text})**")
        
        if dist_type == "Normal":
            if scaled_std_dev > 0:
                x_axis_range = np.linspace(scaled_avg_demand - 4*scaled_std_dev, scaled_avg_demand + 4*scaled_std_dev, 200)
                y_axis_density = stats.norm.pdf(x_axis_range, scaled_avg_demand, scaled_std_dev)
            else:
                x_axis_range = np.array([scaled_avg_demand - 5, scaled_avg_demand, scaled_avg_demand + 5])
                y_axis_density = np.array([0, 1, 0])
            
            fig_rop_dist = go.Figure()
            fig_rop_dist.add_trace(go.Scatter(x=x_axis_range, y=y_axis_density, mode='lines', line=dict(color='#A370F7', width=3), name='Probability Density', fill='tozeroy', fillcolor='rgba(163, 112, 247, 0.1)'))
        else:
            if variation == 0:
                x_axis_range = np.array([scaled_avg_demand - 5, scaled_avg_demand, scaled_avg_demand + 5])
                y_axis_density = np.array([0, 1, 0])
                fig_rop_dist = go.Figure()
                fig_rop_dist.add_trace(go.Scatter(x=x_axis_range, y=y_axis_density, mode='lines'))
            else:
                daily_choices = np.arange(low_bound, high_bound + 1)
                pmf = np.ones(len(daily_choices)) / len(daily_choices)
                lt_pmf = pmf.copy()
                for _ in range(int(risk_window) - 1):
                    lt_pmf = np.convolve(lt_pmf, pmf)
                x_axis_range = np.arange(int(low_bound * risk_window), int(high_bound * risk_window) + 1)
                y_axis_density = lt_pmf
                
                fig_rop_dist = go.Figure()
                fig_rop_dist.add_trace(go.Scatter(x=bin_centers if 'bin_centers' in locals() else x_axis_range, y=y_axis_density, mode='lines', line=dict(color='#A370F7', width=3, shape='spline'), name='Compounded Shape', fill='tozeroy', fillcolor='rgba(163, 112, 247, 0.1)'))

        fig_rop_dist.add_vline(x=scaled_avg_demand, line_width=2, line_dash="dash", line_color="#3A96FF", annotation_text="Expected Demand", annotation_position="top left")
        fig_rop_dist.add_vline(x=suggested_target, line_width=2.5, line_color="#FF5A5A", annotation_text=f"Suggested ({target_service_level}%)", annotation_position="top right")
        
        fig_rop_dist.update_layout(**shared_layout, showlegend=False, height=280)
        fig_rop_dist.update_layout(margin=dict(l=30, r=30, t=20, b=30))
        fig_rop_dist.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        st.plotly_chart(fig_rop_dist, use_container_width=True)

    # =========================================================================
    # ADVANCED SETTINGS PANEL WITH EDITABLE OVERRIDES & DYNAMIC DEFAULTS
    # =========================================================================
    with st.expander("🛠️ Advanced Control Settings (User Overrides Available)", expanded=True):
        col_inv1, col_inv2, col_inv3 = st.columns(3)
        
        with col_inv1:
            if review_policy == "Periodic Review (Interval)":
                target_stock_level = st.number_input("User Target Stock Level (T)", min_value=1, value=int(prepopulated_val), step=10, key="t4_target_max_t_level")
                reorder_point = 0 
                # FIXED INITIALIZATION MECHANISM: Set default starting inventory safely to the Target Level
                default_start_stock = target_stock_level
            else:
                reorder_point = st.number_input("User Reorder Point (ROP)", min_value=0, value=int(prepopulated_val), step=10, key="t4_rop_override")
                target_stock_level = 0
                # Q-System defaults safely to traditional textbook buffer levels
                default_start_stock = int(prepopulated_val + (avg_demand * 1.25))
                
        with col_inv2:
            starting_inventory = st.number_input("Starting On-Hand Inventory", min_value=1, value=default_start_stock, step=10, key="t4_start_inv")
        with col_inv3:
            if review_policy == "Periodic Review (Interval)":
                st.markdown(f"<div style='padding-top: 25px; color: gray; font-size: 13px;'>Batch size (Q) is dynamic:<br><b>T - Current Inventory Position</b></div>", unsafe_allow_html=True)
                order_qty = 0
            else:
                order_qty = st.number_input("Fixed Replenishment Quantity (Q)", min_value=1, value=200, step=10, key="t4_q")

    # =========================================================================
    # SECTION 3: CORE SIMULATION ENGINE WITH PIPELINE PRE-POPULATION
    # =========================================================================
    if 't4_history' not in st.session_state:
        st.session_state.t4_history = pd.DataFrame(columns=[
            'Day', 'Opening Stock', 'Arrived Morning', 'Updated Opening Stock', 
            'Demand Generated', 'Sales Met', 'Shortage', 'Unfulfilled Backlog', 
            'Closing Inventory', 'Order Placed Evening', 'Total Pipeline Inventory', 'Pipeline Status'
        ])
        st.session_state.t4_day_counter = 0
        st.session_state.t4_current_inv = starting_inventory
        st.session_state.t4_backlog = 0  
        
        # FIXED PIPELINE MECHANISM: Pre-seed an order in the pipeline to arrive on Day L+1 morning 
        # to prevent unbalance and stockout spikes on the very first cycle
        if review_policy == "Periodic Review (Interval)":
            st.session_state.t4_pipeline_orders = [{'delivery_day': lead_time + 1, 'qty': int(avg_demand * review_period)}]
        else:
            st.session_state.t4_pipeline_orders = [{'delivery_day': lead_time + 1, 'qty': order_qty}]

    def run_simulation_steps(num_days):
        if 't4_backlog' not in st.session_state:
            st.session_state.t4_backlog = 0
            
        history_df = st.session_state.t4_history.copy()
        day_counter = st.session_state.t4_day_counter
        current_inv = st.session_state.t4_current_inv
        backlog = st.session_state.t4_backlog
        pipeline_orders = list(st.session_state.t4_pipeline_orders)
        
        rng = np.random.RandomState()
        new_records = []

        for _ in range(num_days):
            day_counter += 1
            initial_opening_stock = current_inv
            
            # 1. MORNING PHASE: Deliver incoming orders
            arriving_qty = sum(order['qty'] for order in pipeline_orders if order['delivery_day'] == day_counter)
            pipeline_orders = [order for order in pipeline_orders if order['delivery_day'] != day_counter]
            
            if arriving_qty > 0:
                if allow_backlog == "Yes" and backlog > 0:
                    if arriving_qty >= backlog:
                        arriving_qty -= backlog
                        backlog = 0
                    else:
                        backlog -= arriving_qty
                        arriving_qty = 0
                current_inv += arriving_qty
            
            updated_opening_stock = current_inv
            
            # 2. DAYTIME PHASE: Generate Demand
            if dist_type == "Normal":
                demand = max(0, int(rng.normal(float(avg_demand), float(std_dev))))
            else:
                if low_bound == high_bound:
                    demand = int(avg_demand)
                else:
                    demand = int(rng.randint(int(low_bound), int(high_bound) + 1))
            
            # Fulfill Demand
            total_needed = demand + backlog
            if updated_opening_stock >= total_needed:
                sales_met = demand
                shortage = 0
                backlog = 0
                closing_inv = updated_opening_stock - total_needed
            else:
                sales_met = max(0, updated_opening_stock - backlog)
                raw_shortage = total_needed - updated_opening_stock
                
                if allow_backlog == "Yes":
                    backlog = min(raw_shortage, backlog_limit) if backlog_limit > 0 else raw_shortage
                    shortage = raw_shortage
                else:
                    backlog = 0
                    shortage = demand - updated_opening_stock
                    
                closing_inv = 0
                
            # 3. EVENING PHASE: Evaluate Trigger Criteria
            pipeline_qty_before_order = sum(order['qty'] for order in pipeline_orders)
            inventory_position = closing_inv + pipeline_qty_before_order - backlog
            
            order_placed_tonight = 0
            
            if review_policy == "Periodic Review (Interval)":
                if day_counter % review_period == 0:
                    if inventory_position < target_stock_level:
                        order_placed_tonight = target_stock_level - inventory_position
                        target_delivery = day_counter + lead_time + 1
                        pipeline_orders.append({'delivery_day': target_delivery, 'qty': order_placed_tonight})
            else:
                if inventory_position <= reorder_point:
                    order_placed_tonight = order_qty
                    target_delivery = day_counter + lead_time + 1
                    pipeline_orders.append({'delivery_day': target_delivery, 'qty': order_qty})
            
            total_pipeline_inventory = sum(order['qty'] for order in pipeline_orders)
            
            if order_placed_tonight > 0:
                pipeline_status = f"Placed Order ({order_placed_tonight} units arriving Day {target_delivery} Morning)"
            elif total_pipeline_inventory > 0:
                pipeline_status = f"{total_pipeline_inventory} units en route"
            else:
                pipeline_status = "Clear"

            new_records.append({
                'Day': day_counter, 'Opening Stock': initial_opening_stock, 'Arrived Morning': arriving_qty,
                'Updated Opening Stock': updated_opening_stock, 'Demand Generated': demand, 'Sales Met': sales_met,
                'Shortage': shortage, 'Unfulfilled Backlog': backlog, 'Closing Inventory': closing_inv,
                'Order Placed Evening': order_placed_tonight, 'Total Pipeline Inventory': total_pipeline_inventory,
                'Pipeline Status': pipeline_status
            })
            
            current_inv = closing_inv

        if new_records:
            new_df = pd.DataFrame(new_records)
            st.session_state.t4_history = pd.concat([history_df, new_df], ignore_index=True)
            
        st.session_state.t4_day_counter = day_counter
        st.session_state.t4_current_inv = current_inv
        st.session_state.t4_backlog = backlog
        st.session_state.t4_pipeline_orders = pipeline_orders

    if st.button("🔄 Reset Simulation Data", key="t4_reset_btn"):
        st.session_state.t4_history = pd.DataFrame(columns=[
            'Day', 'Opening Stock', 'Arrived Morning', 'Updated Opening Stock', 
            'Demand Generated', 'Sales Met', 'Shortage', 'Unfulfilled Backlog', 
            'Closing Inventory', 'Order Placed Evening', 'Total Pipeline Inventory', 'Pipeline Status'
        ])
        st.session_state.t4_day_counter = 0
        st.session_state.t4_current_inv = starting_inventory
        st.session_state.t4_backlog = 0
        
        # Reset matching pipeline seed defaults
        if review_policy == "Periodic Review (Interval)":
            st.session_state.t4_pipeline_orders = [{'delivery_day': lead_time + 1, 'qty': int(avg_demand * review_period)}]
        else:
            st.session_state.t4_pipeline_orders = [{'delivery_day': lead_time + 1, 'qty': order_qty}]
        st.rerun()

    st.markdown("---")

    # =========================================================================
    # SECTION 4: SIMULATION INTERACTION BUTTONS
    # =========================================================================
    st.subheader("🕹️ Simulation Actions")
    col_btn1, col_btn2 = st.columns([1, 1.5])

    with col_btn1:
        if st.button("☀️ Next Day (Single Step)", use_container_width=True, key="t4_step_btn"):
            run_simulation_steps(1)

    with col_btn2:
        sim_days = st.number_input("Days to fast-forward", min_value=2, max_value=365, value=30, step=5, label_visibility="collapsed", key="t4_sim_days_input")
        if st.button(f"⏩ Simulate {sim_days} Days", use_container_width=True, key="t4_batch_btn"):
            run_simulation_steps(sim_days)

    # =========================================================================
    # SECTION 5: REAL-TIME ANALYTICS LEDGERS
    # =========================================================================
    if not st.session_state.t4_history.empty:
        df = st.session_state.t4_history
        
        total_shortages = df['Shortage'].sum()
        stockout_days = int((df['Shortage'] > 0).sum())
        service_level = (df['Sales Met'].sum() / df['Demand Generated'].sum()) * 100 if df['Demand Generated'].sum() > 0 else 100

        st.markdown("---")
        st.subheader("📊 Live Performance Scoreboard")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Day", int(df['Day'].iloc[-1]))
        m2.metric("Closing Stock", f"{int(df['Closing Inventory'].iloc[-1])} units")
        m3.metric("Service Level", f"{service_level:.1f}%")
        m4.metric("Pipeline Stock", f"{int(df['Total Pipeline Inventory'].iloc[-1])} units")
        m5.metric("Stock Out Days", f"{stockout_days} days", delta=f"{int(total_shortages)} units missed", delta_color="inverse")

        st.subheader("📈 Real-Time Tracking Analytics")
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.markdown("**Inventory Tracking Over Time**")
            net_inventory_curve = df['Closing Inventory'] - df['Unfulfilled Backlog']
            
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(
                x=df['Day'], y=net_inventory_curve, mode='lines+markers', name='Net Inventory State',
                line=dict(color='#3A96FF', width=2.5, shape='spline'), marker=dict(size=5, color='#3A96FF'),
                fill='tozeroy', fillcolor='rgba(58, 150, 255, 0.1)'
            ))
            
            if review_policy == "Periodic Review (Interval)":
                fig_inv.add_trace(go.Scatter(
                    x=df['Day'], y=[target_stock_level]*len(df), mode='lines', name='Target Stock Limit (T)',
                    line=dict(color='#A370F7', width=2, dash='longdash')
                ))
            else:
                fig_inv.add_trace(go.Scatter(
                    x=df['Day'], y=[reorder_point]*len(df), mode='lines', name='Reorder Point Target (ROP)',
                    line=dict(color='#FF5A5A', width=2, dash='dash')
                ))
                
            fig_inv.update_layout(**shared_layout, xaxis_title="Day", yaxis_title="Units State Balance", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            lowest_point = min(net_inventory_curve.min() - 20, -20)
            highest_point = max(df['Closing Inventory'].max() + 20, (target_stock_level if review_policy == "Periodic Review (Interval)" else reorder_point) + 20)
            fig_inv.update_yaxes(range=[lowest_point, highest_point])
            st.plotly_chart(fig_inv, use_container_width=True)

        is_zero_variation = (dist_type == "Uniform" and variation == 0) or (dist_type == "Normal" and std_dev == 0.0)

        with col_graph2:
            st.markdown("**Generated Demand Distribution**")
            fig_hist = go.Figure()
            
            if is_zero_variation:
                fig_hist.add_trace(go.Bar(
                    x=[avg_demand], y=[len(df)], name='Demand Frequency',
                    marker=dict(color='rgba(58, 150, 255, 0.4)', line=dict(color='#3A96FF', width=1.5)),
                    width=[4.0]
                ))
                fig_hist.update_layout(**shared_layout, bargap=0.08, yaxis_title="Days Logged", showlegend=False)
                fig_hist.update_xaxes(range=[avg_demand - 10, avg_demand + 10], tickvals=[avg_demand], title_text="Demand Bracket")
            else:
                if dist_type == "Uniform":
                    total_elements = high_bound - low_bound + 1
                    bin_size = 5 if total_elements % 5 == 0 else (10 if total_elements % 10 == 0 else max(1, total_elements // 5))
                    breaks = np.arange(low_bound - 0.5, high_bound + 0.5 + bin_size, bin_size)
                else:
                    breaks = np.histogram_bin_edges(df['Demand Generated'], bins='sturges')
                
                fig_hist.add_trace(go.Histogram(
                    x=df['Demand Generated'], xbins=dict(start=breaks[0], end=breaks[-1], size=(breaks[1] - breaks[0])),
                    autobinx=False, name='Demand Frequency',
                    marker=dict(color='rgba(58, 150, 255, 0.4)', line=dict(color='#3A96FF', width=1.5))
                ))
                fig_hist.update_layout(**shared_layout, bargap=0.08, xaxis_title="Demand Bracket", yaxis_title="Days Logged", showlegend=False)
                
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # COLLAPSIBLE TABLE 1: Distribution Table
        with st.expander("📊 View Distribution Bin Analysis Data Table", expanded=False):
            if is_zero_variation:
                bin_records = [{
                    "Demand Bracket Range": f"{avg_demand} to {avg_demand} units (Static)",
                    "Days Sampled (Count)": int(len(df)),
                    "Distribution Share (%)": "100.0%"
                }]
            else:
                counts, edges = np.histogram(df['Demand Generated'], bins=breaks)
                total_elements_count = len(df)
                bin_records = []
                for i in range(len(counts)):
                    lower_lbl = int(np.ceil(edges[i]))
                    upper_lbl = int(np.floor(edges[i+1]))
                    if dist_type == "Uniform" and (upper_lbl < low_bound or lower_lbl > high_bound):
                        continue
                    pct_share = (counts[i] / total_elements_count) * 100
                    bin_records.append({
                        "Demand Bracket Range": f"{lower_lbl} to {upper_lbl} units",
                        "Days Sampled (Count)": int(counts[i]),
                        "Distribution Share (%)": f"{pct_share:.1f}%"
                    })
            st.dataframe(pd.DataFrame(bin_records), use_container_width=True, hide_index=True)

        # COLLAPSIBLE TABLE 2: Operations Ledger
        with st.expander("📋 View Full Operations Ledger History Log", expanded=False):
            display_df = df.copy().sort_values(by='Day', ascending=False)
            st.dataframe(
                display_df, use_container_width=True, hide_index=True,
                column_config={
                    "Opening Stock": st.column_config.NumberColumn("Opening Stock (Yesterday)"),
                    "Arrived Morning": st.column_config.NumberColumn("☀️ Arrived Morning"),
                    "Updated Opening Stock": st.column_config.NumberColumn("🔄 Updated Opening Stock"),
                    "Unfulfilled Backlog": st.column_config.NumberColumn("🚨 Active Backlog"),
                    "Order Placed Evening": st.column_config.NumberColumn("🌙 Ordered Evening"),
                    "Total Pipeline Inventory": st.column_config.NumberColumn("📦 Total Pipeline Inventory"),
                    "Closing Inventory": st.column_config.NumberColumn("Closing Stock")
                }
            )
    else:
        st.info("💡 Interaction Required: Execute steps using the gameplay action controls above to populate tables and performance metrics.")



with tab5:
    st.header("Age Analysis")
    st.markdown("Upload your daily inventory transaction data to generate the aging dashboard.")
    
    # 1. FILE UPLOADER 
    uploaded_file = st.file_uploader(
        "Upload Inventory Data (CSV or Excel)", 
        type=['csv', 'xlsx'],
        key="tab5_uploader" 
    )
    
    # 2. LOCAL FUNCTION DEFINITION 
    def calculate_fifo_aging(df, initial_age_assumption, bucket_cutoffs, bucket_labels):
        """
        Calculates average inventory age and dynamic aging buckets using FIFO.
        """
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Resample to fill missing dates with 0
        df = df.set_index('Date').resample('D').asfreq().fillna(0).reset_index()
        
        inventory_queue = []
        daily_records = []
        
        for index, row in df.iterrows():
            current_date = row['Date']
            
            # Handle Day 1 Opening Balance
            if index == 0 and row['Opening Balance'] > 0:
                inventory_queue.append({
                    'received_date': current_date - pd.Timedelta(days=initial_age_assumption),
                    'qty': row['Opening Balance']
                })
                
            # Add newly received stock to the queue 
            if row['Receiving'] > 0:
                inventory_queue.append({
                    'received_date': current_date,
                    'qty': row['Receiving']
                })
                
            # Process Demand (FIFO: Consume oldest stock first) 
            demand_remaining = row['Demand/Sales']
            while demand_remaining > 0 and len(inventory_queue) > 0:
                oldest_batch = inventory_queue[0]
                if oldest_batch['qty'] <= demand_remaining:
                    demand_remaining -= oldest_batch['qty']
                    inventory_queue.pop(0) 
                else:
                    oldest_batch['qty'] -= demand_remaining
                    demand_remaining = 0 
                    
            # Calculate daily metrics
            total_qty = 0
            weighted_age_sum = 0
            
            # Initialize dynamic buckets for the day
            buckets = {label: 0 for label in bucket_labels}
            
            for batch in inventory_queue:
                qty = batch['qty']
                age_days = (current_date - batch['received_date']).days
                
                total_qty += qty
                weighted_age_sum += (qty * age_days)
                
                # Sort into dynamic user-defined buckets
                placed = False
                for i, cutoff in enumerate(bucket_cutoffs):
                    if age_days <= cutoff:
                        buckets[bucket_labels[i]] += qty
                        placed = True
                        break
                if not placed:
                    buckets[bucket_labels[-1]] += qty
                    
            avg_age = weighted_age_sum / total_qty if total_qty > 0 else 0
            
            # Save the record for this day
            record = {
                'Date': current_date,
                'Calculated Closing Balance': total_qty,
                'Average Age': avg_age,
                'Receiving': row['Receiving'],
                'Demand/Sales': row['Demand/Sales']
            }
            record.update(buckets)
            daily_records.append(record)
            
        return pd.DataFrame(daily_records)

    # 3. DASHBOARD LOGIC 
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                raw_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                raw_data = pd.read_excel(uploaded_file)
            
            # Clean up headers
            raw_data.columns = raw_data.columns.str.strip()

            st.success("Data loaded successfully!")
            st.divider()

            # User Inputs for Parameters
            col_param1, col_param2, col_param3 = st.columns(3)
            with col_param1:
                initial_age = st.slider("Assumed Age of Initial Opening Balance (Days)", 0, 120, 90, 10)
            with col_param2:
                critical_level = st.number_input("Critical Inventory Level", min_value=0, value=1000, step=100)
            with col_param3:
                # Dynamic Bucket Input
                bucket_str = st.text_input("Age Buckets (comma-separated days)", value="30, 60, 90")
                try:
                    custom_buckets = sorted([int(x.strip()) for x in bucket_str.split(',')])
                except:
                    st.warning("Invalid bucket format. Defaulting to 30, 60, 90.")
                    custom_buckets = [30, 60, 90]
            
            # Generate Bucket Labels Dynamically
            labels = []
            prev = 0
            for b in custom_buckets:
                labels.append(f"{prev}-{b}")
                prev = b + 1
            labels.append(f"{prev}+")
                
            # Process the uploaded data
            aging_df = calculate_fifo_aging(
                raw_data, 
                initial_age_assumption=initial_age,
                bucket_cutoffs=custom_buckets,
                bucket_labels=labels
            )
            
            # --- KPI DASHBOARD ---
            st.markdown("### Key Performance Indicators")
            current_balance = aging_df.iloc[-1]['Calculated Closing Balance']
            current_avg_age = aging_df.iloc[-1]['Average Age']
            
            total_days = len(aging_df)
            days_below_critical = len(aging_df[aging_df['Calculated Closing Balance'] < critical_level])
            pct_below_critical = (days_below_critical / total_days) * 100 if total_days > 0 else 0
            
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            kpi_col1.metric("Current Inventory Balance", f"{current_balance:,.0f} units")
            kpi_col2.metric("Current Average Age", f"{current_avg_age:.1f} Days")
            kpi_col3.metric("Time Below Critical Level", f"{pct_below_critical:.1f}%", help=f"Percentage of days inventory dropped below {critical_level}")
            
            st.divider()

            # --- % OF MATERIAL BY AGE (DATE SELECTOR) ---
            st.markdown("### % of Material by Age")
            
            max_date = aging_df['Date'].max().date()
            min_date = aging_df['Date'].min().date()
            
            selected_date = st.date_input(
                "Select a specific date to view inventory breakdown:", 
                value=max_date, 
                min_value=min_date, 
                max_value=max_date
            )
            
            target_date_df = aging_df[aging_df['Date'] == pd.to_datetime(selected_date)]
            
            if not target_date_df.empty:
                date_data = target_date_df.iloc[0]
                total_inv_on_date = date_data['Calculated Closing Balance']
                
                if total_inv_on_date > 0:
                    pct_cols = st.columns(len(labels))
                    for i, label in enumerate(labels):
                        qty_in_bucket = date_data[label]
                        pct_of_total = (qty_in_bucket / total_inv_on_date) * 100
                        
                        pct_cols[i].metric(
                            label=f"Bucket: {label} Days", 
                            value=f"{pct_of_total:.1f}%", 
                            delta=f"{qty_in_bucket:,.0f} units", 
                            delta_color="off"
                        )
                else:
                    st.info(f"Inventory balance was zero on {selected_date}.")
            else:
                st.warning("No data available for the selected date.")

            st.divider()
            
            # --- CHART 1: AVERAGE AGE + TRANSACTIONS (DUAL AXIS) ---
            st.subheader("Inventory Age & Daily Transactions")
            
            aging_df['Negative_Sales'] = aging_df['Demand/Sales'] * -1
            
            fig_age_tx = make_subplots(specs=[[{"secondary_y": True}]])
            
            # SOOTHING PALETTE: Sage Green & Muted Coral
            fig_age_tx.add_trace(
                go.Bar(x=aging_df['Date'], y=aging_df['Receiving'], name='Purchases (Inbound)', marker_color='#81b29a', opacity=0.85, marker_line_width=0),
                secondary_y=False
            )
            fig_age_tx.add_trace(
                go.Bar(x=aging_df['Date'], y=aging_df['Negative_Sales'], name='Sales (Outbound)', marker_color='#e07a5f', opacity=0.85, marker_line_width=0),
                secondary_y=False
            )
            fig_age_tx.add_trace(
                go.Scatter(x=aging_df['Date'], y=aging_df['Average Age'], name='Average Age', mode='lines', line=dict(color='#3d5a80', width=3)),
                secondary_y=True
            )
            
            fig_age_tx.update_layout(
                template="plotly_white", 
                barmode='relative', 
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40)
            )
            # Remove harsh gridlines
            fig_age_tx.update_yaxes(title_text="Transaction Volume (Units)", secondary_y=False, showgrid=True, gridcolor='#f0f0f0')
            fig_age_tx.update_yaxes(title_text="Average Age (Days)", secondary_y=True, showgrid=False)
            fig_age_tx.update_xaxes(showgrid=False)
            st.plotly_chart(fig_age_tx, use_container_width=True)
            
            # --- CHART 2: AGING BUCKETS (STRICT CURATED PALETTE) ---
            st.subheader("Aging Buckets")
            
            # Curated 8-step palette specifically stretching from your Navy to your Orange smoothly
            curated_palette = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
            
            # Pick colors evenly from the curated palette based on how many buckets the user chose
            if len(labels) == 4:
                custom_colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600'] # Exact 4 requested
            else:
                indices = np.linspace(0, len(curated_palette) - 1, len(labels), dtype=int)
                custom_colors = [curated_palette[i] for i in indices]
            
            color_mapping = {label: color for label, color in zip(labels, custom_colors)}
            stack_order = labels[::-1] # Oldest at the bottom
            
            fig_buckets = px.bar(
                aging_df, 
                x='Date', 
                y=stack_order,
                color_discrete_map=color_mapping 
            )
            
            fig_buckets.update_traces(marker_line_width=0)
            fig_buckets.update_layout(
                template="plotly_white", 
                barmode='stack', 
                yaxis_title="Units in Stock", 
                xaxis_title="",
                legend_title="Age (Days)",
                legend_traceorder="reversed"
            )
            fig_buckets.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
            fig_buckets.update_xaxes(showgrid=False)
            st.plotly_chart(fig_buckets, use_container_width=True)

            # --- CHART 3: INVENTORY GRAPH ---
            st.subheader("Total Inventory Trace")
            fig_inv = px.area(
                aging_df,
                x='Date',
                y='Calculated Closing Balance',
                color_discrete_sequence=['#003f5c']
            )
            
            fig_inv.update_traces(line=dict(width=0))
            fig_inv.update_layout(
                template="plotly_white", 
                yaxis_title="Total Units On Hand", 
                xaxis_title=""
            )
            fig_inv.add_hline(y=critical_level, line_dash="dot", annotation_text="Critical Level", line_color="#ef5675")
            fig_inv.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
            fig_inv.update_xaxes(showgrid=False)
            st.plotly_chart(fig_inv, use_container_width=True)

            with st.expander("View Daily Aging Data"):
                st.dataframe(aging_df)

        except KeyError as ke:
            st.error(f"Missing a required column: {ke}. Please ensure your file has 'Date', 'Opening Balance', 'Demand/Sales', and 'Receiving'.")
        except Exception as e:
            st.error(f"Error processing the file. Details: {e}")
            
    else:
        st.info("Awaiting file upload...")





with tab6:
    st.header("⚖️ Advanced Inventory Optimization Suite")
    st.markdown(
        "Analyze your inventory data through a twin-lens framework. First, review a historical backtest audit "
        "to identify legacy profit leaks."
    )
    
    # --- STEP 1: INPUT PARAMETERS ---
    st.subheader("1. Parameters & Cost Drivers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        item_unit_cost = st.number_input("Item Unit Cost ($/Unit)", min_value=0.01, value=25.00, step=1.00, key="unit_cost_global")
        # UPDATED: Replaced flat yearly cost with per-unit/per-day cost
        holding_fixed_daily = st.number_input("Fixed Holding Cost ($/Unit/Day)", min_value=0.0, value=0.0, step=0.01, key="fixed_hold_global")
        
    with col2:
        holding_var_pct = st.number_input("Variable Holding Cost (% of Item Cost/year)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="var_hold_global") / 100.0
        ordering_cost = st.number_input("Ordering Cost ($/order)", min_value=0.1, value=75.0, step=5.0, key="order_cost_global")
        
    with col3:
        lost_sales_penalty = st.number_input("Lost Sales Penalty ($/Unit Lost)", min_value=0.0, value=10.0, step=1.0, key="penalty_global")
        lead_time_days = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1, key="lt_global")

    st.markdown("---")
    col_sys1, col_sys2 = st.columns([1, 2])
    with col_sys1:
        review_system = st.radio("Inventory Review System Strategy", ["Continuous Review (Q, R)", "Periodic Review (P, T)"], key="review_system_global")
    with col_sys2:
        service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="service_level_global") / 100.0

    if review_system == "Periodic Review (P, T)":
        st.markdown("##### ⏳ Periodic Configuration")
        p_col1, _ = st.columns(2)
        with p_col1:
            user_p_days = st.number_input("Review Period Cycle (P in Days)", min_value=1, value=14, step=1, key="p_days_global")
    else:
        user_p_days = 1

    # --- STEP 2: MULTI-FORMAT DATA INGESTION ENGINE ---
    st.subheader("2. Upload Historical Invoices & Demand Data")
    uploaded_file = st.file_uploader(
        "Upload Inventory Ledger (Supports standard templates, raw ERP transactional logs, or stock card snapshots)", 
        type=["csv", "xlsx", "xls"], 
        key="uploader_global"
    )
    
    if uploaded_file is None:
        st.info("📥 Please upload your inventory ledger file (CSV or Excel) above to populate the suite modules.")
        st.stop()
        
    detected_sheet_opening_stock = None
    
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
            
        if "Date" not in raw_df.columns:
            st.error("❌ Missing required column: 'Date'.")
            st.stop()
            
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.sort_values(by="Date").reset_index(drop=True)
        
        open_balance_headers = ["Opening Balance", "Opening", "Opening_Stock", "Opening Stock"]
        for header in open_balance_headers:
            if header in raw_df.columns:
                detected_sheet_opening_stock = int(raw_df[header].iloc[0])
                break
        
        if "Demand_Qty" in raw_df.columns and "Purchase_Qty" in raw_df.columns:
            df_mapped = raw_df[["Date", "Demand_Qty", "Purchase_Qty"]].copy()
        elif "Demand" in raw_df.columns and "Stock Received" in raw_df.columns:
            df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df["Demand"], "Purchase_Qty": raw_df["Stock Received"]})
        elif ("Receiving" in raw_df.columns) and any(col in raw_df.columns for col in ["Demand/Sales", "Demand", "Sales"]):
            outbound_col = "Demand/Sales" if "Demand/Sales" in raw_df.columns else ("Demand" if "Demand" in raw_df.columns else "Sales")
            df_mapped = pd.DataFrame({"Date": raw_df["Date"], "Demand_Qty": raw_df[outbound_col], "Purchase_Qty": raw_df["Receiving"]})
        else:
            st.error("❌ Column layout structure mismatch.")
            st.stop()

        df_mapped = df_mapped.groupby("Date").agg({"Demand_Qty": "sum", "Purchase_Qty": "sum"}).reset_index()
        df_mapped = df_mapped.set_index("Date").resample("1D").asfreq()
        df_mapped["Demand_Qty"] = df_mapped["Demand_Qty"].fillna(0.0)
        df_mapped["Purchase_Qty"] = df_mapped["Purchase_Qty"].fillna(0.0)
        df = df_mapped.reset_index()
            
    except Exception as e:
        st.error(f"Error parsing file elements: {e}")
        st.stop()

    actual_orders_placed = np.count_nonzero(df["Purchase_Qty"])
    actual_total_units_purchased = df["Purchase_Qty"].sum()
    total_demand = df["Demand_Qty"].sum()

    # --- STEP 3: INITIAL BALANCE INPUT OVERRIDES ---
    avg_daily_demand_calc = df["Demand_Qty"].mean()
    
    if detected_sheet_opening_stock is not None:
        default_starting_stock = int(detected_sheet_opening_stock)
        help_text_note = "🚀 Auto-detected starting balance directly from your workbook upload."
    else:
        default_starting_stock = int(1.25 * (avg_daily_demand_calc * lead_time_days))
        help_text_note = "Calculated safety stock benchmark strategy formula default selection value."

    file_state_key = f"last_file_{uploaded_file.name}_{uploaded_file.size}"
    if "current_file_token" not in st.session_state or st.session_state.current_file_token != file_state_key:
        st.session_state.current_file_token = file_state_key
        st.session_state.opening_stock_global = default_starting_stock
        if "q_audit_suite" in st.session_state: del st.session_state.q_audit_suite
        if "rop_audit_suite" in st.session_state: del st.session_state.rop_audit_suite

    st.markdown("---")
    st.markdown("##### 📦 Initial Warehouse Capital Balance")
    opening_stock_override = st.number_input(
        "Initial Opening Stock (Day 1 On-Hand Baseline)",
        min_value=0, step=10, key="opening_stock_global", help=help_text_note
    )

    # --- HISTORICAL RUNNING BALANCE TABLE ---
    with st.expander("📋 View Complete Running Balance Table Snapshots", expanded=False):
        st.markdown(
            "An interactive historical stock card ledger driven directly by your initial opening stock parameter above. "
            "Tracks daily stock movements: $\\text{Closing Balance} = \\max(0, \\text{Opening Balance} + \\text{Receiving} - \\text{Demand})$."
        )
        
        cleaned_open_list = []
        cleaned_close_list = []
        temp_running_balance = opening_stock_override
        
        for idx, row in df.iterrows():
            cleaned_open_list.append(int(temp_running_balance))
            ending_balance_calc = max(0, temp_running_balance + row["Purchase_Qty"] - row["Demand_Qty"])
            cleaned_close_list.append(int(ending_balance_calc))
            temp_running_balance = ending_balance_calc
            
        full_stock_card_df = pd.DataFrame({
            "Timeline Date": df["Date"].dt.strftime('%Y-%m-%d'),
            "Opening Balance": cleaned_open_list,
            "Cleaned Demand Volume (Units)": df["Demand_Qty"].astype(int),
            "Consolidated Stock Received (Units)": df["Purchase_Qty"].astype(int),
            "Closing Balance": cleaned_close_list
        })
        
        st.dataframe(
            full_stock_card_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Opening Balance": st.column_config.NumberColumn(format="%d"),
                "Cleaned Demand Volume (Units)": st.column_config.NumberColumn(format="%d"),
                "Consolidated Stock Received (Units)": st.column_config.NumberColumn(format="%d"),
                "Closing Balance": st.column_config.NumberColumn(format="%d")
            }
        )

# --- STEP 4: ADVANCED STATISTICAL FIT RUNNER ---
    std_daily_demand = df["Demand_Qty"].std()
    annual_demand = avg_daily_demand_calc * 365
    
    annual_fixed_holding_per_unit = holding_fixed_daily * 365
    unit_holding_cost = annual_fixed_holding_per_unit + (item_unit_cost * holding_var_pct)
    
    cov = std_daily_demand / max(0.1, avg_daily_demand_calc)
    
    risk_horizon_days = lead_time_days if review_system == "Continuous Review (Q, R)" else (user_p_days + lead_time_days)
    rolling_risk_demand = df["Demand_Qty"].rolling(window=int(risk_horizon_days)).sum().dropna().values
    risk_mean = np.mean(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0
    risk_std = np.std(rolling_risk_demand) if len(rolling_risk_demand) > 0 else 0

    if len(rolling_risk_demand) > 0:
        if np.max(rolling_risk_demand) <= 0:
            # Safety net: If there is literally 0 demand across the entire dataset
            best_fit_name = "Zero Demand Base"
            raw_target_level = 0.0
        else:
            empirical_rop_raw = np.percentile(rolling_risk_demand, service_level * 100)
            
            # CRITICAL FIX: Epsilon smoothing to prevent SciPy FitDataError on exact 0s
            safe_demand = np.where(rolling_risk_demand <= 0, 1e-5, rolling_risk_demand)
            
            log_params = stats.lognorm.fit(safe_demand, floc=0)
            gam_params = stats.gamma.fit(safe_demand, floc=0)

            counts, bins = np.histogram(rolling_risk_demand, bins=20, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            rss_norm = np.sum((counts - stats.norm.pdf(bin_centers, loc=risk_mean, scale=risk_std)) ** 2)
            rss_log = np.sum((counts - stats.lognorm.pdf(bin_centers, *log_params)) ** 2)
            rss_gam = np.sum((counts - stats.gamma.pdf(bin_centers, *gam_params)) ** 2)

            if cov > 0.75:
                best_fit_name = "Empirical (Data-Driven)"
                raw_target_level = empirical_rop_raw
            else:
                errors = {"Normal": rss_norm, "Log-Normal": rss_log, "Gamma": rss_gam}
                best_fit_name = min(errors, key=errors.get)
                if best_fit_name == "Normal":
                    raw_target_level = stats.norm.ppf(service_level, loc=risk_mean, scale=risk_std)
                elif best_fit_name == "Log-Normal":
                    raw_target_level = stats.lognorm.ppf(service_level, *log_params)
                else:
                    raw_target_level = stats.gamma.ppf(service_level, *gam_params)
    else:
        best_fit_name = "Default (Insufficient Data)"
        raw_target_level = avg_daily_demand_calc * risk_horizon_days
        
    raw_optimal_q = np.sqrt((2 * annual_demand * ordering_cost) / max(0.01, unit_holding_cost))

    if "q_audit_suite" not in st.session_state:
        st.session_state.q_audit_suite = max(1, int(raw_optimal_q)) if review_system == "Continuous Review (Q, R)" else int(avg_daily_demand_calc * user_p_days)
    if "rop_audit_suite" not in st.session_state:
        st.session_state.rop_audit_suite = max(0, int(raw_target_level))

    

    # --- HISTOGRAM EXPANDER ---
    with st.expander("📊 View Cleaned Demand Distribution & Best-Fit Curve Metrics", expanded=False):
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Average Daily Demand", f"{avg_daily_demand_calc:.2f} units")
        stat_col2.metric("Coefficient of Variation (CV)", f"{cov:.2f}")
        stat_col3.metric("Engine Selection", f"✨ {best_fit_name}")
        st.markdown("---")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=df["Demand_Qty"], name="Historical Days", marker_color='#1F77B4', opacity=0.6))
        hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Demand Quantity (Units / Day)", yaxis_title="Frequency", margin=dict(l=40, r=40, t=10, b=40), height=300)
        st.plotly_chart(hist_fig, use_container_width=True)

    # --- STEP 5: POLICY OPTIMIZATION TUNING CONFIG ---
    st.subheader("3. Policy Optimization & Parameter Tuning")
    adjust_col1, adjust_col2 = st.columns(2)
    with adjust_col1:
        if review_system == "Continuous Review (Q, R)":
            final_q = st.number_input("Target Order Quantity (Q)", min_value=1, step=10, key="q_audit_suite")
        else:
            cycle_demand_baseline = int(avg_daily_demand_calc * user_p_days)
            final_q = st.number_input("Average Target Batch Size (Q)", min_value=1, value=cycle_demand_baseline, step=10, disabled=True, key="q_audit_suite_disabled")
    with adjust_col2:
        final_buffer_target = st.number_input("Reorder Point (ROP) / Target Level (T)", min_value=0, step=10, key="rop_audit_suite")

    if review_system == "Continuous Review (Q, R)":
        st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Optimal Order Quantity (EOQ): **{int(raw_optimal_q):,}** units | Recommended Reorder Point (ROP): **{int(raw_target_level):,}** units.")
    else:
        st.info(f"🎯 **Engine-Calculated Benchmarks ({best_fit_name}):** Expected Cycle Batch Size: **{int(avg_daily_demand_calc * user_p_days):,}** units | Recommended Max Order Up-To Level (T): **{int(raw_target_level):,}** units.")

    optimal_p_days = max(1, int((final_q / max(0.1, avg_daily_demand_calc)))) if review_system == "Continuous Review (Q, R)" else int(user_p_days)

    # ==========================================
    #      SECTION A: HISTORICAL BACKTEST
    # ==========================================
    st.markdown("---")
    st.header("📊 Section A: Historical Backtest Audit")
    st.markdown(
        "This analysis compares your **Historical Actuals** (what actually happened in your ledger) against our **Recommended Optimized Policy** "
        "under identical historical demand constraints. This reveals the true 'Efficiency Gap'."
    )
    
    # --- SIMULATION 1: ACTUAL HISTORICAL RECORD ---
    current_inv_act = opening_stock_override  
    inv_levels_act = []
    lost_sales_qty_act = 0
    stockout_days_act = 0
    zero_stock_days_act = 0
    
    for i, row in df.iterrows():
        demand_today = row["Demand_Qty"]
        current_inv_act += row["Purchase_Qty"] 
        
        if current_inv_act < demand_today:
            lost_sales_qty_act += (demand_today - current_inv_act)
            stockout_days_act += 1
            current_inv_act = 0
        else:
            current_inv_act -= demand_today
            
        if current_inv_act == 0:
            zero_stock_days_act += 1
            
        inv_levels_act.append(current_inv_act)

    # --- SIMULATION 2: OPTIMIZED POLICY TRACE ---
    current_inv_opt = opening_stock_override
    inv_levels_opt, lost_sales_series_opt, pipeline_orders_opt, policy_orders_series = [], [], [], []
    opt_orders_placed, stockout_days_opt, lost_sales_qty_opt = 0, 0, 0
    policy_total_units_ordered = 0
    zero_stock_days_opt = 0

    for i, row in df.iterrows():
        demand_today = row["Demand_Qty"]
        arriving_stock_opt = sum(qty for delivery_day, qty in pipeline_orders_opt if delivery_day == i)
        current_inv_opt += arriving_stock_opt
        today_placed_opt = 0
        
        if current_inv_opt < demand_today:
            today_lost_opt = demand_today - current_inv_opt
            lost_sales_qty_opt += today_lost_opt
            stockout_days_opt += 1
            current_inv_opt = 0
        else:
            today_lost_opt = 0
            current_inv_opt -= demand_today
            
        if current_inv_opt == 0:
            zero_stock_days_opt += 1
            
        if review_system == "Continuous Review (Q, R)":
            total_net_stock_opt = current_inv_opt + sum(qty for dd, qty in pipeline_orders_opt if dd > i)
            if total_net_stock_opt <= final_buffer_target:
                pipeline_orders_opt.append((i + int(lead_time_days), final_q))
                opt_orders_placed += 1
                today_placed_opt = final_q
                policy_total_units_ordered += final_q
        else:
            if i % optimal_p_days == 0:
                total_net_stock_opt = current_inv_opt + sum(qty for dd, qty in pipeline_orders_opt if dd > i)
                order_qty_opt = max(0, final_buffer_target - total_net_stock_opt)
                if order_qty_opt > 0:
                    pipeline_orders_opt.append((i + int(lead_time_days), order_qty_opt))
                    opt_orders_placed += 1
                    today_placed_opt = order_qty_opt
                    policy_total_units_ordered += order_qty_opt
                    
        inv_levels_opt.append(current_inv_opt)
        lost_sales_series_opt.append(today_lost_opt)
        policy_orders_series.append(today_placed_opt)

    # --- MASTER CALCULATION COMPILING ENGINE ---
    actual_max_inventory = np.max(inv_levels_act)
    actual_min_inventory = np.min(inv_levels_act)
    actual_avg_inventory = np.mean(inv_levels_act)
    actual_fill_rate = max(0.0, 1.0 - (lost_sales_qty_act / max(1, total_demand)))
    actual_cycle_time = 365 / actual_orders_placed if actual_orders_placed > 0 else 365.0
    actual_avg_order_size = actual_total_units_purchased / actual_orders_placed if actual_orders_placed > 0 else 0.0

    actual_total_ordering_cost = actual_orders_placed * ordering_cost
    # Because unit_holding_cost is now explicitly calculated as an annual cost based on the per-day parameter, 
    # scaling this by actual_avg_inventory structurally models holding costs rising dynamically with stock volume.
    actual_total_holding_cost = actual_avg_inventory * unit_holding_cost
    actual_lost_sales_financial = lost_sales_qty_act * lost_sales_penalty
    actual_total_cost = actual_total_ordering_cost + actual_total_holding_cost + actual_lost_sales_financial

    simmed_avg_opt_inv = np.mean(inv_levels_opt)
    simmed_max_opt_inv = np.max(inv_levels_opt)
    simmed_min_inventory = np.min(inv_levels_opt)
    simmed_opt_fill_rate = max(0.0, 1.0 - (lost_sales_qty_opt / max(1, total_demand)))
    policy_cycle_time = 365 / opt_orders_placed if opt_orders_placed > 0 else 365.0
    policy_avg_order_size = policy_total_units_ordered / opt_orders_placed if opt_orders_placed > 0 else 0.0

    optimal_ordering_cost = opt_orders_placed * ordering_cost
    optimal_holding_cost = simmed_avg_opt_inv * unit_holding_cost
    optimal_lost_sales_financial = lost_sales_qty_opt * lost_sales_penalty
    optimal_total_cost = optimal_ordering_cost + optimal_holding_cost + optimal_lost_sales_financial

    true_net_benefit = actual_total_cost - optimal_total_cost
    
    if true_net_benefit > 0:
        st.success(f"### 🎯 The Efficiency Opportunity\nBy shifting to the recommended optimized policy, you would have recovered **${true_net_benefit:,.2f}** over this historical period. This validates that the optimized parameters structurally outperform legacy ordering habits.")
    else:
        st.error(f"⚠️ **Operational Margin Deficit Risk:** This setup increases operational overhead by **${abs(true_net_benefit):,.2f} / year** compared to actuals.")


    # =========================================================
    # --- NEW: EXECUTIVE KPI SCORECARD ---
    # =========================================================
    st.markdown("### 🏆 Executive Summary: Value Realization")
    
    # Calculate Working Capital metrics for the scorecard
    act_avg_wc = actual_avg_inventory * item_unit_cost
    opt_avg_wc = simmed_avg_opt_inv * item_unit_cost
    cash_released = act_avg_wc - opt_avg_wc

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric(label="Total Cost Saving", value=f"${true_net_benefit:,.0f}")

    with kpi_col2:
        st.metric(label="Optimized Fill Rate", value=f"{simmed_opt_fill_rate * 100:.1f}%")

    with kpi_col3:
        st.metric(label="Avg Working Capital (Opt)", value=f"${opt_avg_wc:,.0f}")
        # Custom HTML to display the historical data in small text right below the metric
        st.markdown(
            f"<div style='margin-top: -15px; font-size: 0.85rem; color: gray;'>Historical: ${act_avg_wc:,.0f}</div>", 
            unsafe_allow_html=True
        )

    with kpi_col4:
        release_label = "Cash Released" if cash_released >= 0 else "Capital Added (Tied Up)"
        st.metric(label=release_label, value=f"${abs(cash_released):,.0f}")

    st.markdown("---")

    
    # =========================================================
    # =========================================================
    # --- CLUSTERED EXECUTIVE MATRIX TABLES ---
    # =========================================================
    def render_clustered_matrix(title, metrics, act_vals, pol_vals, formats):
        st.markdown(f"#### {title}")
        abs_var = [a - p for a, p in zip(act_vals, pol_vals)]
        pct_var = []
        for a, p in zip(act_vals, pol_vals):
            if a == 0: pct_var.append(0.0)
            else: pct_var.append(((a - p) / a) * 100)
            
        m_df = pd.DataFrame({"Operational Attribute Pillar": metrics})
        for idx in range(len(metrics)):
            fmt = formats[idx]
            if fmt == "currency":
                m_df.at[idx, "Historical Actuals"] = f"${act_vals[idx]:,.2f}"
                m_df.at[idx, "Optimized Policy"] = f"${pol_vals[idx]:,.2f}"
                m_df.at[idx, "Net Delta Variance"] = f"${abs_var[idx]:,.2f}" if abs_var[idx] >= 0 else f"-${abs(abs_var[idx]):,.2f}"
                m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
            elif fmt == "pct":
                m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:.1f}%"
                m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:.1f}%"
                m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+.1f}% pts"
                m_df.at[idx, "% Impact Efficiency"] = f"{pol_vals[idx] - act_vals[idx]:+.1f}% pts"
            elif fmt == "days":
                m_df.at[idx, "Historical Actuals"] = f"{act_vals[idx]:,.1f} days"
                m_df.at[idx, "Optimized Policy"] = f"{pol_vals[idx]:,.1f} days"
                m_df.at[idx, "Net Delta Variance"] = f"{abs_var[idx]:+,.1f} days"
                m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"
            else:
                m_df.at[idx, "Historical Actuals"] = f"{int(act_vals[idx]):,}"
                m_df.at[idx, "Optimized Policy"] = f"{int(pol_vals[idx]):,}"
                m_df.at[idx, "Net Delta Variance"] = f"{int(abs_var[idx]):+1,}"
                m_df.at[idx, "% Impact Efficiency"] = f"{pct_var[idx]:+.1f}%"

        def apply_matrix_styles(x):
            colors = pd.DataFrame('', index=x.index, columns=x.columns)
            fav = 'background-color: #1A3E2B; color: #81C784; font-weight: bold;'
            unfav = 'background-color: #3E1A1A; color: #E57373;'
            for i, metric in enumerate(metrics):
                v = abs_var[i]
                if title == "1. Financial Breakdown Matrix":
                    if v > 0: colors.iloc[i, 3:] = fav
                    elif v < 0: colors.iloc[i, 3:] = unfav
                elif title == "3. Working Capital Release Matrix":
                    if v > 0: colors.iloc[i, 3:] = fav
                    elif v < 0: colors.iloc[i, 3:] = unfav
                elif title == "4. Stockout Risk & Vulnerability Matrix":
                    if "Fill Rate" in metric:
                        if pol_vals[i] > act_vals[i]: colors.iloc[i, 3:] = fav
                        elif pol_vals[i] < act_vals[i]: colors.iloc[i, 3:] = unfav
                    else:
                        if v > 0: colors.iloc[i, 3:] = fav
                        elif v < 0: colors.iloc[i, 3:] = unfav
            return colors
            
        st.dataframe(m_df.style.apply(apply_matrix_styles, axis=None), use_container_width=True, hide_index=True)

    # ==========================
    # MATRIX 1: FINANCIALS
    # ==========================
    render_clustered_matrix(
        "1. Financial Breakdown Matrix",
        ["Annual Ordering Fees ($)", "Annual Storage Carrying Cost ($)", "Financial Penalty from Stockouts ($)", "Total Policy Operating Cost ($)"],
        [actual_total_ordering_cost, actual_total_holding_cost, actual_lost_sales_financial, actual_total_cost],
        [optimal_ordering_cost, optimal_holding_cost, optimal_lost_sales_financial, optimal_total_cost],
        ["currency", "currency", "currency", "currency"]
    )
    
    with st.expander("📊 View Financial Impact Breakdown", expanded=True):
        cost_categories = ['Ordering Fees', 'Storage Carrying Cost', 'Lost Sales Penalty', 'Total Operating Cost']
        cost_fig = go.Figure(data=[
            go.Bar(name='Historical Actuals', x=cost_categories, y=[actual_total_ordering_cost, actual_total_holding_cost, actual_lost_sales_financial, actual_total_cost], marker_color='#B0C4DE', text=[actual_total_ordering_cost, actual_total_holding_cost, actual_lost_sales_financial, actual_total_cost], texttemplate='$%{text:,.0f}', textposition='outside'),
            go.Bar(name='Optimized Policy', x=cost_categories, y=[optimal_ordering_cost, optimal_holding_cost, optimal_lost_sales_financial, optimal_total_cost], marker_color='#1F77B4', text=[optimal_ordering_cost, optimal_holding_cost, optimal_lost_sales_financial, optimal_total_cost], texttemplate='$%{text:,.0f}', textposition='outside')
        ])
        cost_fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Annual Cost ($)", margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        cost_fig.update_yaxes(range=[0, max(actual_total_cost, optimal_total_cost) * 1.15])
        st.plotly_chart(cost_fig, use_container_width=True)

    st.markdown("---")

    # ==========================
    # MATRIX 2: LOGISTICS
    # ==========================
    render_clustered_matrix(
        "2. Logistical Operations Footprint Matrix",
        ["Average Volume Kept On-Hand", "Maximum Storage Spike Level", "Total Orders Dispatched", "Average Logistics Cycle Time", "Average Order Shipment Size"],
        [actual_avg_inventory, actual_max_inventory, actual_orders_placed, actual_cycle_time, actual_avg_order_size],
        [simmed_avg_opt_inv, simmed_max_opt_inv, opt_orders_placed, policy_cycle_time, policy_avg_order_size],
        ["units", "units", "count", "days", "units"]
    )
    
    with st.expander("📊 View Logistical Footprint Breakdown", expanded=False):
        log_fig = go.Figure(data=[
            go.Bar(name='Historical Actuals', x=['Avg Volume On-Hand', 'Max Storage Spike', 'Avg Order Size'], y=[actual_avg_inventory, actual_max_inventory, actual_avg_order_size], marker_color='#B0C4DE', text=[actual_avg_inventory, actual_max_inventory, actual_avg_order_size], texttemplate='%{text:,.0f} units', textposition='outside'),
            go.Bar(name='Optimized Policy', x=['Avg Volume On-Hand', 'Max Storage Spike', 'Avg Order Size'], y=[simmed_avg_opt_inv, simmed_max_opt_inv, policy_avg_order_size], marker_color='#1F77B4', text=[simmed_avg_opt_inv, simmed_max_opt_inv, policy_avg_order_size], texttemplate='%{text:,.0f} units', textposition='outside')
        ])
        log_fig.update_layout(title="Inventory Volume Comparison", barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Units", margin=dict(t=40, b=40), legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        log_fig.update_yaxes(range=[0, max(actual_max_inventory, simmed_max_opt_inv) * 1.15])
        st.plotly_chart(log_fig, use_container_width=True)

    st.markdown("---")

    # ==========================
    # MATRIX 3: WORKING CAPITAL
    # ==========================
    act_max_wc = actual_max_inventory * item_unit_cost
    act_min_wc = actual_min_inventory * item_unit_cost
    act_avg_wc = actual_avg_inventory * item_unit_cost

    opt_max_wc = simmed_max_opt_inv * item_unit_cost
    opt_min_wc = simmed_min_inventory * item_unit_cost
    opt_avg_wc = simmed_avg_opt_inv * item_unit_cost
    
    render_clustered_matrix(
        "3. Working Capital Release Matrix",
        ["Peak Working Capital Tied Up ($)", "Average Working Capital Tied Up ($)", "Minimum Base Working Capital ($)"],
        [act_max_wc, act_avg_wc, act_min_wc],
        [opt_max_wc, opt_avg_wc, opt_min_wc],
        ["currency", "currency", "currency"]
    )
    
    with st.expander("📊 View Working Capital Breakdown", expanded=False):
        wc_fig = go.Figure(data=[
            go.Bar(name='Historical Actuals', x=['Peak Capital Tied Up', 'Average Capital Tied Up', 'Minimum Capital Tied Up'], y=[act_max_wc, act_avg_wc, act_min_wc], marker_color='#B0C4DE', text=[act_max_wc, act_avg_wc, act_min_wc], texttemplate='$%{text:,.0f}', textposition='outside'),
            go.Bar(name='Optimized Policy', x=['Peak Capital Tied Up', 'Average Capital Tied Up', 'Minimum Capital Tied Up'], y=[opt_max_wc, opt_avg_wc, opt_min_wc], marker_color='#1F77B4', text=[opt_max_wc, opt_avg_wc, opt_min_wc], texttemplate='$%{text:,.0f}', textposition='outside')
        ])
        wc_fig.update_layout(title="Capital Allocation Status", barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Capital Frozen ($)", margin=dict(t=40, b=40), legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        wc_fig.update_yaxes(range=[0, max(act_max_wc, opt_max_wc) * 1.15])
        st.plotly_chart(wc_fig, use_container_width=True)

    st.markdown("---")

    # ==========================
    # MATRIX 4: STOCKOUT RISK
    # ==========================
    render_clustered_matrix(
        "4. Stockout Risk & Vulnerability Matrix",
        ["Absolute Minimum Buffer Stock", "Stockout Events (Unfulfilled Days)", "Total Unfulfilled Deficit Volume", "Days with Absolute Zero Closing Stock", "Achieved Order Fill Rate (%)"],
        [actual_min_inventory, stockout_days_act, lost_sales_qty_act, zero_stock_days_act, actual_fill_rate * 100],
        [simmed_min_inventory, stockout_days_opt, lost_sales_qty_opt, zero_stock_days_opt, simmed_opt_fill_rate * 100],
        ["units", "count", "units", "count", "pct"]
    )
    
    with st.expander("📊 View Stockout Risk Breakdown", expanded=False):
        risk_fig = go.Figure(data=[
            go.Bar(name='Historical Actuals', x=['Stockout Events (Days)', 'Zero Stock Events (Days)'], y=[stockout_days_act, zero_stock_days_act], marker_color='#B0C4DE', text=[stockout_days_act, zero_stock_days_act], texttemplate='%{text:,.0f} Days', textposition='outside'),
            go.Bar(name='Optimized Policy', x=['Stockout Events (Days)', 'Zero Stock Events (Days)'], y=[stockout_days_opt, zero_stock_days_opt], marker_color='#1F77B4', text=[stockout_days_opt, zero_stock_days_opt], texttemplate='%{text:,.0f} Days', textposition='outside')
        ])
        risk_fig.update_layout(title="Supply Chain Disruptions", barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Count (Days)", margin=dict(t=40, b=40), legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        risk_fig.update_yaxes(range=[0, max(stockout_days_act, stockout_days_opt, zero_stock_days_act, zero_stock_days_opt) * 1.25])
        st.plotly_chart(risk_fig, use_container_width=True)

    st.markdown("---")

    # --- SIMULATION CHARTS (TIMELINE) ---
    st.markdown("### 📈 Tactical Operations Timeline Visualizations")
    timeline_fig = go.Figure()
    timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_act, name="Historical Actuals (Ledger)", line=dict(color='#B0C4DE', width=2), fill='tozeroy', fillcolor='rgba(176, 196, 222, 0.15)'))
    timeline_fig.add_trace(go.Scatter(x=df["Date"], y=inv_levels_opt, name=f"Recommended Optimized Policy ({best_fit_name.split(' ')[0]})", line=dict(color='#1F77B4', width=2.5)))
    timeline_fig.add_trace(go.Scatter(x=df["Date"], y=[max(0, raw_target_level - risk_mean)] * len(df), name="Calculated Safety Stock Floor", line=dict(color='#FF4B4B', width=1.5, dash='dot')))
    timeline_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Timeline Date", yaxis_title="On-Hand Inventory (Units)", height=350, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
    st.plotly_chart(timeline_fig, use_container_width=True)

    # --- UNDERLYING DATA TABLE ---
    st.markdown("### 📊 Underlying Timeline Data")
    chart_data_df = pd.DataFrame({
        "Date": df["Date"].dt.strftime('%Y-%m-%d'),
        "Actual Historical (Units)": inv_levels_act,
        "Optimized Model (Units)": inv_levels_opt,
        "Safety Stock Floor (Units)": [max(0, raw_target_level - risk_mean)] * len(df)
    })
    
    st.dataframe(
        chart_data_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Actual Historical (Units)": st.column_config.NumberColumn(format="%d"),
            "Optimized Model (Units)": st.column_config.NumberColumn(format="%d"),
            "Safety Stock Floor (Units)": st.column_config.NumberColumn(format="%d")
        }
    )


    
    # =========================================================
    # --- NEW: OPTIMIZED POLICY DETAILED LEDGER ---
    # =========================================================
    st.markdown("---")
    with st.expander("📋 View Detailed Optimized Policy Ledger", expanded=False):
        st.markdown(
            "A day-by-day breakdown of how the Recommended Optimized Policy handles historical demand, "
            "including specific order placement triggers and stockout mitigation."
        )
        
        optimized_ledger_df = pd.DataFrame({
            "Date": df["Date"].dt.strftime('%Y-%m-%d'),
            "Historical Demand (Units)": df["Demand_Qty"].astype(int),
            "Optimized Orders Placed (Units)": np.array(policy_orders_series).astype(int),
            "Optimized Closing Balance (Units)": np.array(inv_levels_opt).astype(int),
            "Optimized Lost Sales (Units)": np.array(lost_sales_series_opt).astype(int)
        })
        
        st.dataframe(
            optimized_ledger_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Historical Demand (Units)": st.column_config.NumberColumn(format="%d"),
                "Optimized Orders Placed (Units)": st.column_config.NumberColumn(format="%d"),
                "Optimized Closing Balance (Units)": st.column_config.NumberColumn(format="%d"),
                "Optimized Lost Sales (Units)": st.column_config.NumberColumn(format="%d")
            }
        )
        
        st.download_button(
            label="📥 Download Optimized Policy Ledger (CSV)", 
            data=optimized_ledger_df.to_csv(index=False).encode('utf-8'), 
            file_name="optimized_inventory_ledger.csv", 
            mime="text/csv"
        )
