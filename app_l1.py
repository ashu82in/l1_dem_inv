import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import plotly.graph_objects as go
from prophet import Prophet

# --- Session State Initialization ---
if 'next_clicked' not in st.session_state:
    st.session_state.next_clicked = False
if 'seed_counter' not in st.session_state:
    st.session_state.seed_counter = 42

st.set_page_config(page_title="Supply Chain Analytics Platform", layout="wide")

st.title("🚀 Supply Chain Analytics Platform")

tab1, tab2, tab3, tab4 = st.tabs(["Average Demand", "📊 Demand Histogram", "📈 Demand Forecasting", "📦 Inventory Optimization"])

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
    st.header("Inventory Optimization Insights")
    st.info("Safety stock and reorder point metrics can be implemented here.")
