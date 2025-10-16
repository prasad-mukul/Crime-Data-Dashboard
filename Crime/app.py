import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from backend import (
    load_data, get_years, get_states, filter_state_district, 
    calculate_safety_ratio, get_top_crime_composition, 
    authenticate_user, register_user, is_username_registered
)

# Set page config once
st.set_page_config(page_title="Crime Visualization Dashboard", layout="centered")

# --- Initial Data Load & Setup ---
data = load_data() 

# --- INITIALIZE LOGIN STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    st.session_state.page = page

def logout():
    """Resets session state to log the user out."""
    st.session_state.logged_in = False
    st.session_state.page = "Home"
    st.session_state.username = None
    # st.rerun() removed to avoid "no-op" warning.

# ============================
# LOGIN/REGISTRATION PAGE FUNCTION (SIMPLE FULL-WIDTH)
# ============================
def login_page():
    st.markdown("<h1 style='text-align:center; color: #1f77b4;'>🕵 Crime Data Visualization Dashboard</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; font-size: 1.1em;'>Please log in or register to access the full analytical features.</p>", unsafe_allow_html=True)
    st.divider()

    # Login/Register tabs spanning full width (simple design)
    login_tab, register_tab = st.tabs(["🔒 Login", "📝 Register"])

    with login_tab:
        st.subheader("Existing User Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            login_submitted = st.form_submit_button("Log In", type="primary", width='stretch')

            if login_submitted:
                if authenticate_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success(f"Welcome back, {login_username}!")
                    st.rerun()
                else:
                    # Enhanced error feedback
                    if is_username_registered(login_username):
                         st.error("Login failed: Incorrect password.")
                    else:
                         st.error("Login failed: This username is not registered.")


    with register_tab:
        st.subheader("New User Registration")
        with st.form("register_form"):
            reg_username = st.text_input("New Username", key="reg_user")
            reg_password = st.text_input("New Password", type="password", key="reg_pass")
            # Added Confirm Password
            reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_pass")
            reg_submitted = st.form_submit_button("Register", width='stretch')

            if reg_submitted:
                if reg_password != reg_confirm_password:
                    st.error("Registration failed: Passwords do not match.")
                else:
                    success, message = register_user(reg_username, reg_password)
                    if success:
                        st.success(message)
                    else:
                        st.error("Registration failed. " + message)


# =================================================================
# MAIN APPLICATION FLOW CONTROL (Dashboard content protected by login)
# =================================================================

if not st.session_state.logged_in:
    login_page()
else:
    # --- GLOBAL HEADER & LOGOUT BUTTON ---
    # The header is now compact, combining the welcome message and logout button
    header_col1, header_col2 = st.columns([7, 2])
    
    with header_col1:
        st.markdown(f"*Welcome, {st.session_state.username}!*") # Welcome message
    
    with header_col2:
        # Logout button styled to fit on one line
        st.button("🔓 Logout", on_click=logout, type='primary', width='stretch')
    
    # Main Title and Subtitle (placed below the welcome/logout row for clean layout)
    st.markdown("<h1 style='text-align:center; color: #1f77b4;'>🕵 Crime Data Visualization Dashboard</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; font-size: 1.1em;'>Explore crime statistics, safety insights, and predictive trends across Indian states (2013 data)</p>", unsafe_allow_html=True)
    
    st.divider()


    # ============================
    # HOME PAGE
    # ============================
    if st.session_state.page == "Home":
        st.markdown("<h3 style='text-align:center; color: #34495E;'>Choose an Option to Explore</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.button("🔍 State Crime Search", width='stretch', on_click=lambda: go_to("CrimeSearch"))
        with col2:
            st.button("🛡 Safety Ratio", width='stretch', on_click=lambda: go_to("SafetyRatio"))
        with col3:
            st.button("⚖ Compare Two States", width='stretch', on_click=lambda: go_to("Compare"))

        st.markdown("")
        col4, col5 = st.columns(2, gap="large")
        with col4:
            st.button("📈 Crime Trends", width='stretch', on_click=lambda: go_to("Trends"))
        with col5:
            st.button("🔮 Future Prediction", width='stretch', on_click=lambda: go_to("Predict"))

        st.markdown("---")
        st.caption("Tip: Click any button to explore detailed insights for crime data visualization.")

        # ============================
    # STATE CRIME SEARCH (fixed)
    # ============================
    elif st.session_state.page == "CrimeSearch":
        st.markdown("## 🔍 State Crime Search")
        st.button("⬅ Back to Home", on_click=lambda: go_to("Home"))
        
        # --- Filters ---
        years = get_years(data)
        selected_year = st.selectbox("Select Year", years) if years and len(years) > 0 else None

        states = get_states(data)
        selected_state = st.selectbox("Select State/UT", states) if states else None

        # Defensive check: ensure data is loaded and has STATE/UT
        if data.empty:
            st.error("Data not loaded. Please check crime.csv and backend.load_data().")
            st.stop()
        if "STATE/UT" not in data.columns:
            st.error("Column 'STATE/UT' missing in dataset. Please verify your CSV header.")
            st.stop()

        # Filter state data safely (STATE/UT values in backend are normalized to UPPER)
        # ---- Robust State Filtering ----
        selected_district = None  # always initialize

        if selected_state:
    # Match more loosely to avoid missing due to stray spaces or formatting
            state_data_filtered = data[data["STATE/UT"].str.contains(selected_state.upper(), na=False)]
        else:
            state_data_filtered = pd.DataFrame()

        # ---- District Dropdown ----
        if not state_data_filtered.empty and "DISTRICT" in state_data_filtered.columns:
            districts = sorted([d for d in state_data_filtered["DISTRICT"].unique() if isinstance(d, str) and d.strip() != ""])
    
            if districts:
                selected_district = st.selectbox("Select District", districts)
            else:
                st.warning(f"⚠ No districts found for state: {selected_state}")
        else:
            st.warning(f"⚠ No matching rows for state: {selected_state} (Check CSV formatting)")


        # Safe district dropdown: only show when DISTRICT column exists and there is data
        if not state_data_filtered.empty and "DISTRICT" in state_data_filtered.columns:
            # Remove empty / nan district labels and keep unique values
            districts = [d for d in state_data_filtered["DISTRICT"].unique() if pd.notna(d) and str(d).strip() != ""]
            if len(districts) > 0:
                selected_district = st.selectbox("Select District", districts)
            else:
                st.warning("⚠️ No districts found for this state (DISTRICT values are empty).")
        else:
            # If no data found for the state or DISTRICT column missing, inform the user
            if state_data_filtered.empty and selected_state:
                st.warning(f"⚠️ No data rows found for state: {selected_state}. Check CSV capitalization or data.")
            elif "DISTRICT" not in data.columns:
                st.warning("⚠️ 'DISTRICT' column missing in the dataset. Please verify your CSV header.")
            districts = []

        st.markdown("---")

        # Only proceed if both a state and a district are selected
        if selected_state and selected_district:
            # Use backend filter function; pass parameters in the order your function expects
            district_data = filter_state_district(data, selected_state.upper(), selected_district, selected_year)

            # --- 1. District Summary ---
            st.markdown(f"### 📋 Summary for {selected_district}, {selected_state}")
            st.dataframe(district_data, use_container_width=True, height=200)

            st.markdown("---")

            # --- 2. Crime Comparison Bar Chart ---
            crime_columns = ["MURDER", "RAPE", "KIDNAPPING & ABDUCTION",
                             "THEFT", "BURGLARY", "DOWRY DEATHS", "TOTAL IPC CRIMES"]

            # Defensive: keep only columns that actually exist in the dataframe
            available_crime_cols = [c for c in crime_columns if c in district_data.columns]
            if len(available_crime_cols) == 0:
                st.info("No matching crime columns available to plot for this district.")
            else:
                crime_sums = district_data[available_crime_cols].sum(numeric_only=True)

                st.markdown("### 📊 Major Crime Comparison")
                fig, ax = plt.subplots(figsize=(8, 5))

                bar_color = "#3498DB"
                crime_sums.plot(kind="bar", color=bar_color, ax=ax)

                # Add data labels on top of bars (works even with pandas plotting)
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9)

                ax.set_title(f"Major Crimes in {selected_district}", fontsize=14, fontweight='bold')
                ax.set_ylabel("Number of Cases", fontsize=11)
                ax.set_xlabel("Crime Type", fontsize=11)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_axisbelow(True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("👉 Please select a state and district to view data.")



    # ============================
    # SAFETY RATIO PAGE
    # ============================
    elif st.session_state.page == "SafetyRatio":
        st.markdown("## 🛡 Safety Ratio & Crime Share Analysis")
        st.button("⬅ Back to Home", on_click=lambda: go_to("Home")) 
        st.divider()

        states = get_states(data)
        selected_state = st.selectbox("Select State/UT", states)

        if selected_state:
            # --- 1. Safety Metric ---
            ratio = calculate_safety_ratio(data, selected_state)
            st.metric(f"Safety Ratio for {selected_state}", f"{ratio:.2f}%", 
                      help="Higher ratio means lower total crime contribution relative to the dataset total.")
            st.caption("Safety Ratio is calculated as inverse of total crime share.")
            
            st.divider()

            # --- 2. PIE CHART ---
            st.markdown("### 🌐 State's Contribution to Total Crime")
            
            total_crime = data["TOTAL IPC CRIMES"].sum()
            state_crime = data[data["STATE/UT"] == selected_state]["TOTAL IPC CRIMES"].sum()
            others = total_crime - state_crime
            
            if state_crime == 0 and others == 0:
                st.info("No crime data found to generate pie chart.")
            else:
                pie_data = pd.Series([state_crime, others],
                                     index=[f"{selected_state}'s Crime", "Other States' Crime"])
                
                colors = ['#3498DB', '#D5DBDB'] 
                explode_values = [0.03, 0] 
        
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90, 
                       colors=colors, explode=explode_values,
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1}, 
                       textprops={'fontsize': 10, 'color': 'black'})
                ax.set_title(f"Crime Share: {selected_state} vs Dataset Total", fontsize=12)
                st.pyplot(fig)
                plt.close(fig) # Resource Cleanup
        else:
            st.info("👉 Please select a state to calculate its safety ratio.")


    # ============================
    # COMPARE TWO STATES (ENHANCED)
    # ============================
    elif st.session_state.page == "Compare":
        st.markdown("## ⚖ State-to-State Crime Comparison")
        st.button("⬅ Back to Home", on_click=lambda: go_to("Home")) 
        st.divider()

        states = get_states(data)
        colA, colB = st.columns(2)
        with colA:
            state1 = st.selectbox("Select First State", states, key="s1")
        with colB:
            state2 = st.selectbox("Select Second State", states, key="s2")

        st.divider()

        if state1 != state2 and state1 and state2:
            
            # --- 1. Calculate Core Metrics ---
            crime1 = data[data["STATE/UT"] == state1]["TOTAL IPC CRIMES"].sum()
            crime2 = data[data["STATE/UT"] == state2]["TOTAL IPC CRIMES"].sum()
            ratio1 = calculate_safety_ratio(data, state1)
            ratio2 = calculate_safety_ratio(data, state2)
            
            # --- 2. Summary Metrics Table ---
            st.markdown(f"### 📋 Analytical Summary")
            
            # Calculate percentage difference for Total Crimes
            if crime1 != 0:
                diff_percent = (crime2 - crime1) / crime1 * 100
                diff_text = f"{'+' if diff_percent > 0 else ''}{diff_percent:.1f}%"
            elif crime2 > 0:
                diff_text = "N/A (State 1 Crime is Zero)"
            else:
                diff_text = "0.0%"

            # Determine safer state
            safer_state = state1 if ratio1 >= ratio2 else state2
            
            # Create a DataFrame for metric display
            metric_df = pd.DataFrame({
                "Metric": ["Total IPC Crimes", "Safety Ratio", "Overall Safer State"],
                state1: [f"{crime1:,}", f"{ratio1:.2f}%", ""],
                state2: [f"{crime2:,}", f"{ratio2:.2f}%", ""],
                "Comparison": [diff_text, f"{ratio2 - ratio1:+.2f}%", safer_state]
            }).set_index("Metric")

            st.table(metric_df)
            
            st.divider()

            # --- 3. Crime Composition Pie Charts ---
            st.markdown("### 🥧 Top Crime Composition Analysis")
            
            comp_col1, comp_col2 = st.columns(2)
            
            # Helper function to draw the pie chart
            def draw_composition_pie(state, column_obj):
                composition = get_top_crime_composition(data, state)
                
                if composition.empty or composition.sum() == 0:
                     column_obj.info(f"No crime data available for {state}")
                     return

                fig, ax = plt.subplots(figsize=(5, 5))
                
                colors = plt.colormaps.get_cmap('Spectral')(np.linspace(0, 1, len(composition)))
                
                ax.pie(composition, labels=composition.index, autopct="%1.1f%%", startangle=90, 
                       colors=colors,
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                       textprops={'fontsize': 9})
                ax.set_title(f"Composition in {state}", fontsize=11)
                column_obj.pyplot(fig)
                plt.close(fig) # Resource Cleanup

            with comp_col1:
                draw_composition_pie(state1, comp_col1)
                
            with comp_col2:
                draw_composition_pie(state2, comp_col2)

            st.divider()
            
            # --- 4. Auto-Generated Insight Text ---
            st.markdown("### ✨ Key Analytical Insight")
            
            # Find dominant crime categories for insight (Top 1)
            top_crime_s1 = get_top_crime_composition(data, state1, top_n=1).index[0] if not get_top_crime_composition(data, state1, top_n=1).empty and get_top_crime_composition(data, state1, top_n=1).sum() > 0 else "Unspecified Crimes"
            top_crime_s2 = get_top_crime_composition(data, state2, top_n=1).index[0] if not get_top_crime_composition(data, state2, top_n=1).empty and get_top_crime_composition(data, state2, top_n=1).sum() > 0 else "Unspecified Crimes"
            
            # Generate the insightful sentence
            insight = f"Based on the analysis, *{safer_state}* is the safer of the two states, showing a Safety Ratio difference of *{abs(ratio1 - ratio2):.2f} percentage points*."
            insight += f" The dominant crime category in *{state1}* is *{top_crime_s1}, while **{state2}* is primarily characterized by *{top_crime_s2}*."
            
            st.success(insight)
            
        else:
            st.warning("⚠ Please select two different states to start the comparison.")

        st.divider()


    # ============================
    # CRIME TRENDS
    # ============================
    elif st.session_state.page == "Trends":
        st.markdown("## 📈 Crime Trends Over Years")
        st.button("⬅ Back to Home", on_click=lambda: go_to("Home"))
        st.divider()

        states = get_states(data)
        selected_state = st.selectbox("Select State/UT", states)

        if selected_state:
            trend_data = data[data["STATE/UT"] == selected_state].groupby("YEAR")["TOTAL IPC CRIMES"].sum().reset_index()

            st.markdown(f"### Total IPC Crimes Trend in {selected_state}")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(trend_data["YEAR"], trend_data["TOTAL IPC CRIMES"], marker="o", color="#E67E22", linewidth=2)
            ax.set_xlabel("Year")
            ax.set_ylabel("Total Crimes")
            ax.grid(axis='y', linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) # Resource Cleanup
        else:
            st.info("👉 Please select a state to see its trend.")

        st.divider()


    # ============================
    # FUTURE PREDICTION
    # ============================
    elif st.session_state.page == "Predict":
        st.markdown("## 🔮 Future Crime Predictions")
        st.button("⬅ Back to Home", on_click=lambda: go_to("Home"))
        st.divider()

        states = get_states(data)
        selected_state = st.selectbox("Select State/UT", states)

        if selected_state:
            state_data = data[data["STATE/UT"] == selected_state]
            yearly_data = state_data.groupby("YEAR")["TOTAL IPC CRIMES"].sum().reset_index()

            if len(yearly_data) < 2:
                st.error(f"Cannot generate prediction for {selected_state}. Insufficient historical data.")
            else:
                X = yearly_data["YEAR"].values.reshape(-1, 1)
                y = yearly_data["TOTAL IPC CRIMES"].values

                model = LinearRegression()
                model.fit(X, y)

                last_year = X.flatten()[-1]
                future_years = np.arange(last_year + 1, last_year + 6).reshape(-1, 1)
                predictions = model.predict(future_years)

                future_df = pd.DataFrame({
                    "YEAR": future_years.flatten(),
                    "PREDICTED CRIMES": predictions.astype(int)
                })

                st.markdown(f"### 🔢 Predicted Crimes for Next 5 Years ({selected_state})")
                st.dataframe(future_df, width='stretch') 

                st.divider()
                
                # Plot
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(X.flatten(), y, label="Actual Crime Data", marker="o", color="#3498DB", linewidth=2)
                ax.plot(future_years.flatten(), predictions, label="Linear Prediction", marker="x", color="#E74C3C", linestyle="--", linewidth=2)
                ax.set_title(f"Crime Prediction for {selected_state}")
                ax.set_xlabel("Year")
                ax.set_ylabel("Total IPC Crimes")
                ax.legend()
                ax.grid(axis='y', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Resource Cleanup
        else:
            st.info("👉 Please select a state to generate future predictions.")

        st.divider()


