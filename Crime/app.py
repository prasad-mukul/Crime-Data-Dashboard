import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
# This will import from the "backend_fixed.py" file you have in the Canvas
from backend_fixed import (
    load_data, get_years, get_states, filter_state_district,
    calculate_safety_ratio, get_top_crime_composition,
    authenticate_user, register_user, is_username_registered
)

# Set page config once at the very top of the script
st.set_page_config(page_title="Crime Visualization Dashboard", layout="centered")

# --- Initial Data Load & Caching ---
# By decorating with @st.cache_data, we ensure the data is loaded only once,
# which significantly speeds up the app after the first run.
@st.cache_data
def cached_load_data():
    return load_data()

data = cached_load_data()

# --- INITIALIZE SESSION STATE ---
# This is the correct way to initialize session state variables.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    """Callback function to change the page in session state."""
    st.session_state.page = page

def logout():
    """Resets session state to log the user out and forces a rerun."""
    st.session_state.logged_in = False
    st.session_state.page = "Home"
    st.session_state.username = None
    st.rerun()

# ============================
# LOGIN/REGISTRATION PAGE
# ============================
def login_page():
    st.markdown("<h1 style='text-align:center; color: #1f77b4;'>🕵 Crime Data Visualization Dashboard</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; font-size: 1.1em;'>Please log in or register to access the full analytical features.</p>", unsafe_allow_html=True)
    st.divider()

    st.info("""
        *Note for Judges & Users:* The user registration is for demonstration purposes.
        If the app has been inactive, newly registered accounts may be cleared.
        Please use the default credentials judge / hackathon2024 for guaranteed access.
    """)

    login_tab, register_tab = st.tabs(["🔒 Login", "📝 Register"])

    with login_tab:
        st.subheader("Existing User Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            login_submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

            if login_submitted:
                if authenticate_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success(f"Welcome back, {login_username}!")
                    st.rerun()
                else:
                    if is_username_registered(login_username):
                        st.error("Login failed: Incorrect password.")
                    else:
                        st.error("Login failed: This username is not registered.")

    with register_tab:
        st.subheader("New User Registration")
        with st.form("register_form"):
            reg_username = st.text_input("New Username", key="reg_user")
            reg_password = st.text_input("New Password", type="password", key="reg_pass")
            reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_pass")
            reg_submitted = st.form_submit_button("Register", use_container_width=True)

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
# MAIN APPLICATION ROUTER
# =================================================================

if not st.session_state.logged_in:
    login_page()
else:
    # --- GLOBAL HEADER & LOGOUT BUTTON ---
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"*Welcome, {st.session_state.username}!*")
    with header_col2:
        st.button("🔓 Logout", on_click=logout, type='primary', use_container_width=True)

    st.markdown("<h1 style='text-align:center; color: #1f77b4;'>🕵 Crime Data Visualization Dashboard</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; font-size: 1.1em;'>Explore crime statistics, safety insights, and predictive trends across Indian states (Data from 2001-2013)</p>", unsafe_allow_html=True)
    st.divider()


    # ============================
    # HOME PAGE (Navigation Hub)
    # ============================
    if st.session_state.page == "Home":
        st.markdown("<h3 style='text-align:center; color: #34495E;'>Choose an Option to Explore</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.button("🔍 State Crime Search", use_container_width=True, on_click=lambda: go_to("CrimeSearch"))
        with col2:
            st.button("🛡 Safety Ratio", use_container_width=True, on_click=lambda: go_to("SafetyRatio"))
        with col3:
            st.button("⚖ Compare Two States", use_container_width=True, on_click=lambda: go_to("Compare"))

        st.markdown("")
        col4, col5 = st.columns(2, gap="large")
        with col4:
            st.button("📈 Crime Trends", use_container_width=True, on_click=lambda: go_to("Trends"))
        with col5:
            st.button("🔮 Future Prediction", use_container_width=True, on_click=lambda: go_to("Predict"))

        st.markdown("---")
        st.caption("Tip: Click any button to explore detailed insights for crime data visualization.")


    # ============================
    # STATE CRIME SEARCH
    # ============================
    elif st.session_state.page == "CrimeSearch":
        st.markdown("## 🔍 State Crime Search")
        st.button("⬅ Back to Home", on_click=lambda: go_to("Home"))

        # --- Filters ---
        years = get_years(data)
        states = get_states(data)

        selected_year = st.selectbox("Select Year", years) if years else None
        selected_state = st.selectbox("Select State/UT", states) if states else None

        if selected_state:
            state_data_filtered = data[data["STATE/UT"] == selected_state]
            selected_district = None  # Initialize to None

            if "DISTRICT" in state_data_filtered.columns:
                districts = sorted(state_data_filtered["DISTRICT"].unique())

                # FIX: Only show the district dropdown if there are districts.
                # This prevents an error if the list of districts is empty.
                if districts:
                    selected_district = st.selectbox("Select District", districts)
                else:
                    st.warning(f"No specific district data found for {selected_state}.")
            else:
                st.warning("⚠ 'DISTRICT' column not found in the data for this state.")

            st.divider()

            if selected_district:
                district_data = filter_state_district(data, selected_state, selected_district, selected_year)

                # --- 1. District Summary ---
                st.markdown(f"### 📋 Summary for {selected_district}, {selected_state} ({selected_year})")
                st.dataframe(district_data, use_container_width=True, height=80)
                st.divider()

                # --- 2. Crime Comparison Bar Chart ---
                crime_columns = [
                    "MURDER", "RAPE", "KIDNAPPING & ABDUCTION",
                    "THEFT", "BURGLARY", "DOWRY DEATHS", "TOTAL IPC CRIMES"
                ]
                display_cols = [col for col in crime_columns if col in district_data.columns]
                crime_sums = district_data[display_cols].sum(numeric_only=True)

                st.markdown("### 📊 Major Crime Comparison")
                fig, ax = plt.subplots(figsize=(8, 5))
                crime_sums.plot(kind="bar", color="#3498DB", ax=ax)

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
                plt.close(fig) # Resource Cleanup
        else:
            st.info("👉 Please select a state to begin.")


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
            st.metric(
                f"Safety Ratio for {selected_state}",
                f"{ratio:.2f}%",
                help="Higher ratio means lower total crime contribution relative to the dataset total."
            )
            st.caption("Safety Ratio is calculated as (1 - [State's Crime / Total Crime]) * 100.")
            st.divider()

            # --- 2. PIE CHART ---
            st.markdown("### 🌐 State's Contribution to Total Crime")
            total_crime = data["TOTAL IPC CRIMES"].sum()
            state_crime = data[data["STATE/UT"] == selected_state]["TOTAL IPC CRIMES"].sum()

            if total_crime > 0:
                others_crime = total_crime - state_crime
                pie_data = pd.Series([state_crime, others_crime], index=[f"{selected_state}", "All Other States"])

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(
                    pie_data,
                    labels=pie_data.index,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=['#3498DB', '#D5DBDB'],
                    explode=[0.03, 0],
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                    textprops={'fontsize': 10, 'color': 'black'}
                )
                ax.set_title(f"Crime Share: {selected_state} vs Dataset Total", fontsize=12)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No crime data available to generate a chart.")
        else:
            st.info("👉 Please select a state to calculate its safety ratio.")


    # ============================
    # COMPARE TWO STATES
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
            state2_options = [s for s in states if s != state1]
            state2 = st.selectbox("Select Second State", state2_options, key="s2")

        st.divider()

        if state1 and state2 and state1 != state2:
            # --- 1. Calculate Core Metrics ---
            crime1 = data[data["STATE/UT"] == state1]["TOTAL IPC CRIMES"].sum()
            crime2 = data[data["STATE/UT"] == state2]["TOTAL IPC CRIMES"].sum()
            ratio1 = calculate_safety_ratio(data, state1)
            ratio2 = calculate_safety_ratio(data, state2)

            # --- 2. Summary Metrics Table ---
            st.markdown("### 📋 Analytical Summary")
            if crime1 > 0:
                diff_percent = ((crime2 - crime1) / crime1) * 100
                diff_text = f"{diff_percent:+.1f}%"
            else:
                diff_text = "N/A" # Avoid division by zero

            safer_state = state1 if ratio1 >= ratio2 else state2
            
            metric_data = {
                "Metric": ["Total IPC Crimes", "Safety Ratio", "Overall Safer State"],
                 state1: [f"{crime1:,}", f"{ratio1:.2f}%", "—"],
                 state2: [f"{crime2:,}", f"{ratio2:.2f}%", "—"],
                 "Comparison": [diff_text, f"{ratio2 - ratio1:+.2f} pts", safer_state]
            }
            metric_df = pd.DataFrame(metric_data).set_index("Metric")
            st.table(metric_df)
            st.divider()

            # --- 3. Crime Composition Pie Charts ---
            st.markdown("### 🥧 Top Crime Composition Analysis")
            comp_col1, comp_col2 = st.columns(2)

            def draw_composition_pie(state, column_obj):
                composition = get_top_crime_composition(data, state)
                if composition.empty or composition.sum() == 0:
                    column_obj.info(f"No specific crime data for {state}")
                    return
                fig, ax = plt.subplots(figsize=(5, 5))
                colors = plt.colormaps.get_cmap('Spectral')(np.linspace(0, 1, len(composition)))
                ax.pie(
                    composition, labels=composition.index, autopct="%1.1f%%", startangle=90,
                    colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                    textprops={'fontsize': 9}
                )
                ax.set_title(f"Composition in {state}", fontsize=11)
                column_obj.pyplot(fig)
                plt.close(fig)

            with comp_col1:
                draw_composition_pie(state1, comp_col1)
            with comp_col2:
                draw_composition_pie(state2, comp_col2)
            st.divider()
            
            # --- 4. Auto-Generated Insight Text ---
            st.markdown("### ✨ Key Analytical Insight")
            top_crime_s1 = get_top_crime_composition(data, state1, top_n=1).index[0]
            top_crime_s2 = get_top_crime_composition(data, state2, top_n=1).index[0]
            insight = (
                f"Based on the analysis, *{safer_state}* appears safer, with a Safety Ratio "
                f"{abs(ratio1 - ratio2):.2f} points** higher than the other. The dominant crime "
                f"in *{state1}* is *{top_crime_s1}, while **{state2}'s is **{top_crime_s2}*."
            )
            st.success(insight)
        else:
            st.warning("⚠ Please select two different states to start the comparison.")


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
            trend_data = data[data["STATE/UT"] == selected_state].groupby("YEAR")["TOTAL IPC CRIMES"].sum()
            if trend_data.empty:
                st.warning(f"No yearly data available for {selected_state}.")
            else:
                st.markdown(f"### Total IPC Crimes Trend in {selected_state}")
                st.line_chart(trend_data)
        else:
            st.info("👉 Please select a state to see its trend.")


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
            yearly_data = data[data["STATE/UT"] == selected_state].groupby("YEAR")["TOTAL IPC CRIMES"].sum().reset_index()

            if len(yearly_data) < 2:
                st.error(f"Cannot generate prediction for {selected_state}. Insufficient historical data (need at least 2 years).")
            else:
                X = yearly_data["YEAR"].values.reshape(-1, 1)
                y = yearly_data["TOTAL IPC CRIMES"].values

                model = LinearRegression()
                model.fit(X, y)

                last_year = X.flatten()[-1]
                future_years = np.arange(last_year + 1, last_year + 6).reshape(-1, 1)
                predictions = model.predict(future_years)
                predictions[predictions < 0] = 0

                future_df = pd.DataFrame({
                    "YEAR": future_years.flatten(),
                    "PREDICTED CRIMES": predictions.astype(int)
                })

                st.markdown(f"### 🔢 Predicted Crimes for Next 5 Years ({selected_state})")
                st.dataframe(future_df, use_container_width=True)
                st.divider()

                # Plot
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(X.flatten(), y, label="Actual Crime Data", marker="o", color="#3498DB", linewidth=2)
                ax.plot(future_years.flatten(), predictions, label="Linear Prediction", marker="x", color="#E74C3C", linestyle="--", linewidth=2)
                ax.set_title(f"Crime Prediction for {selected_state}")
                ax.set_xlabel("Year")
                ax.set_ylabel("Total IPC Crimes")
                ax.legend()
                ax.grid(axis='y', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("👉 Please select a state to generate future predictions.")
