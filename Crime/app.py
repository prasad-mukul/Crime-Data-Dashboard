# app.py - cleaned and robust version
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

# Page config
st.set_page_config(page_title="Crime Visualization Dashboard", layout="centered")

# Load data
data = load_data()

# --- Basic defensive checks immediately after load ---
if data is None:
    st.error("Error loading data (load_data() returned None). Check backend.")
    st.stop()

# session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    st.session_state.page = page

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Home"

# --- Login/Register page ---
def login_page():
    st.markdown("<h1 style='text-align:center; color: #1f77b4;'>🕵 Crime Data Visualization Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Please log in or register to access the analytical features.</p>", unsafe_allow_html=True)
    st.markdown("---")

    login_tab, register_tab = st.tabs(["🔒 Login", "📝 Register"])

    with login_tab:
        st.subheader("Existing User Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Log In")
            if submitted:
                try:
                    if authenticate_user(login_username, login_password):
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.success(f"Welcome back, {login_username}!")
                        st.experimental_rerun()
                    else:
                        if is_username_registered(login_username):
                            st.error("Login failed: Incorrect password.")
                        else:
                            st.error("Login failed: This username is not registered.")
                except Exception as e:
                    st.error("Authentication error. Check backend.")
                    st.exception(e)

    with register_tab:
        st.subheader("New User Registration")
        with st.form("register_form"):
            reg_username = st.text_input("New Username", key="reg_user")
            reg_password = st.text_input("New Password", type="password", key="reg_pass")
            reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_pass")
            reg_submitted = st.form_submit_button("Register")
            if reg_submitted:
                if reg_password != reg_confirm_password:
                    st.error("Registration failed: Passwords do not match.")
                else:
                    try:
                        success, message = register_user(reg_username, reg_password)
                        if success:
                            st.success(message)
                        else:
                            st.error("Registration failed. " + message)
                    except Exception as e:
                        st.error("Registration error. Check backend.")
                        st.exception(e)

# If not logged in show login page
if not st.session_state.logged_in:
    login_page()
    st.stop()

# --- Main header after login ---
header_col1, header_col2 = st.columns([7, 2])
with header_col1:
    st.markdown(f"**Welcome, {st.session_state.username}!**")
with header_col2:
    if st.button("🔓 Logout"):
        logout()
        st.experimental_rerun()

st.markdown("<h1 style='text-align:center; color: #1f77b4;'>🕵 Crime Data Visualization Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Explore crime statistics, safety insights, and predictive trends across Indian states (2013 data)</p>", unsafe_allow_html=True)
st.markdown("---")

# Defensive: make sure dataset has STATE/UT and DISTRICT
if data.empty:
    st.error("Data is empty. Ensure crime.csv is present and formatted correctly.")
    st.stop()
if "STATE/UT" not in data.columns:
    st.error("Dataset missing 'STATE/UT' column. Please verify CSV header.")
    st.stop()

# Helper: present states in Title Case to the user but map back to uppercase for filtering
raw_states = get_states(data)  # backend returns already normalized items (upper)
display_states = [s.title() for s in raw_states]

# Main navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# HOME
if st.session_state.page == "Home":
    st.markdown("<h3 style='text-align:center; color: #34495E;'>Choose an Option to Explore</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    if col1.button("🔍 State Crime Search"):
        go_to("CrimeSearch")
    if col2.button("🛡 Safety Ratio"):
        go_to("SafetyRatio")
    if col3.button("⚖ Compare Two States"):
        go_to("Compare")

    col4, col5 = st.columns(2)
    if col4.button("📈 Crime Trends"):
        go_to("Trends")
    if col5.button("🔮 Future Prediction"):
        go_to("Predict")

    st.markdown("---")
    st.caption("Tip: Click any button to explore detailed insights for crime data visualization.")

# -------------------------
# Crime Search Page
# -------------------------
elif st.session_state.page == "CrimeSearch":
    st.markdown("## 🔍 State Crime Search")
    if st.button("⬅ Back to Home"):
        go_to("Home")

    # Year & state selectors
    years = get_years(data)
    selected_year = st.selectbox("Select Year", years) if years else None

    selected_state_display = st.selectbox("Select State/UT", ["Select a state"] + display_states)
    selected_state = None
    if selected_state_display and selected_state_display != "Select a state":
        selected_state = selected_state_display.upper()

    selected_district = None

    # Filter data by selected_state and year (use uppercase values in dataset)
    if selected_state:
        # dataset STATE/UT is normalized in backend.load_data() to uppercase
        filtered = data[data["STATE/UT"] == selected_state]
        if selected_year is not None:
            if "YEAR" in filtered.columns:
                filtered = filtered[filtered["YEAR"] == selected_year]

        # Build clean district list (title-case for display)
        if "DISTRICT" in filtered.columns and not filtered.empty:
            raw_districts = filtered["DISTRICT"].dropna().unique().tolist()
            # Keep only meaningful text entries (avoid purely numeric / blank)
            clean_districts = [d for d in raw_districts if isinstance(d, str) and d.strip() and not d.strip().isdigit()]
            clean_districts_sorted = sorted(clean_districts)
            display_districts = [d.title() for d in clean_districts_sorted]
            if display_districts:
                sel_d_display = st.selectbox("Select District", ["Select a district"] + display_districts)
                if sel_d_display and sel_d_display != "Select a district":
                    selected_district = sel_d_display.upper()
            else:
                st.warning(f"⚠️ No valid districts found for {selected_state_display} in {selected_year}.")
        else:
            st.warning(f"⚠️ No district column or no rows for {selected_state_display} in the data.")
    else:
        st.info("👉 Select a state to load districts.")

    st.markdown("---")

    # Proceed to show district-level data if both selected
    if selected_state and selected_district:
        # Use backend filter function; backend expects state/district in same case as dataset (uppercase)
        district_data = filter_state_district(data, selected_state, selected_district, selected_year)

        # Show summary
        st.markdown(f"### 📋 Summary for {selected_district.title()}, {selected_state_display}")
        st.dataframe(district_data, use_container_width=True, height=250)

        st.markdown("---")

        # Crime comparison columns (defensive)
        crime_columns = ["MURDER", "RAPE", "KIDNAPPING & ABDUCTION", "THEFT", "BURGLARY", "DOWRY DEATHS", "TOTAL IPC CRIMES"]
        available = [c for c in crime_columns if c in district_data.columns]

        if available:
            crime_sums = district_data[available].sum(numeric_only=True)
            st.markdown("### 📊 Major Crime Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            crime_sums.plot(kind="bar", ax=ax)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.0f")
            ax.set_title(f"Major Crimes in {selected_district.title()}")
            ax.set_ylabel("Number of Cases")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No crime columns available to plot for this selection.")
    else:
        st.info("👉 Please select a state and district (and optionally a year) to view data.")

# -------------------------
# Safety Ratio Page
# -------------------------
elif st.session_state.page == "SafetyRatio":
    st.markdown("## 🛡 Safety Ratio & Crime Share Analysis")
    if st.button("⬅ Back to Home"):
        go_to("Home")
    st.markdown("---")

    states_list = get_states(data)
    display_states = [s.title() for s in states_list]
    selected_state_display = st.selectbox("Select State/UT", ["Select a state"] + display_states)
    if selected_state_display and selected_state_display != "Select a state":
        selected_state = selected_state_display.upper()
    else:
        selected_state = None

    if selected_state:
        ratio = calculate_safety_ratio(data, selected_state)
        st.metric(f"Safety Ratio for {selected_state_display}", f"{ratio:.2f}%")
        st.caption("Safety Ratio (higher = safer)")

        # Pie: state vs others
        total_crime = data["TOTAL IPC CRIMES"].sum() if "TOTAL IPC CRIMES" in data.columns else 0
        state_crime = data[data["STATE/UT"] == selected_state]["TOTAL IPC CRIMES"].sum() if "TOTAL IPC CRIMES" in data.columns else 0
        others = total_crime - state_crime
        if total_crime == 0:
            st.info("No crime totals available to display.")
        else:
            pie_data = pd.Series([state_crime, others], index=[f"{selected_state_display}'s Crime", "Other States' Crime"])
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Crime Share: {selected_state_display}")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("👉 Please select a state to calculate its safety ratio.")

# -------------------------
# Compare Page
# -------------------------
elif st.session_state.page == "Compare":
    st.markdown("## ⚖ State-to-State Crime Comparison")
    if st.button("⬅ Back to Home"):
        go_to("Home")
    st.markdown("---")

    states_list = get_states(data)
    display_states = [s.title() for s in states_list]
    colA, colB = st.columns(2)
    with colA:
        s1_display = st.selectbox("Select First State", ["Select a state"] + display_states, key="s1")
    with colB:
        s2_display = st.selectbox("Select Second State", ["Select a state"] + display_states, key="s2")

    if s1_display and s2_display and s1_display != "Select a state" and s2_display != "Select a state" and s1_display != s2_display:
        s1 = s1_display.upper()
        s2 = s2_display.upper()

        crime1 = data[data["STATE/UT"] == s1]["TOTAL IPC CRIMES"].sum() if "TOTAL IPC CRIMES" in data.columns else 0
        crime2 = data[data["STATE/UT"] == s2]["TOTAL IPC CRIMES"].sum() if "TOTAL IPC CRIMES" in data.columns else 0
        ratio1 = calculate_safety_ratio(data, s1)
        ratio2 = calculate_safety_ratio(data, s2)

        # Summary table
        safer_state = s1_display if ratio1 >= ratio2 else s2_display
        diff_percent = "N/A"
        if crime1:
            diff_percent = f"{(crime2 - crime1) / crime1 * 100:+.1f}%"
        metric_df = pd.DataFrame({
            "Metric": ["Total IPC Crimes", "Safety Ratio", "Overall Safer State"],
            s1_display: [f"{crime1:,}", f"{ratio1:.2f}%", ""],
            s2_display: [f"{crime2:,}", f"{ratio2:.2f}%", ""],
            "Comparison": [diff_percent, f"{ratio2 - ratio1:+.2f}%", safer_state]
        }).set_index("Metric")
        st.table(metric_df)

        # Composition pies
        comp_col1, comp_col2 = st.columns(2)
        def draw_comp(st_name_display, col_obj):
            comp = get_top_crime_composition(data, st_name_display.upper())
            if comp.empty or comp.sum() == 0:
                col_obj.info(f"No crime composition data for {st_name_display}")
                return
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie(comp, labels=comp.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Composition in {st_name_display}")
            col_obj.pyplot(fig)
            plt.close(fig)

        with comp_col1:
            draw_comp(s1_display, comp_col1)
        with comp_col2:
            draw_comp(s2_display, comp_col2)
    else:
        st.info("👉 Select two different states to compare.")

# -------------------------
# Trends Page
# -------------------------
elif st.session_state.page == "Trends":
    st.markdown("## 📈 Crime Trends Over Years")
    if st.button("⬅ Back to Home"):
        go_to("Home")
    st.markdown("---")

    display_states = [s.title() for s in get_states(data)]
    sel_display = st.selectbox("Select State/UT", ["Select a state"] + display_states)
    if sel_display and sel_display != "Select a state":
        sel_state = sel_display.upper()
        if "YEAR" in data.columns and "TOTAL IPC CRIMES" in data.columns:
            trend = data[data["STATE/UT"] == sel_state].groupby("YEAR")["TOTAL IPC CRIMES"].sum().reset_index()
            if trend.empty:
                st.info("No trend data available for this state.")
            else:
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(trend["YEAR"], trend["TOTAL IPC CRIMES"], marker="o")
                ax.set_xlabel("Year"); ax.set_ylabel("Total IPC Crimes")
                ax.grid(axis='y', linestyle='--')
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Required columns for trend not available.")
    else:
        st.info("👉 Please select a state to view trends.")

# -------------------------
# Predict Page
# -------------------------
elif st.session_state.page == "Predict":
    st.markdown("## 🔮 Future Crime Predictions")
    if st.button("⬅ Back to Home"):
        go_to("Home")
    st.markdown("---")

    display_states = [s.title() for s in get_states(data)]
    sel_display = st.selectbox("Select State/UT", ["Select a state"] + display_states)
    if sel_display and sel_display != "Select a state":
        sel_state = sel_display.upper()
        if "YEAR" in data.columns and "TOTAL IPC CRIMES" in data.columns:
            yearly = data[data["STATE/UT"] == sel_state].groupby("YEAR")["TOTAL IPC CRIMES"].sum().reset_index()
            if len(yearly) < 2:
                st.error("Insufficient historical data for prediction.")
            else:
                X = yearly["YEAR"].values.reshape(-1,1)
                y = yearly["TOTAL IPC CRIMES"].values
                model = LinearRegression()
                model.fit(X,y)
                last = int(X.flatten()[-1])
                future_years = np.arange(last+1, last+6).reshape(-1,1)
                preds = model.predict(future_years).astype(int)
                future_df = pd.DataFrame({"YEAR": future_years.flatten(), "PREDICTED CRIMES": preds})
                st.dataframe(future_df, use_container_width=True)
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(X.flatten(), y, marker="o", label="Actual")
                ax.plot(future_years.flatten(), preds, marker="x", linestyle="--", label="Predicted")
                ax.set_xlabel("Year"); ax.set_ylabel("Total IPC Crimes")
                ax.legend(); ax.grid(axis='y', linestyle='--')
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Required columns for prediction not available.")
    else:
        st.info("👉 Please select a state to generate predictions.")

# End of file
