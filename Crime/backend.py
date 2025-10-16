import pandas as pd
import numpy as np

# Hardcoded credentials for authentication
USER_CREDENTIALS = {
    "judge": "hackathon2024",
    "user1": "pass123"
}

# NOTE: This dictionary is updated dynamically by register_user, 
# but starts with these initial values.

def load_data():
    """
    Loads data, cleans column names, ensures numerical columns are numeric, 
    and **normalizes STATE/UT and DISTRICT names** to ensure consistent filtering.
    """
    try:
        import os
        data_path = os.path.join(os.path.dirname(__file__), "crime.csv")
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: crime.csv not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
        
    data.columns = data.columns.str.strip().str.upper()
    
    # --- Data Cleaning and Normalization ---
    
    # 1. Normalize State/UT and District Names to UPPECRCASE for consistent filtering
    if 'STATE/UT' in data.columns:
        data['STATE/UT'] = data['STATE/UT'].astype(str).str.strip().str.upper()
    if 'DISTRICT' in data.columns:
        data['DISTRICT'] = data['DISTRICT'].astype(str).str.strip().str.upper()
        
    # 2. Ensure crime columns are numeric
    crime_cols = [col for col in data.columns if col not in ['STATE/UT', 'DISTRICT', 'YEAR']]
    for col in crime_cols:
        # Coerce non-numeric values to NaN, fill NaN with 0, and convert to integer
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
        
    return data

# --- Authentication Functions ---

def authenticate_user(username, password):
    """Checks if the provided credentials are valid."""
    return USER_CREDENTIALS.get(username) == password

def is_username_registered(username):
    """Checks if a username exists in the credentials dictionary."""
    return username in USER_CREDENTIALS

def register_user(username, password):
    """Registers a new user and updates the in-memory credential dictionary."""
    if not username or not password:
        return False, "Username and password cannot be empty."
    if username in USER_CREDENTIALS:
        return False, "Username already exists. Please choose another."
    
    USER_CREDENTIALS[username] = password
    return True, f"User '{username}' registered successfully! You can now log in."

# --- Data Utility Functions ---

def get_states(data):
    """Returns a sorted list of unique states."""
    if 'STATE/UT' in data.columns:
        return sorted(data["STATE/UT"].unique())
    return []

def get_years(data):
    """Returns a sorted list of unique years."""
    if "YEAR" in data.columns:
        return sorted(data["YEAR"].unique())
    return []

def filter_state_district(data, state, district=None, year=None):
    """Filters data based on state, district, and year."""
    if data.empty:
        return pd.DataFrame()
        
    # Apply filtering sequentially
    filtered_data = data.copy()
    
    if year is not None and "YEAR" in filtered_data.columns:
        filtered_data = filtered_data[filtered_data["YEAR"] == year]
        
    if state is not None and 'STATE/UT' in filtered_data.columns:
        # Note: State is already normalized to UPPERCASE by load_data
        filtered_data = filtered_data[filtered_data["STATE/UT"] == state]
        
    if district and 'DISTRICT' in filtered_data.columns:
        # Note: District is already normalized to UPPERCASE by load_data
        filtered_data = filtered_data[filtered_data["DISTRICT"] == district]
        
    return filtered_data

def calculate_safety_ratio(data, selected_state):
    """Calculates the safety ratio for a given state."""
    if data.empty or "TOTAL IPC CRIMES" not in data.columns:
        return 0.0
        
    total_state_crimes = data[data["STATE/UT"] == selected_state]["TOTAL IPC CRIMES"].sum()
    total_crimes = data["TOTAL IPC CRIMES"].sum()
    
    if total_crimes == 0:
        return 100.0 # Perfectly safe if no crime recorded
        
    # Ratio is calculated as inverse of total crime contribution
    return (1 - (total_state_crimes / total_crimes)) * 100

def get_top_crime_composition(data, state, top_n=5):
    """
    Calculates the top N crimes and groups the rest into 'OTHER IPC CRIMES' 
    for composition analysis.
    """
    crime_cols = [
        "MURDER", "RAPE", "KIDNAPPING & ABDUCTION", 
        "THEFT", "BURGLARY", "DOWRY DEATHS", 
        "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY", 
        "CRUELTY BY HUSBAND OR HIS RELATIVES", "ARSON"
    ]
    
    state_data = data[data["STATE/UT"] == state].copy()
    
    if state_data.empty:
        return pd.Series()
        
    # Sum up all available crime columns for the state
    crime_sums = state_data[crime_cols].sum(numeric_only=True)
    
    # Filter out columns with zero total crime and sort
    crime_sums = crime_sums[crime_sums > 0].sort_values(ascending=False)
    
    if crime_sums.empty:
        return pd.Series({"NO MAJOR CRIMES": 1})

    # Isolate top crimes and calculate the 'other' category
    top_crimes = crime_sums.head(top_n)
    other_sum = crime_sums.iloc[top_n:].sum()
    
    if other_sum > 0:
        composition = pd.concat([top_crimes, pd.Series([other_sum], index=['OTHER IPC CRIMES'])])
    else:
        composition = top_crimes

    return composition

