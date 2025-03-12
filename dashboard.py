import streamlit as st
import json
import os
from datetime import datetime, date
import pandas as pd
import plotly.express as px
import random
import time

# Constants
DATA_FILE = "atm_dataFile.json"
PREDICTIONS_FILE = "predictions.csv"

def read_data():
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        data = {}

    defaults = {
        "total_amount": 0,
        "transactions": [],
        "auto_refill_amount": 50000,
        "cash_trend": [],
        "max_withdrawal_amount": 20000,
        "refill_threshold": 10000,
        "withdrawal_history": [],
        "last_updated_date": None,
        "daily_withdrawals": 0,
        "daily_withdrawal_amount": 0,
    }

    updated = False
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
            updated = True

    if updated:
        write_data(data)

    return data

def write_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def simulate_random_withdrawals(num_withdrawals):
    data = read_data()
    max_withdrawal_amount = data["max_withdrawal_amount"]

    with st.spinner("⏳ Simulating withdrawals... Please wait."):
        for _ in range(num_withdrawals):
            withdrawal_amount = random.randrange(500, max_withdrawal_amount + 1, 500)

            if withdrawal_amount <= data["total_amount"]:
                data["total_amount"] -= withdrawal_amount
                data["transactions"].append({
                    "account_number": generate_random_account_number(),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "amount": -withdrawal_amount,
                    "type": "Withdrawal"
                })
                data["withdrawal_history"].append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": withdrawal_amount
                })

            # Move auto-refill check **AFTER** the withdrawal
            if data["total_amount"] <= data["refill_threshold"]:
                data["total_amount"] += data["auto_refill_amount"]
                data["transactions"].append({
                    "account_number": "SYSTEM",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "amount": data['auto_refill_amount'],
                    "type": "Auto Refill"
                })
            # Introduce random delay between 1 to 3 seconds
            time.sleep(random.uniform(1, 3))

    write_data(data)
    st.session_state.data = data  # Update session state
    st.success(f"✅ {num_withdrawals} random withdrawals simulated successfully!")
    st.rerun()

def generate_random_account_number(length=10):
    return ''.join(str(random.randint(0, 9)) for _ in range(length))

def get_predicted_withdrawals(selected_date):
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        if not isinstance(selected_date, date):
            selected_date = selected_date.date()

        predicted_withdrawals = df[df['Date'] == selected_date]['Predicted Withdrawals'].values[0]
        return predicted_withdrawals
    except (FileNotFoundError, IndexError) as e:
        st.warning(f"Predicted withdrawals data not found for this date. Using a default value of 0. Error: {e}")
        return 0

st.set_page_config(layout="wide")

# Initialize session state variables
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "thread_running" not in st.session_state:
    st.session_state.thread_running = False
if "data" not in st.session_state:
    st.session_state.data = {
        "total_amount": 0,
        "transactions": [],
        "auto_refill_amount": 50000,
        "cash_trend": [],
        "max_withdrawal_amount": 20000,
        "refill_threshold": 10000,
        "withdrawal_history": [],
        "last_updated_date": None,
        "daily_withdrawals": 0,
        "daily_withdrawal_amount": 0,
    }
    if not os.path.exists(DATA_FILE):
        write_data(st.session_state.data)

st.title("ATM Cash Replenishment System")

tab_admin, tab_atm, tab_settings = st.tabs(["Admin Dashboard", "ATM Panel", "Settings"])

with tab_admin:
    st.header("Admin Dashboard")

    # Use current date
    selected_date = date.today()
    today_date = selected_date.strftime("%Y-%m-%d")

    # Fetch predicted withdrawals from CSV
    initial_amount = get_predicted_withdrawals(selected_date)

    # Initialize data if it hasn't been initialized, or it's a new day
    data = read_data()
    
    if data.get("last_updated_date") != today_date:  # Check for a new day
        current_balance = data["total_amount"]
        
        # Determine adjustment needed
        adjustment = initial_amount - current_balance
        if adjustment > 0:
            # If balance is too low, add money to reach predicted amount
            data["total_amount"] += adjustment
            transaction_type = "Daily Adjustment (Added)"
        elif adjustment < 0:
            # If balance is too high, remove money to match prediction
            data["total_amount"] += adjustment  # This subtracts because adjustment is negative
            transaction_type = "Daily Adjustment (Removed)"
        else:
            # If balance is already correct, no adjustment
            transaction_type = None

        # Record adjustment if needed
        if transaction_type:
            data["transactions"].append({
                "account_number": "SYSTEM",
                "date": today_date,
                "time": datetime.now().strftime("%H:%M:%S"),
                "amount": adjustment,
                "type": transaction_type
            })
        
        # Reset daily counters
        data["daily_withdrawals"] = 0
        data["daily_withdrawal_amount"] = 0
        data["last_updated_date"] = today_date
        
        # Save updated data
        write_data(data)

    st.session_state.data = data  # Ensure session state is updated

    # Correct calculation of ATM balance
    atm_balance = data["total_amount"]  # Use the actual stored amount

    # Function to calculate 30% threshold
    def calculate_low_balance_threshold(initial_amount):
        return 0.3 * initial_amount

    # Calculate the low balance threshold
    low_balance_threshold = calculate_low_balance_threshold(initial_amount)

    # Determine the color of the ATM balance based on the threshold
    atm_balance_color = "green" if data['total_amount'] > low_balance_threshold else "red"
    
    # Alert for low cash reserve
    low_cash_alert = "" if data['total_amount'] > low_balance_threshold else "<⚠️ Low Cash Reserve!>"
    
    atm_balance_style = f"""
        <div style="background-color:lightblue;padding:15px;border-radius:10px;text-align:left;display:flex;align-items:center;">
            <b style="color:black;font-size:20px;">ATM Balance</b>
            <span style="color:{atm_balance_color};font-size:25px;font-weight:bold; margin-left: 30px;">
                ₦{atm_balance:,.0f}
            </span>
            <span style="color:red; font-size:18px; font-weight:bold; margin-left: 50px;">{low_cash_alert}</span>
        </div>
    """

    st.markdown(atm_balance_style, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Withdrawals", value=len(
            [t for t in data['transactions'] if t['date'] == today_date and t['type'] == 'Withdrawal']))

    with col2:
        total_withdrawals = sum(-t["amount"] for t in data["transactions"]
                                if t["date"] == today_date and t["type"] == "Withdrawal")
        st.metric(label="Amount Withdrawn", value=f"₦{total_withdrawals:,.0f}")

    with col3:
        total_replenished_today = sum(t["amount"] for t in data["transactions"]
                                      if t["date"] == today_date and t["type"] == "Auto Refill")
        st.metric(label="Total Replenished", value=f"₦{total_replenished_today:,.0f}")

    with col4:
        user_deposits = sum(t["amount"] for t in data["transactions"]
                            if t["amount"] > 0 and t["account_number"] != "SYSTEM")
        total_deposited = initial_amount + user_deposits + total_replenished_today
        st.metric(label="Total Deposited", value=f"₦{total_deposited:,.0f}")

    # Define 30% low balance threshold
    low_balance_threshold = 0.3 * atm_balance
    
    

    # Determine ATM balance color
    atm_balance_color = "green" if atm_balance > low_balance_threshold else "red"
    atm_balance_style = f"""
        <div style="background-color:lightblue;padding:15px;border-radius:10px;text-align:left;">
            <b style="color:black;font-size:20px;">ATM Balance</b>
            <span style="color:{atm_balance_color};font-size:30px;font-weight:bold;">₦{atm_balance:,.0f}</span>
        </div>
    """

    # Layout for transaction records and charts
    col_df, col_chart, col_visual = st.columns([3, 3, 3])

    with col_df:
        st.subheader("_Transaction Record_")
        df = pd.DataFrame(data["transactions"])
        st.dataframe(df, use_container_width=True)

    with col_chart:
        st.subheader("_Real-time Withdrawals_")
        if data["withdrawal_history"]:
            df_withdrawals = pd.DataFrame(data["withdrawal_history"])
            df_withdrawals["date"] = pd.to_datetime(df_withdrawals["date"])
            fig_withdrawals = px.line(df_withdrawals, x="date", y="amount")
            st.plotly_chart(fig_withdrawals, use_container_width=True, key="realtime_withdrawals_chart")
        else:
            st.write("No withdrawal history available.")

    with col_visual:
        st.subheader("_Cash Reserve Visualization_")
        balance_percentage = min(1.0, atm_balance / initial_amount) if initial_amount != 0 else 0
        balance_color = "green" if balance_percentage >= 0.8 else ("orange" if balance_percentage > 0.2 else "red")

        chart_data = pd.DataFrame({
            "Category": ["Initial Deposit", "Current Balance"],
            "Amount": [initial_amount, atm_balance],
            "Color": ["green", balance_color]
        })
        fig_bars = px.bar(
            chart_data,
            x="Category",
            y="Amount",
            color="Category",
            color_discrete_map={
                "Initial Deposit": "green",
                "Current Balance": balance_color
            },
            labels={"Amount": "Amount (₦)", "Category": "Category"}
        )
        fig_bars.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font_color="black")
        st.plotly_chart(fig_bars, use_container_width=True, key="balance_bars")


with tab_atm:
    st.header("ATM Panel")
    
    account_number = st.text_input("Enter Account Number")

    # Ensure max withdrawal amount is set
    max_withdrawal_amount = st.session_state.data.get("max_withdrawal_amount", 50000)

    # Input field with dynamic max withdrawal limit
    withdrawal_amount = st.number_input(
        "Enter Withdrawal Amount", 
        min_value=1, 
        step=1, 
        max_value=max_withdrawal_amount
    )

    if st.button("Withdraw"):
        if account_number:
            data = read_data()

            # Ensure valid withdrawal amount
            if withdrawal_amount > max_withdrawal_amount:
                st.error(f"❌ Withdrawal exceeds max limit of ₦{max_withdrawal_amount:,}")
                st.stop()
            elif withdrawal_amount > data["total_amount"]:
                st.error("❌ Insufficient balance!")
                st.stop()
                
            else:
                # Process withdrawal
                data["total_amount"] -= withdrawal_amount
                data["transactions"].append({
                    "account_number": account_number,
                    "date": today_date,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "amount": -withdrawal_amount,
                    "type": "Withdrawal"
                })
                data["withdrawal_history"].append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": withdrawal_amount
                })

                # Auto-refill if below threshold
                if data["total_amount"] < data["refill_threshold"]:
                    data["total_amount"] += data["auto_refill_amount"]
                    data["transactions"].append({
                        "account_number": "SYSTEM",
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "amount": data['auto_refill_amount'],
                        "type": "Auto Refill"
                    })

                write_data(data)
                st.session_state.data = data  
                st.success(f"✅ Withdrawal of ₦{withdrawal_amount:,.0f} successful!")
                st.rerun()  # Optional, but may cause lag

        else:
            st.error("❌ Please enter a valid account number!")
            
            

    # ATM Simulation
    st.subheader("Simulation")
    num_transactions = st.number_input("Number of transactions to simulate", min_value=1, step=1, value=10)
    if st.button("Simulate Random Withdrawals"):
        simulate_random_withdrawals(num_transactions)

         
        
with tab_settings:
    st.header("Settings")
    with st.form("settings_form"):
        auto_refill_amount = st.number_input("Auto Refill Amount", min_value=1, step=1,
                                             value=st.session_state.data["auto_refill_amount"])
        max_withdrawal_amount = st.number_input("Maximum Withdrawal Amount", min_value=1, step=1,
                                               value=st.session_state.data["max_withdrawal_amount"])
        refill_threshold = st.number_input("Refill Threshold", min_value=1, step=1,
                                           value=st.session_state.data["refill_threshold"])

        if st.form_submit_button("Save Settings"):
            if refill_threshold >= auto_refill_amount:
                st.error("Refill threshold must be less than auto-refill amount.")
            else:
                data = read_data()
                data["auto_refill_amount"] = auto_refill_amount
                data["max_withdrawal_amount"] = max_withdrawal_amount
                data["refill_threshold"] = refill_threshold
                write_data(data)
                st.session_state.data = data  
                st.success("Settings saved successfully!")
                st.rerun()

