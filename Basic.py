import streamlit as st

# --- SIDEBAR: DATASHEET INPUTS ---
st.sidebar.header("1. Datasheet Inputs")

r_th_jc = st.sidebar.number_input("Junction-to-Case Resistance (deg C/W)", value=0.402)
r_th_cs = st.sidebar.number_input("Case-to-Sink Resistance (deg C/W)", value=0.500)
t_j_max = st.sidebar.number_input("Max Junction Temp (deg C)", value=175.0)

r_ds_on = st.sidebar.number_input("Max On-State Resistance (milli-Ohms)", value=8.4) / 1000.0  
e_on = st.sidebar.number_input("Turn-On Energy (micro-Joules)", value=163.3) / 1e6            
e_off = st.sidebar.number_input("Turn-Off Energy (micro-Joules)", value=214.5) / 1e6          

# --- SIDEBAR: OPERATING CONDITIONS ---
st.sidebar.header("2. Operating Conditions")

current_mode = st.sidebar.radio("Current Input Mode", ["ON-State Current + Duty Cycle", "True Full-Cycle RMS"])
current = st.sidebar.number_input("Drain Current (Amps)", value=75.0)

if current_mode == "ON-State Current + Duty Cycle":
    duty_cycle = st.sidebar.slider("Duty Cycle (%)", 1, 100, 50) / 100.0
else:
    duty_cycle = 1.0 

f_sw = st.sidebar.number_input("Switching Frequency (kilo-Hertz)", value=20.0) * 1000.0 
t_a = st.sidebar.number_input("Ambient Temp (deg C)", value=40.0)
r_th_sa = st.sidebar.number_input("Sink-to-Ambient Resistance (deg C/W)", value=1.0)
t_j_target = st.sidebar.number_input("Target Junction Temp (deg C)", value=125.0)

# Just to prove the variables are working, let's print one to the main screen
# --- CHUNK 2: THE CALCULATION ENGINE ---
# 1. Power Loss Calculations
p_cond = (current ** 2) * r_ds_on * duty_cycle
p_sw = (e_on + e_off) * f_sw
p_total = p_cond + p_sw

# 2. Thermal Calculations (Steady-State 1D)
t_s = t_a + (p_total * r_th_sa)
t_c = t_s + (p_total * r_th_cs)
t_j = t_c + (p_total * r_th_jc)

# 3. Margins & Cooling Requirements
margin_target = t_j_target - t_j
margin_max = t_j_max - t_j

# Prevent dividing by zero if current is 0
if p_total > 0:
    r_th_sa_req = ((t_j_target - t_a) / p_total) - r_th_jc - r_th_cs
else:
    r_th_sa_req = float('inf')

# 4. Safety Logic
if t_j >= t_j_max:
    status = "🚨 FAIL (Exceeds Absolute Maximum)"
elif t_j >= t_j_target:
    status = "⚠️ WARNING (Exceeds Design Target)"
else:
    status = "✅ PASS (Thermally Safe)"


# --- CHUNK 3: DASHBOARD RESULTS DISPLAY ---
# --- CHUNK 3: DASHBOARD RESULTS DISPLAY ---
st.markdown("---")
st.subheader(f"System Status: {status}")

st.markdown("### ⚡ Power Loss Breakdown")
col1, col2, col3 = st.columns(3)
with col1.container(border=True):
    st.metric("Conduction Loss", f"{p_cond:.2f} W")
with col2.container(border=True):
    st.metric("Switching Loss", f"{p_sw:.2f} W")
with col3.container(border=True):
    st.metric("Total Power Loss", f"{p_total:.2f} W")

st.markdown("### 🌡️ Thermal Network Temperatures")
col4, col5, col6 = st.columns(3)
with col4.container(border=True):
    st.metric("Heatsink Temp (Ts)", f"{t_s:.2f} °C")
with col5.container(border=True):
    st.metric("Case Temp (Tc)", f"{t_c:.2f} °C")
with col6.container(border=True):
    st.metric("Junction Temp (Tj)", f"{t_j:.2f} °C")

st.markdown("### ❄️ Cooling Requirements & Margins")
col7, col8, col9 = st.columns(3)
with col7.container(border=True):
    st.metric("Margin to Target", f"{margin_target:.2f} °C")
with col8.container(border=True):
    st.metric("Margin to Absolute Max", f"{margin_max:.2f} °C")
with col9.container(border=True):
    st.metric("Required RθSA", f"{r_th_sa_req:.3f} °C/W")
# --- CHUNK 4: VISUALIZATIONS ---
# --- CHUNK 4: VISUALIZATIONS ---
import pandas as pd
import plotly.express as px

st.markdown("---")
st.markdown("### 📊 System Visualizations")

# Create two columns for side-by-side charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**Power Loss Distribution**")
    
    # We package our data into a Pandas DataFrame for Plotly
    pie_df = pd.DataFrame(
        {"Loss Type": ["Conduction", "Switching"],
         "Watts": [p_cond, p_sw]}
    )
    
    # Create the pie chart (hole=0.4 makes it a donut chart for a cleaner look)
    fig = px.pie(pie_df, values="Watts", names="Loss Type", hole=0.4)
    
    # Force the labels and percentages to display cleanly inside the slices
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    
    # Render the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

with col_chart2:
    st.markdown("**Thermal Path Gradient**")
    # This plots the temperature climbing up the thermal resistance network
    temp_df = pd.DataFrame(
        {"Temperature (°C)": [t_a, t_s, t_c, t_j]},
        index=["1. Ambient (Ta)", "2. Heatsink (Ts)", "3. Case (Tc)", "4. Junction (Tj)"]
    )
    st.line_chart(temp_df)