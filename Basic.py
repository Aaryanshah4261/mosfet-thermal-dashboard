import json
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MOSFET Electro-Thermal Dashboard",
    page_icon="🌡️",
    layout="wide",
)

st.title("MOSFET Electro-Thermal Design Dashboard")
st.info(
    "This application estimates MOSFET steady-state and transient junction "
    "temperature using user-defined electrical properties, operating conditions, "
    "and thermal-network parameters."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def rds_at_temperature(
    rds_25_ohm: float,
    alpha_per_c: float,
    junction_temperature_c: float,
) -> float:
    """
    Linear RDS(on) temperature model:

        RDS(Tj) = RDS(25°C) * [1 + alpha * (Tj - 25°C)]

    alpha_per_c must be entered as a decimal value.
    Example: 0.7 %/°C -> 0.007 /°C.
    """
    multiplier = 1.0 + alpha_per_c * (junction_temperature_c - 25.0)

    # Prevent an unphysical negative resistance when extrapolating
    # far below 25°C.
    multiplier = max(multiplier, 0.05)

    return rds_25_ohm * multiplier


def calculate_switching_energy(
    bus_voltage_v: float,
    switching_current_a: float,
    transition_time_s: float,
    correction_factor: float,
) -> float:
    """
    Basic linear voltage-current overlap estimate:

        E = 0.5 * VDS * ID * t_transition * correction_factor
    """
    return (
        0.5
        * bus_voltage_v
        * switching_current_a
        * transition_time_s
        * correction_factor
    )


def calculate_steady_state(
    *,
    ambient_temperature_c: float,
    conduction_current_a: float,
    conduction_mode: str,
    duty_cycle: float,
    switching_current_a: float,
    switching_frequency_hz: float,
    bus_voltage_v: float,
    rise_time_s: float,
    fall_time_s: float,
    switching_correction_factor: float,
    rds_25_ohm: float,
    alpha_per_c: float,
    rth_jc_c_per_w: float,
    rth_cs_c_per_w: float,
    rth_sa_c_per_w: float,
    devices_on_shared_sink: int,
    tolerance_c: float = 1e-6,
    maximum_iterations: int = 500,
) -> dict:
    """
    Calculates the temperature of one representative MOSFET.

    All MOSFETs sharing the heatsink are assumed identical and operating
    under the same conditions.
    """
    tj_guess_c = ambient_temperature_c
    converged = False
    thermal_runaway = False

    e_on_j = calculate_switching_energy(
        bus_voltage_v,
        switching_current_a,
        rise_time_s,
        switching_correction_factor,
    )
    e_off_j = calculate_switching_energy(
        bus_voltage_v,
        switching_current_a,
        fall_time_s,
        switching_correction_factor,
    )
    switching_loss_w = (e_on_j + e_off_j) * switching_frequency_hz

    result = {}

    for iteration in range(1, maximum_iterations + 1):
        rds_hot_ohm = rds_at_temperature(
            rds_25_ohm,
            alpha_per_c,
            tj_guess_c,
        )

        if conduction_mode == "ON-State Current + Duty Cycle":
            conduction_loss_w = (
                conduction_current_a**2
                * rds_hot_ohm
                * duty_cycle
            )
            full_cycle_rms_current_a = (
                conduction_current_a * math.sqrt(duty_cycle)
            )
        else:
            conduction_loss_w = (
                conduction_current_a**2
                * rds_hot_ohm
            )
            full_cycle_rms_current_a = conduction_current_a

        device_loss_w = conduction_loss_w + switching_loss_w
        total_sink_loss_w = device_loss_w * devices_on_shared_sink

        sink_temperature_c = (
            ambient_temperature_c
            + total_sink_loss_w * rth_sa_c_per_w
        )
        case_temperature_c = (
            sink_temperature_c
            + device_loss_w * rth_cs_c_per_w
        )
        calculated_tj_c = (
            case_temperature_c
            + device_loss_w * rth_jc_c_per_w
        )

        if (
            not np.isfinite(calculated_tj_c)
            or calculated_tj_c > 2000.0
        ):
            thermal_runaway = True
            break

        if abs(calculated_tj_c - tj_guess_c) <= tolerance_c:
            tj_guess_c = calculated_tj_c
            converged = True
            break

        # Under-relaxation improves stability when electro-thermal
        # feedback is strong.
        tj_guess_c = 0.5 * tj_guess_c + 0.5 * calculated_tj_c

    if not thermal_runaway:
        rds_hot_ohm = rds_at_temperature(
            rds_25_ohm,
            alpha_per_c,
            tj_guess_c,
        )

        if conduction_mode == "ON-State Current + Duty Cycle":
            conduction_loss_w = (
                conduction_current_a**2
                * rds_hot_ohm
                * duty_cycle
            )
            full_cycle_rms_current_a = (
                conduction_current_a * math.sqrt(duty_cycle)
            )
        else:
            conduction_loss_w = (
                conduction_current_a**2
                * rds_hot_ohm
            )
            full_cycle_rms_current_a = conduction_current_a

        device_loss_w = conduction_loss_w + switching_loss_w
        total_sink_loss_w = device_loss_w * devices_on_shared_sink

        sink_temperature_c = (
            ambient_temperature_c
            + total_sink_loss_w * rth_sa_c_per_w
        )
        case_temperature_c = (
            sink_temperature_c
            + device_loss_w * rth_cs_c_per_w
        )
        junction_temperature_c = (
            case_temperature_c
            + device_loss_w * rth_jc_c_per_w
        )
    else:
        conduction_loss_w = float("nan")
        device_loss_w = float("nan")
        total_sink_loss_w = float("nan")
        sink_temperature_c = float("nan")
        case_temperature_c = float("nan")
        junction_temperature_c = float("inf")
        rds_hot_ohm = float("nan")
        full_cycle_rms_current_a = float("nan")

    result.update(
        {
            "converged": converged,
            "thermal_runaway": thermal_runaway,
            "iterations": iteration,
            "e_on_j": e_on_j,
            "e_off_j": e_off_j,
            "switching_loss_w": switching_loss_w,
            "conduction_loss_w": conduction_loss_w,
            "device_loss_w": device_loss_w,
            "total_sink_loss_w": total_sink_loss_w,
            "sink_temperature_c": sink_temperature_c,
            "case_temperature_c": case_temperature_c,
            "junction_temperature_c": junction_temperature_c,
            "rds_hot_ohm": rds_hot_ohm,
            "full_cycle_rms_current_a": full_cycle_rms_current_a,
        }
    )
    return result


def calculate_required_rth_sa(
    *,
    target_junction_temperature_c: float,
    ambient_temperature_c: float,
    conduction_current_a: float,
    conduction_mode: str,
    duty_cycle: float,
    switching_loss_w: float,
    rds_25_ohm: float,
    alpha_per_c: float,
    rth_jc_c_per_w: float,
    rth_cs_c_per_w: float,
    devices_on_shared_sink: int,
) -> tuple[float, float]:
    """
    Calculates the required sink-to-ambient thermal resistance.

    Losses are evaluated at the target junction temperature.
    Returns:
        required RthSA, device loss at target
    """
    rds_target_ohm = rds_at_temperature(
        rds_25_ohm,
        alpha_per_c,
        target_junction_temperature_c,
    )

    if conduction_mode == "ON-State Current + Duty Cycle":
        conduction_loss_target_w = (
            conduction_current_a**2
            * rds_target_ohm
            * duty_cycle
        )
    else:
        conduction_loss_target_w = (
            conduction_current_a**2
            * rds_target_ohm
        )

    device_loss_target_w = (
        conduction_loss_target_w + switching_loss_w
    )
    total_sink_loss_target_w = (
        device_loss_target_w * devices_on_shared_sink
    )

    if total_sink_loss_target_w <= 0.0:
        return float("inf"), device_loss_target_w

    available_sink_temperature_rise_c = (
        target_junction_temperature_c
        - ambient_temperature_c
        - device_loss_target_w
        * (rth_jc_c_per_w + rth_cs_c_per_w)
    )

    required_rth_sa = (
        available_sink_temperature_rise_c
        / total_sink_loss_target_w
    )

    return required_rth_sa, device_loss_target_w


def build_transient_time_array(
    *,
    t_on_s: float,
    t_off_s: float,
    number_of_pulses: int,
    minimum_tau_s: float,
    maximum_points: int = 200_000,
) -> tuple[np.ndarray, bool]:
    """
    Builds a time array that includes every turn-on and turn-off event.

    Returns:
        time_array, was_coarsened
    """
    period_s = t_on_s + t_off_s
    total_time_s = period_s * number_of_pulses

    requested_dt_s = min(
        t_on_s / 100.0,
        t_off_s / 100.0,
        minimum_tau_s / 30.0,
    )

    requested_dt_s = max(requested_dt_s, 1e-12)

    approximate_points = int(
        math.ceil(total_time_s / requested_dt_s)
    ) + 1

    was_coarsened = approximate_points > maximum_points

    if was_coarsened:
        uniform_dt_s = total_time_s / (maximum_points - 1)
    else:
        uniform_dt_s = requested_dt_s

    uniform_time = np.arange(
        0.0,
        total_time_s,
        uniform_dt_s,
        dtype=float,
    )

    turn_on_times = np.arange(number_of_pulses) * period_s
    turn_off_times = turn_on_times + t_on_s

    time_array = np.concatenate(
        [
            uniform_time,
            turn_on_times,
            turn_off_times,
            np.array([total_time_s]),
        ]
    )

    # Rounding removes floating-point duplicates at switching boundaries.
    time_array = np.unique(np.round(time_array, 15))
    time_array = time_array[
        (time_array >= 0.0)
        & (time_array <= total_time_s + 1e-15)
    ]
    time_array.sort()

    return time_array, was_coarsened


def simulate_transient_junction_temperature(
    *,
    fixed_case_temperature_c: float,
    pulse_current_a: float,
    switching_current_a: float,
    bus_voltage_v: float,
    t_on_s: float,
    t_off_s: float,
    number_of_pulses: int,
    rds_25_ohm: float,
    alpha_per_c: float,
    rise_time_s: float,
    fall_time_s: float,
    switching_correction_factor: float,
    foster_resistances_c_per_w: np.ndarray,
    foster_time_constants_s: np.ndarray,
) -> dict:
    """
    Junction-to-case Foster-network simulation.

    The case temperature is treated as a fixed boundary condition.
    Conduction loss is applied during each ON interval.
    Turn-on and turn-off switching energies are applied as thermal impulses.
    """
    period_s = t_on_s + t_off_s

    time_array_s, was_coarsened = build_transient_time_array(
        t_on_s=t_on_s,
        t_off_s=t_off_s,
        number_of_pulses=number_of_pulses,
        minimum_tau_s=float(np.min(foster_time_constants_s)),
    )

    number_of_steps = len(time_array_s)
    number_of_branches = len(foster_resistances_c_per_w)

    thermal_states_c = np.zeros(
        (number_of_branches, number_of_steps),
        dtype=float,
    )
    junction_temperature_c = np.full(
        number_of_steps,
        fixed_case_temperature_c,
        dtype=float,
    )
    displayed_power_w = np.zeros(number_of_steps, dtype=float)
    conduction_power_w = np.zeros(number_of_steps, dtype=float)

    e_on_j = calculate_switching_energy(
        bus_voltage_v,
        switching_current_a,
        rise_time_s,
        switching_correction_factor,
    )
    e_off_j = calculate_switching_energy(
        bus_voltage_v,
        switching_current_a,
        fall_time_s,
        switching_correction_factor,
    )

    turn_on_times = np.arange(number_of_pulses) * period_s
    turn_off_times = turn_on_times + t_on_s

    turn_on_keys = {
        round(float(value), 12) for value in turn_on_times
    }
    turn_off_keys = {
        round(float(value), 12) for value in turn_off_times
    }

    for k in range(number_of_steps):
        current_time_s = time_array_s[k]
        current_time_key = round(float(current_time_s), 12)

        event_energy_j = 0.0

        # For a Foster branch:
        # C_i = tau_i / R_i
        # Instantaneous temperature jump from energy E:
        # Delta T_i = E / C_i = E * R_i / tau_i
        if current_time_key in turn_on_keys:
            thermal_states_c[:, k] += (
                e_on_j
                * foster_resistances_c_per_w
                / foster_time_constants_s
            )
            event_energy_j += e_on_j

        if current_time_key in turn_off_keys:
            thermal_states_c[:, k] += (
                e_off_j
                * foster_resistances_c_per_w
                / foster_time_constants_s
            )
            event_energy_j += e_off_j

        junction_temperature_c[k] = (
            fixed_case_temperature_c
            + np.sum(thermal_states_c[:, k])
        )

        if k == number_of_steps - 1:
            break

        next_time_s = time_array_s[k + 1]
        dt_s = next_time_s - current_time_s
        midpoint_s = 0.5 * (current_time_s + next_time_s)

        position_in_period_s = midpoint_s % period_s
        is_on = position_in_period_s < t_on_s

        if is_on:
            rds_hot_ohm = rds_at_temperature(
                rds_25_ohm,
                alpha_per_c,
                junction_temperature_c[k],
            )
            p_cond_w = pulse_current_a**2 * rds_hot_ohm
        else:
            p_cond_w = 0.0

        conduction_power_w[k] = p_cond_w

        # Event energy is shown as an equivalent power over this interval
        # only for visualization.
        displayed_power_w[k] = (
            p_cond_w
            + (event_energy_j / dt_s if dt_s > 0.0 else 0.0)
        )

        decay = np.exp(
            -dt_s / foster_time_constants_s
        )

        thermal_states_c[:, k + 1] = (
            thermal_states_c[:, k] * decay
            + p_cond_w
            * foster_resistances_c_per_w
            * (1.0 - decay)
        )

    peak_index = int(np.argmax(junction_temperature_c))

    return {
        "time_array_s": time_array_s,
        "junction_temperature_c": junction_temperature_c,
        "displayed_power_w": displayed_power_w,
        "conduction_power_w": conduction_power_w,
        "peak_temperature_c": float(
            junction_temperature_c[peak_index]
        ),
        "peak_time_s": float(time_array_s[peak_index]),
        "e_on_j": e_on_j,
        "e_off_j": e_off_j,
        "was_coarsened": was_coarsened,
    }


# ============================================================
# SIDEBAR: CUSTOM MOSFET INPUTS
# ============================================================
st.sidebar.header("1. Custom MOSFET Properties")

device_name = st.sidebar.text_input(
    "Device Name",
    value="Custom MOSFET",
)

vds_max_v = st.sidebar.number_input(
    "Maximum VDS Rating (V)",
    min_value=0.1,
    value=100.0,
    step=1.0,
)

tj_max_c = st.sidebar.number_input(
    "Maximum Junction Temperature (°C)",
    value=175.0,
    step=1.0,
)

rds_25_mohm = st.sidebar.number_input(
    "RDS(on) at 25°C (mΩ)",
    min_value=0.001,
    value=8.4,
    step=0.1,
)
rds_25_ohm = rds_25_mohm / 1000.0

alpha_percent_per_c = st.sidebar.number_input(
    "RDS(on) Temperature Coefficient (%/°C)",
    min_value=0.0,
    value=0.70,
    step=0.01,
    help=(
        "Linear approximation. Example: 0.70 %/°C means "
        "alpha = 0.007 /°C."
    ),
)
alpha_per_c = alpha_percent_per_c / 100.0

rise_time_ns = st.sidebar.number_input(
    "Rise Time, tr (ns)",
    min_value=0.0,
    value=67.0,
    step=1.0,
)
fall_time_ns = st.sidebar.number_input(
    "Fall Time, tf (ns)",
    min_value=0.0,
    value=88.0,
    step=1.0,
)
rise_time_s = rise_time_ns * 1e-9
fall_time_s = fall_time_ns * 1e-9

switching_correction_factor = st.sidebar.number_input(
    "Switching-Loss Correction Factor",
    min_value=0.0,
    value=1.0,
    step=0.05,
    help=(
        "Multiplies the basic 0.5 × VDS × ID × transition-time "
        "switching-energy estimate."
    ),
)

st.sidebar.header("2. Thermal Properties")

rth_jc_c_per_w = st.sidebar.number_input(
    "Junction-to-Case RθJC (°C/W)",
    min_value=0.0,
    value=0.402,
    step=0.01,
)

rth_cs_c_per_w = st.sidebar.number_input(
    "Case-to-Sink RθCS (°C/W)",
    min_value=0.0,
    value=0.50,
    step=0.01,
)

rth_sa_c_per_w = st.sidebar.number_input(
    "Sink-to-Ambient RθSA (°C/W)",
    min_value=0.0,
    value=1.0,
    step=0.05,
)

# This dashboard now models a single MOSFET/heatsink path.
devices_on_shared_sink = 1

st.sidebar.header("3. Steady-State Operating Conditions")

conduction_mode = st.sidebar.radio(
    "Conduction Current Input",
    [
        "ON-State Current + Duty Cycle",
        "True Full-Cycle RMS Current",
    ],
)

conduction_current_a = st.sidebar.number_input(
    (
        "ON-State Drain Current (A)"
        if conduction_mode == "ON-State Current + Duty Cycle"
        else "Full-Cycle MOSFET RMS Current (A)"
    ),
    min_value=0.0,
    value=75.0,
    step=0.1,
)

if conduction_mode == "ON-State Current + Duty Cycle":
    duty_cycle_percent = st.sidebar.slider(
        "Conduction Duty Cycle (%)",
        min_value=0.1,
        max_value=100.0,
        value=50.0,
        step=0.1,
    )
    duty_cycle = duty_cycle_percent / 100.0
else:
    duty_cycle = 1.0

switching_current_a = st.sidebar.number_input(
    "Current During Switching Transition (A)",
    min_value=0.0,
    value=75.0,
    step=0.1,
)

bus_voltage_v = st.sidebar.number_input(
    "DC-Bus Voltage (V)",
    min_value=0.0,
    value=65.0,
    step=0.1,
)

switching_frequency_khz = st.sidebar.number_input(
    "Switching Frequency (kHz)",
    min_value=0.0,
    value=20.0,
    step=0.1,
)
switching_frequency_hz = switching_frequency_khz * 1000.0

ambient_temperature_c = st.sidebar.number_input(
    "Ambient Temperature (°C)",
    value=40.0,
    step=0.5,
)

target_junction_temperature_c = st.sidebar.number_input(
    "Target Junction Temperature (°C)",
    value=125.0,
    step=1.0,
)


# ============================================================
# INPUT WARNINGS
# ============================================================
if bus_voltage_v > vds_max_v:
    st.error(
        f"DC-bus voltage ({bus_voltage_v:.1f} V) exceeds the entered "
        f"device VDS rating ({vds_max_v:.1f} V)."
    )

if target_junction_temperature_c > tj_max_c:
    st.warning(
        "The selected design target is above the entered maximum "
        "junction-temperature rating."
    )


# ============================================================
# PHASE 1: STEADY-STATE CALCULATION
# ============================================================
steady_state = calculate_steady_state(
    ambient_temperature_c=ambient_temperature_c,
    conduction_current_a=conduction_current_a,
    conduction_mode=conduction_mode,
    duty_cycle=duty_cycle,
    switching_current_a=switching_current_a,
    switching_frequency_hz=switching_frequency_hz,
    bus_voltage_v=bus_voltage_v,
    rise_time_s=rise_time_s,
    fall_time_s=fall_time_s,
    switching_correction_factor=switching_correction_factor,
    rds_25_ohm=rds_25_ohm,
    alpha_per_c=alpha_per_c,
    rth_jc_c_per_w=rth_jc_c_per_w,
    rth_cs_c_per_w=rth_cs_c_per_w,
    rth_sa_c_per_w=rth_sa_c_per_w,
    devices_on_shared_sink=int(devices_on_shared_sink),
)

required_rth_sa, target_device_loss_w = calculate_required_rth_sa(
    target_junction_temperature_c=target_junction_temperature_c,
    ambient_temperature_c=ambient_temperature_c,
    conduction_current_a=conduction_current_a,
    conduction_mode=conduction_mode,
    duty_cycle=duty_cycle,
    switching_loss_w=steady_state["switching_loss_w"],
    rds_25_ohm=rds_25_ohm,
    alpha_per_c=alpha_per_c,
    rth_jc_c_per_w=rth_jc_c_per_w,
    rth_cs_c_per_w=rth_cs_c_per_w,
    devices_on_shared_sink=int(devices_on_shared_sink),
)

steady_tj_c = steady_state["junction_temperature_c"]
steady_tc_c = steady_state["case_temperature_c"]
steady_ts_c = steady_state["sink_temperature_c"]

if steady_state["thermal_runaway"]:
    steady_status = "🚨 FAIL — Thermal runaway or nonphysical result"
elif steady_tj_c >= tj_max_c:
    steady_status = "🚨 FAIL — Exceeds maximum junction temperature"
elif steady_tj_c >= target_junction_temperature_c:
    steady_status = "⚠️ WARNING — Exceeds design target"
else:
    steady_status = "✅ PASS — Below design target"

st.markdown("---")
st.header("Phase 1: Steady-State Electro-Thermal Analysis")
st.subheader(f"System Status: {steady_status}")

if not steady_state["converged"] and not steady_state["thermal_runaway"]:
    st.warning(
        "The electro-thermal iteration reached its iteration limit. "
        "Review the operating point and temperature coefficient."
    )

power_col1, power_col2, power_col3, power_col4 = st.columns(4)

with power_col1.container(border=True):
    st.metric(
        "Conduction Loss / MOSFET",
        f"{steady_state['conduction_loss_w']:.2f} W",
    )

with power_col2.container(border=True):
    st.metric(
        "Switching Loss / MOSFET",
        f"{steady_state['switching_loss_w']:.2f} W",
    )

with power_col3.container(border=True):
    st.metric(
        "Total Loss / MOSFET",
        f"{steady_state['device_loss_w']:.2f} W",
    )

with power_col4.container(border=True):
    st.metric(
        "Total Heat into Heatsink",
        f"{steady_state['total_sink_loss_w']:.2f} W",
    )

temperature_col1, temperature_col2, temperature_col3, temperature_col4 = (
    st.columns(4)
)

with temperature_col1.container(border=True):
    st.metric(
        "Heatsink Temperature",
        f"{steady_ts_c:.2f} °C",
    )

with temperature_col2.container(border=True):
    st.metric(
        "Case Temperature",
        f"{steady_tc_c:.2f} °C",
    )

with temperature_col3.container(border=True):
    st.metric(
        "Junction Temperature",
        f"{steady_tj_c:.2f} °C",
    )

with temperature_col4.container(border=True):
    st.metric(
        "Hot RDS(on)",
        f"{steady_state['rds_hot_ohm'] * 1000.0:.3f} mΩ",
    )

margin_target_c = target_junction_temperature_c - steady_tj_c
margin_max_c = tj_max_c - steady_tj_c

cooling_col1, cooling_col2, cooling_col3 = st.columns(3)

with cooling_col1.container(border=True):
    st.metric(
        "Margin to Design Target",
        f"{margin_target_c:.2f} °C",
    )

with cooling_col2.container(border=True):
    st.metric(
        "Margin to Maximum Rating",
        f"{margin_max_c:.2f} °C",
    )

with cooling_col3.container(border=True):
    if math.isinf(required_rth_sa):
        st.metric("Required RθSA", "Not applicable")
    elif required_rth_sa < 0.0:
        st.metric("Required RθSA", "Not physically achievable")
    else:
        st.metric(
            "Required RθSA",
            f"{required_rth_sa:.3f} °C/W",
        )

if required_rth_sa < 0.0:
    st.error(
        "The target junction temperature cannot be achieved by improving "
        "the heatsink alone. The junction-to-case and case-to-sink "
        "temperature rise already consumes the available temperature margin."
    )

with st.expander("Steady-State Calculation Details"):
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.write(
            f"Full-cycle RMS current used for conduction: "
            f"**{steady_state['full_cycle_rms_current_a']:.3f} A**"
        )
        st.write(
            f"Estimated turn-on energy: "
            f"**{steady_state['e_on_j'] * 1e6:.3f} µJ**"
        )
        st.write(
            f"Estimated turn-off energy: "
            f"**{steady_state['e_off_j'] * 1e6:.3f} µJ**"
        )

    with detail_col2:
        st.write(
            f"Electro-thermal iterations: "
            f"**{steady_state['iterations']}**"
        )
        st.write(
            f"Device loss evaluated at target temperature: "
            f"**{target_device_loss_w:.3f} W**"
        )
        st.write(
            "The heatsink temperature rise is calculated for one MOSFET."
        )

visual_col1, visual_col2 = st.columns(2)

with visual_col1:
    st.subheader("Power-Loss Distribution")
    power_dataframe = pd.DataFrame(
        {
            "Loss Type": ["Conduction", "Switching"],
            "Power (W)": [
                steady_state["conduction_loss_w"],
                steady_state["switching_loss_w"],
            ],
        }
    )
    power_figure = px.pie(
        power_dataframe,
        values="Power (W)",
        names="Loss Type",
        hole=0.40,
    )
    power_figure.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )
    power_figure.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
    )
    st.plotly_chart(power_figure, use_container_width=True)

with visual_col2:
    st.subheader("Steady-State Thermal Path")
    thermal_path_dataframe = pd.DataFrame(
        {
            "Location": [
                "Ambient",
                "Heatsink",
                "Case",
                "Junction",
            ],
            "Temperature (°C)": [
                ambient_temperature_c,
                steady_ts_c,
                steady_tc_c,
                steady_tj_c,
            ],
        }
    )
    thermal_path_figure = px.line(
        thermal_path_dataframe,
        x="Location",
        y="Temperature (°C)",
        markers=True,
    )
    thermal_path_figure.update_layout(
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(
        thermal_path_figure,
        use_container_width=True,
    )


# ============================================================
# PHASE 2: TRANSIENT JUNCTION-TO-CASE SIMULATION
# ============================================================
st.markdown("---")
st.header("Phase 2: Transient Junction-to-Case Simulation")

st.info(
    "The Foster network describes transient thermal behavior from the "
    "junction to the case. The case temperature is therefore treated as a "
    "fixed boundary condition. Use the Phase 1 case temperature for a "
    "combined steady-state base temperature plus short-term junction ripple."
)

st.caption(
    "Phase 2 uses the ON-state drain current, switching-transition current, "
    "and DC-bus voltage entered in the left sidebar."
)

transient_col1, transient_col2 = st.columns(2)

with transient_col1.container(border=True):
    t_on_ms = st.number_input(
        "ON Duration (ms)",
        min_value=0.000001,
        value=1.0,
        step=0.1,
        format="%.6f",
    )
    t_off_ms = st.number_input(
        "OFF Duration (ms)",
        min_value=0.000001,
        value=1.0,
        step=0.1,
        format="%.6f",
    )
    number_of_pulses = st.number_input(
        "Number of Pulses",
        min_value=1,
        max_value=1000,
        value=3,
        step=1,
    )

with transient_col2.container(border=True):
    case_temperature_source = st.radio(
        "Fixed Case Temperature Source",
        [
            "Use Phase 1 Case Temperature",
            "Enter Case Temperature Manually",
        ],
    )

    if case_temperature_source == "Use Phase 1 Case Temperature":
        fixed_case_temperature_c = float(steady_tc_c)
        st.metric(
            "Case Temperature Used",
            f"{fixed_case_temperature_c:.2f} °C",
        )
    else:
        fixed_case_temperature_c = st.number_input(
            "Fixed Case Temperature (°C)",
            value=80.0,
            step=0.5,
        )

st.subheader("User-Defined Foster Thermal Network")

st.caption(
    "Enter the junction-to-case Foster branches from the selected MOSFET "
    "datasheet or a validated thermal model. Add or remove rows as needed."
)

default_foster_dataframe = pd.DataFrame(
    {
        "R_i (°C/W)": [
            0.09876251,
            0.20666970,
            0.09510464,
        ],
        "Tau_i (s)": [
            0.000111,
            0.001743,
            0.012269,
        ],
    }
)

foster_dataframe = st.data_editor(
    default_foster_dataframe,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="foster_network_editor",
)

foster_dataframe = foster_dataframe.dropna()

if foster_dataframe.empty:
    st.error(
        "Enter at least one valid Foster-network branch."
    )
    st.stop()

try:
    foster_resistances = foster_dataframe[
        "R_i (°C/W)"
    ].to_numpy(dtype=float)
    foster_time_constants = foster_dataframe[
        "Tau_i (s)"
    ].to_numpy(dtype=float)
except (ValueError, TypeError):
    st.error(
        "Foster resistance and time-constant entries must be numeric."
    )
    st.stop()

if np.any(foster_resistances <= 0.0):
    st.error(
        "Every Foster resistance must be greater than zero."
    )
    st.stop()

if np.any(foster_time_constants <= 0.0):
    st.error(
        "Every Foster time constant must be greater than zero."
    )
    st.stop()

foster_rth_sum = float(np.sum(foster_resistances))
foster_difference_percent = (
    abs(foster_rth_sum - rth_jc_c_per_w)
    / max(rth_jc_c_per_w, 1e-12)
    * 100.0
)

foster_metric_col1, foster_metric_col2 = st.columns(2)

with foster_metric_col1.container(border=True):
    st.metric(
        "Sum of Foster Resistances",
        f"{foster_rth_sum:.6f} °C/W",
    )

with foster_metric_col2.container(border=True):
    st.metric(
        "Entered Steady-State RθJC",
        f"{rth_jc_c_per_w:.6f} °C/W",
    )

if foster_difference_percent > 10.0:
    st.warning(
        "The sum of the Foster resistances differs from the entered "
        f"steady-state RθJC by {foster_difference_percent:.1f}%. "
        "Check that the transient network belongs to the same MOSFET."
    )

t_on_s = t_on_ms * 1e-3
t_off_s = t_off_ms * 1e-3

transient_result = simulate_transient_junction_temperature(
    fixed_case_temperature_c=fixed_case_temperature_c,
    pulse_current_a=conduction_current_a,
    switching_current_a=switching_current_a,
    bus_voltage_v=bus_voltage_v,
    t_on_s=t_on_s,
    t_off_s=t_off_s,
    number_of_pulses=int(number_of_pulses),
    rds_25_ohm=rds_25_ohm,
    alpha_per_c=alpha_per_c,
    rise_time_s=rise_time_s,
    fall_time_s=fall_time_s,
    switching_correction_factor=switching_correction_factor,
    foster_resistances_c_per_w=foster_resistances,
    foster_time_constants_s=foster_time_constants,
)

if transient_result["was_coarsened"]:
    st.warning(
        "The requested transient simulation would create too many points, "
        "so the uniform time step was automatically coarsened. All switching "
        "boundaries are still included in the time array."
    )

peak_transient_tj_c = transient_result["peak_temperature_c"]
peak_transient_time_ms = transient_result["peak_time_s"] * 1000.0
transient_margin_target_c = (
    target_junction_temperature_c - peak_transient_tj_c
)
transient_margin_max_c = tj_max_c - peak_transient_tj_c

if peak_transient_tj_c >= tj_max_c:
    transient_status = "🚨 FAIL — Exceeds maximum junction temperature"
elif peak_transient_tj_c >= target_junction_temperature_c:
    transient_status = "⚠️ WARNING — Exceeds design target"
else:
    transient_status = "✅ PASS — Below design target"

st.subheader(f"Transient Status: {transient_status}")

transient_metric_col1, transient_metric_col2, transient_metric_col3, transient_metric_col4 = (
    st.columns(4)
)

with transient_metric_col1.container(border=True):
    st.metric(
        "Maximum Transient Tj",
        f"{peak_transient_tj_c:.2f} °C",
    )

with transient_metric_col2.container(border=True):
    st.metric(
        "Time of Maximum Tj",
        f"{peak_transient_time_ms:.4f} ms",
    )

with transient_metric_col3.container(border=True):
    st.metric(
        "Margin to Target",
        f"{transient_margin_target_c:.2f} °C",
    )

with transient_metric_col4.container(border=True):
    st.metric(
        "Margin to Maximum",
        f"{transient_margin_max_c:.2f} °C",
    )

transient_dataframe = pd.DataFrame(
    {
        "Time (ms)": (
            transient_result["time_array_s"] * 1000.0
        ),
        "Junction Temperature (°C)": (
            transient_result["junction_temperature_c"]
        ),
        "Equivalent Power for Plot (W)": (
            transient_result["displayed_power_w"]
        ),
    }
)

transient_figure = make_subplots(
    specs=[[{"secondary_y": True}]]
)

transient_figure.add_trace(
    go.Scatter(
        x=transient_dataframe["Time (ms)"],
        y=transient_dataframe[
            "Junction Temperature (°C)"
        ],
        name="Junction Temperature",
        line=dict(width=2),
    ),
    secondary_y=False,
)

transient_figure.add_trace(
    go.Scatter(
        x=transient_dataframe["Time (ms)"],
        y=transient_dataframe[
            "Equivalent Power for Plot (W)"
        ],
        name="Power Loss",
        line=dict(width=1, dash="dot"),
        fill="tozeroy",
    ),
    secondary_y=True,
)

transient_figure.update_layout(
    title_text="Junction Temperature During Repeated Pulses",
    hovermode="x unified",
    margin=dict(t=50, b=20, l=20, r=20),
)

transient_figure.update_xaxes(
    title_text="Time (ms)"
)
transient_figure.update_yaxes(
    title_text="Junction Temperature (°C)",
    secondary_y=False,
)
transient_figure.update_yaxes(
    title_text="Equivalent Power (W)",
    secondary_y=True,
    showgrid=False,
)

st.plotly_chart(
    transient_figure,
    use_container_width=True,
)

with st.expander("Transient Calculation Details"):
    st.write(
        f"Estimated turn-on energy: "
        f"**{transient_result['e_on_j'] * 1e6:.3f} µJ**"
    )
    st.write(
        f"Estimated turn-off energy: "
        f"**{transient_result['e_off_j'] * 1e6:.3f} µJ**"
    )
    st.write(
        "Turn-on and turn-off energies are applied at their switching "
        "instants as thermal impulses. Conduction loss is applied only "
        "during each ON interval."
    )
    st.write(
        "The case temperature remains fixed throughout this transient "
        "simulation. Case, heatsink, and ambient thermal capacitances are "
        "not included."
    )


# ============================================================
# DEVICE PROFILE EXPORT
# ============================================================
st.markdown("---")
st.header("Export Device Profile")

device_profile = {
    "device_name": device_name,
    "vds_max_v": vds_max_v,
    "tj_max_c": tj_max_c,
    "rds_on_25_ohm": rds_25_ohm,
    "rds_temperature_coefficient_percent_per_c": (
        alpha_percent_per_c
    ),
    "rise_time_s": rise_time_s,
    "fall_time_s": fall_time_s,
    "switching_loss_correction_factor": (
        switching_correction_factor
    ),
    "rth_jc_c_per_w": rth_jc_c_per_w,
    "rth_cs_c_per_w": rth_cs_c_per_w,
    "foster_r_c_per_w": foster_resistances.tolist(),
    "foster_tau_s": foster_time_constants.tolist(),
}

st.download_button(
    label="Download Device Profile as JSON",
    data=json.dumps(device_profile, indent=2),
    file_name=(
        f"{device_name.strip().replace(' ', '_') or 'mosfet'}"
        "_profile.json"
    ),
    mime="application/json",
)


# ============================================================
# MODEL BOUNDARIES
# ============================================================
st.markdown("---")
st.header("Model Scope and Limitations")

st.markdown(
    """
- The steady-state model includes MOSFET conduction loss, basic switching-overlap loss, and a one-dimensional thermal-resistance path.
- The linear RDS(on) temperature coefficient is an approximation and should be selected for the intended temperature range.
- The transient model requires user-provided junction-to-case Foster resistances and time constants.
- Phase 2 treats the case temperature as fixed; it is not a complete junction-to-ambient transient model.
- Body-diode conduction, reverse-recovery loss, output-capacitance loss, gate-drive loss, avalanche energy, busbar parasitics, and thermal spreading are not included.
- Before using the dashboard for hardware limits, validate it against a manufacturer model, simulation, or measured temperature data.
"""
)
