import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pandas as pd
from scipy import signal, ndimage, interpolate
import copy

timestamp_0 = 539330432

raw_time_takeoff = 577  # Note: data from first 10 seconds is dirty
raw_time_landing = 840  # Note: data from last 10 seconds is dirty

t_max = raw_time_landing - raw_time_takeoff

data_sources = {
    "airspeed"  : ("./data/flight3_airspeed_validated_0.csv", "calibrated_airspeed_m_s"),
    "barometer" : ("./data/flight3_sensor_baro_0.csv", "pressure"),
    "baro_alt"  : ("./data/flight3_vehicle_air_data_0.csv", "baro_alt_meter"),
    "gps_alt_mm": ("./data/flight3_vehicle_gps_position_0.csv", "alt"),
    "voltage"   : ("./data/flight3_battery_status_1.csv", "voltage_v"),
    "current"   : ("./data/flight3_battery_status_1.csv", "current_a"),
}


def read(name):
    source = data_sources[name][0]
    colname = data_sources[name][1]

    df = pd.read_csv(source)

    raw_time = (df["timestamp"].values - timestamp_0) / 1e6
    data = df[colname].values

    mask = (raw_time > raw_time_takeoff) & (raw_time < raw_time_landing)

    time = raw_time[mask] - raw_time_takeoff
    data = data[mask]

    estimated_error = np.std(np.diff(np.diff(data))) / np.sqrt(6)
    print(f"{name} estimated error: {estimated_error}")

    w = 1 / estimated_error * np.ones_like(data)

    interpolator = interpolate.UnivariateSpline(
        x=time,
        y=data,
        k=5,
        w=w,
        s=len(w),
        check_finite=True,
        ext='raise',
    )

    return interpolator, time, data


airspeed = read("airspeed")[0]
baro_alt = read("baro_alt")[0]
gps_alt_mm = read("gps_alt_mm")[0]
voltage = read("voltage")[0]
current = read("current")[0]

t = np.linspace(10, t_max - 10, 5000)
dt = np.diff(t)[0]


def f(x):
    # return x
    return ndimage.gaussian_filter(x, sigma=5 / dt)
    # return ndimage.uniform_filter(x, size=int(15 / dt))


opti = asb.Opti()

mass_total = 9.5

avionics_power = 5  # opti.variable(init_guess=4, lower_bound=0, upper_bound=20)

mean_current = np.mean(f(current(t)))
mean_airspeed = np.mean(f(airspeed(t)))

propto_rpm = np.softmax(
    f(current(t)) / mean_current,
    1e-3,
    softness=0.01
) ** (1 / 3)

propto_J = (f(airspeed(t)) / mean_airspeed) / propto_rpm

### Model 0
prop_eff_params = {
    "eff" : opti.variable(init_guess=0.8, lower_bound=0, upper_bound=1),
}

prop_efficiency = prop_eff_params["eff"]

### Model 1
# prop_eff_params = {
#     "Jc" : opti.variable(init_guess=propto_J.mean()),
#     "Js" : opti.variable(init_guess=propto_J.std(), log_transform=True, lower_bound=0.001, upper_bound=1000),
#     "max": opti.variable(init_guess=0.5, lower_bound=0, upper_bound=1),
#     "min": opti.variable(init_guess=0.5, lower_bound=0, upper_bound=1),
# }
#
# prop_efficiency = np.blend(
#     (
#             (propto_J - prop_eff_params["Jc"]) / prop_eff_params["Js"]
#     ),
#     prop_eff_params["max"],
#     prop_eff_params["min"]
# )

### Model 2
# prop_eff_params = {
#     "scale"        : opti.variable(init_guess=1, lower_bound=0, upper_bound=1),
#     "J_pitch_speed": opti.variable(init_guess=2, lower_bound=0),
#     "sharpness"    : opti.variable(init_guess=10, lower_bound=2, upper_bound=50),
# }
#
# prop_efficiency = np.softmax(
#     prop_eff_params["scale"] * (
#             (propto_J / prop_eff_params["J_pitch_speed"]) -
#             (propto_J / prop_eff_params["J_pitch_speed"]) ** prop_eff_params["sharpness"]
#     ),
#         -1,
#     softness=0.1
# )


propulsion_air_power = prop_efficiency * (f(voltage(t)) * f(current(t)) - avionics_power)

q = 0.5 * 1.225 * f(airspeed(t)) ** 2
S = 1.499
CL = mass_total * 9.81 / (q * S)

CD_params = {
    "CD0"  : opti.variable(init_guess=0.07, lower_bound=0, upper_bound=1),
    "CLCD0": opti.variable(init_guess=0.5, lower_bound=0, upper_bound=1.5),
    "CD2"  : opti.variable(init_guess=0.05, lower_bound=0, upper_bound=10),
    "CD3"  : opti.variable(init_guess=0.05, lower_bound=0, upper_bound=10),
    "CD4"  : opti.variable(init_guess=0.05, lower_bound=0, upper_bound=10),
}

CD = (
        CD_params["CD0"]
        + CD_params["CD2"] * np.abs(CL - CD_params["CLCD0"]) ** 2
        + CD_params["CD3"] * np.abs(CL - CD_params["CLCD0"]) ** 3
        + CD_params["CD4"] * np.abs(CL - CD_params["CLCD0"]) ** 4
)

drag_power = CD * q * S * f(airspeed(t))

residuals = (
        propulsion_air_power
        - drag_power
        - mass_total * f(airspeed(t)) * f(airspeed(t, 1))
        - mass_total * 9.81 * f(baro_alt(t, 1))
)

##### Objective

### L2-norm
opti.minimize(
    np.mean(residuals ** 2)
)

# ### L1-norm
# abs_residual = opti.variable(init_guess=0, n_vars=np.length(residuals))
# opti.subject_to([
#     abs_residual >= residuals,
#     abs_residual >= -residuals,
# ])
# opti.minimize(np.mean(abs_residual))

##### Solve

sol = opti.solve(
    # verbose=False
)

##### Post-Process, Print
print("\nMean Absolute Error:", np.mean(np.abs(sol(residuals))), "W")

l = copy.copy(locals())
for k, v in l.items():
    import casadi as cas

    if isinstance(v, (
            cas.MX, float, int, np.ndarray, list, tuple, dict
    )) and not k.startswith("_") and k != "l":
        # print(k)
        exec(f"{k} = sol({k})")

for k, v in CD_params.items():
    CD_params[k] = np.maximum(0, v)

from pprint import pprint

print("-" * 50)
for var in [
    "avionics_power",
    "prop_eff_params",
    "CD_params",
]:
    print(var)
    pprint(sol(eval(var)))

########## Reconstruct Steady-State Solution

mean_propto_J = np.mean(propto_J)


def steady_state_CD(CL):
    return (
            CD_params["CD0"]
            + CD_params["CD2"] * np.abs(CL - CD_params["CLCD0"]) ** 2
            + CD_params["CD3"] * np.abs(CL - CD_params["CLCD0"]) ** 3
            + CD_params["CD4"] * np.abs(CL - CD_params["CLCD0"]) ** 4
    )


def steady_state_prop_efficiency(propto_J):
    return prop_eff_params["eff"] * np.ones_like(propto_J)

    # return np.blend(
    #     (
    #             (propto_J - prop_eff_params["Jc"]) / prop_eff_params["Js"]
    #     ),
    #     prop_eff_params["max"],
    #     prop_eff_params["min"]
    # )

    # return np.softmax(
    #     prop_eff_params["scale"] * (
    #             (propto_J / prop_eff_params["J_pitch_speed"]) -
    #             (propto_J / prop_eff_params["J_pitch_speed"]) ** prop_eff_params["sharpness"]
    #     ),
    #     -1,
    #     softness=0.1
    # )


def steady_state_required_electrical_power(airspeed):
    q = 0.5 * 1.225 * airspeed ** 2
    CL = mass_total * 9.81 / (q * S)
    CD = steady_state_CD(CL)

    drag_power = CD * q * S * airspeed

    opti2 = asb.Opti()
    propto_J = opti2.variable(init_guess=mean_propto_J, n_vars=len(airspeed))

    prop_efficiency = steady_state_prop_efficiency(propto_J)

    required_current = drag_power / (voltage(t).mean() * prop_efficiency)
    required_propto_rpm = np.softmax(
        required_current / mean_current,
        1e-3,
        softness=0.01
    ) ** (1 / 3)
    required_propto_J = (airspeed / mean_airspeed) / required_propto_rpm

    opti2.subject_to(
        propto_J == required_propto_J
    )

    sol2 = opti2.solve(
        verbose=False
    )

    return sol2(
        drag_power / prop_efficiency + avionics_power
    )


########## Plot Energy Polar
fig, ax = plt.subplots()


def jitter(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    return data + np.random.uniform(-1, 1, len(data)) * iqr * 0.04


##### Plot data
plt.plot(
    jitter(f(airspeed(t))),
    jitter(steady_state_required_electrical_power(f(airspeed(t))) + residuals),
    ".",
    color="k",
    alpha=(1 - (1 - 0.5) ** (1 / (len(t) / 500))),
    markersize=3,
)
plt.plot(
    [],
    [],
    ".k",
    markersize=3,
    label="Flight Data, with total-energy\ncorrections, powertrain eff.",
)

##### Plot fit

plot_speeds = np.linspace(6, 20, 1000)
plot_powers = steady_state_required_electrical_power(plot_speeds)

_line, = plt.plot(
    plot_speeds,
    plot_powers,
    linewidth=2.5,
    color=p.adjust_lightness("red", 1.2),
    alpha=0.75,
    zorder=4,
    label="Model Fit (physics-informed; $L_1$ norm)",
)
i_min = np.argmin(plot_powers)
plt.plot(
    plot_speeds[i_min],
    plot_powers[i_min],
    "o",
    color=_line._color,
    markersize=5,
    alpha=0.8,
)
plt.annotate(
    text=f"Min. Power:\n{plot_speeds[i_min]:.2f} m/s\n{plot_powers[i_min]:.0f} W",
    xy=(plot_speeds[i_min], plot_powers[i_min]),
    xytext=(0, 100),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=10,
    color=p.adjust_lightness(_line._color, 0.8),
    alpha=0.8,
    arrowprops=dict(
        arrowstyle="->",
        color=p.adjust_lightness(_line._color, 0.8),
        linewidth=1.5,
        alpha=0.8,
        shrinkB=10,
    )
)

solar_power_input = 140

p.hline(
    y=solar_power_input,
    color="darkgoldenrod",
    text=f"Solar Power Input ({solar_power_input:.0f} W)",
    text_xloc=0.95,
    text_ha="right",
    text_kwargs=dict(
        fontsize=10,
    ),
    zorder=3.5,
)

plt.annotate(
    text="Watermark: Plot made by Peter",
    xy=(0.98, 0.02),
    xycoords="axes fraction",
    ha="right",
    va="bottom",
    fontsize=13,
    alpha=0.2,
)

plt.xlim(6, 20)
plt.ylim(0, 500)

p.set_ticks(1, 0.25, 100, 25)

plt.legend(
    # loc="lower right",
)
p.show_plot(
    "Solar Seaplane Power Polar",
    "Cruise Airspeed [m/s]",
    "Required Electrical Power for Cruise [W]",
    legend=False
)

########## Plot L/D Polar
fig, ax = plt.subplots()

##### Plot data
QS_plot = (0.5 * 1.225 * (f(airspeed(t)) ** 2) * S)

CL_plot = mass_total * 9.81 / QS_plot
CD_plot = steady_state_CD(CL_plot) + residuals / QS_plot / f(airspeed(t))

plt.plot(
    jitter(CD_plot),
    jitter(CL_plot),
    ".",
    color="k",
    alpha=(1 - (1 - 0.5) ** (1 / (len(t) / 500))),
    markersize=3,
)
plt.plot(
    [],
    [],
    ".k",
    markersize=3,
    label="Flight Data, with total-energy\ncorrections, powertrain eff.",
)

##### Plot fit
CL_plot = np.linspace(0, 1.8, 1000)
CD_plot = steady_state_CD(CL_plot)

_line, = plt.plot(
    CD_plot,
    CL_plot,
    linewidth=2.5,
    color=p.adjust_lightness("red", 1.2),
    alpha=0.75,
    zorder=4,
    label="Model Fit (physics-informed; $L_1$ norm)",
)

plt.xlim(0, 0.2)
plt.ylim(0, 1.8)

p.set_ticks(0.02, 0.005, 0.2, 0.05)

plt.legend(
    # loc="lower right",
)
p.show_plot(
    "Solar Seaplane Aerodynamic Polar",
    "Drag Coefficient $C_D$",
    "Lift Coefficient $C_L$",
    legend=False
)

########## Plot Propeller Efficiency Polar
fig, ax = plt.subplots()

##### Plot data
mask = f(current(t)) > 1

prop_efficiency_plot = (
                               propulsion_air_power - residuals
                       ) / (f(voltage(t)) * f(current(t)) - avionics_power)

plt.plot(
    jitter(propto_J[mask]),
    jitter(prop_efficiency_plot[mask]),
    ".",
    color="k",
    alpha=(1 - (1 - 0.5) ** (1 / (len(t) / 500))),
    markersize=3,
)
plt.plot(
    [],
    [],
    ".k",
    markersize=3,
    label="Flight Data, with total-energy\ncorrections, powertrain eff.",
)

##### Plot fit
propto_J_plot = np.linspace(
    0, propto_J[mask].max(), 1000
)

prop_efficiency_plot = steady_state_prop_efficiency(propto_J_plot)

_line, = plt.plot(
    propto_J_plot,
    prop_efficiency_plot,
    linewidth=2.5,
    color=p.adjust_lightness("red", 1.2),
    alpha=0.75,
    zorder=4,
    label="Model Fit (physics-informed; $L_1$ norm)",
)

plt.xlim(0, propto_J[mask].max())
plt.ylim(bottom=0)
#
p.set_ticks(0.5, 0.1, 0.2, 0.05)

from matplotlib import ticker

ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))

plt.legend(
    # loc="lower right",
)
p.show_plot(
    "Solar Seaplane Propulsion Polar",
    "$V\\ /\\ (\\rm current)^{1/3}$, proportional to advance ratio $J$",
    "Propulsive Efficiency $\\eta_p$",
    legend=False
)
