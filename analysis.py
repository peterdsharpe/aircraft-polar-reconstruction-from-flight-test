import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pandas as pd
from scipy import signal, ndimage, interpolate
import copy

timestamp_0 = 539330432

raw_time_takeoff = 576
raw_time_landing = 838

t_max = raw_time_landing - raw_time_takeoff

data_sources = {
    "airspeed" : ("./data/flight3_airspeed_validated_0.csv", "calibrated_airspeed_m_s"),
    "barometer": ("./data/flight3_sensor_baro_0.csv", "pressure"),
    "baro_alt" : ("./data/flight3_vehicle_air_data_0.csv", "baro_alt_meter"),
    "voltage"  : ("./data/flight3_battery_status_1.csv", "voltage_v"),
    "current"  : ("./data/flight3_battery_status_1.csv", "current_a"),
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
        w=w,
        s=len(w),
        check_finite=True,
    )

    return interpolator, time, data


airspeed = read("airspeed")[0]
baro_alt = read("baro_alt")[0]
voltage = read("voltage")[0]
current = read("current")[0]

t = np.linspace(10, t_max, 20000)
dt = np.diff(t)[0]


def f(x):
    # return x
    return ndimage.gaussian_filter(x, sigma=7 / dt)


opti = asb.Opti()

mass_total = 9.5

avionics_power = opti.variable(init_guess=4, lower_bound=0)

propto_rpm = f(current(t)) ** (1 / 3)
propto_J = f(airspeed(t)) / propto_rpm

prop_eff_params = {
    "Jc" : opti.variable(init_guess=6),
    "Js" : opti.variable(init_guess=2, log_transform=True, lower_bound=0.5),
    "max": opti.variable(init_guess=1, lower_bound=0, upper_bound=1),
    "min": opti.variable(init_guess=0.2, lower_bound=0, upper_bound=1),
}

prop_efficiency = np.blend(
    (
            (propto_J - prop_eff_params["Jc"]) / prop_eff_params["Js"]
    ),
    prop_eff_params["max"],
    prop_eff_params["min"]
)

propulsion_air_power = prop_efficiency * (f(voltage(t)) * f(current(t)) - avionics_power)

q = 0.5 * 1.225 * f(airspeed(t)) ** 2
S = 1.499
CL = mass_total * 9.81 / (q * S)

CD_params = {
    "CD0"  : opti.variable(init_guess=0.07, lower_bound=0, upper_bound=1),
    "CLCD0": opti.variable(init_guess=0.5, lower_bound=0, upper_bound=2),
    "CD2"  : opti.variable(init_guess=0.1, lower_bound=0, upper_bound=10),
    "CD3"  : opti.variable(init_guess=0.03, lower_bound=0, upper_bound=10),
    "CD4"  : opti.variable(init_guess=0.01, lower_bound=0, upper_bound=10),
    "ReExp": opti.variable(init_guess=-0.5, lower_bound=-1, upper_bound=0),
}

CD = (
             CD_params["CD0"]
             + CD_params["CD2"] * np.abs(CL - CD_params["CLCD0"]) ** 2
             + CD_params["CD3"] * np.abs(CL - CD_params["CLCD0"]) ** 3
             + CD_params["CD4"] * np.abs(CL - CD_params["CLCD0"]) ** 4
     ) * f(airspeed(t) / 10) ** CD_params["ReExp"]

drag_power = CD * q * S * f(airspeed(t))

residuals = (
        propulsion_air_power
        - drag_power
        - mass_total * f(airspeed(t)) * f(airspeed(t, 1))
        - mass_total * 9.81 * f(baro_alt(t, 1))
)

##### Objective

opti.minimize(
    np.mean(residuals ** 2)
)

##### Solve

sol = opti.solve(verbose=False)

l = copy.copy(locals())
for k, v in l.items():
    import casadi as cas

    if isinstance(v, (
            cas.MX, float, int, np.ndarray, list, tuple, dict
    )) and not k.startswith("_") and k != "l":
        # print(k)
        exec(f"{k} = sol({k})")

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
def steady_state_power(airspeed):
    q = 0.5 * 1.225 * airspeed ** 2
    CL = mass_total * 9.81 / (q * S)
    CD = (
                 CD_params["CD0"]
                 + CD_params["CD2"] * np.abs(CL - CD_params["CLCD0"]) ** 2
                 + CD_params["CD3"] * np.abs(CL - CD_params["CLCD0"]) ** 3
                 + CD_params["CD4"] * np.abs(CL - CD_params["CLCD0"]) ** 4
         ) * (airspeed / 10) ** CD_params["ReExp"]

    drag_power = CD * q * S * airspeed

    opti2 = asb.Opti()
    propto_J = opti2.variable(init_guess=mean_propto_J, n_vars=len(airspeed))

    prop_efficiency = np.blend(
        (
                (propto_J - prop_eff_params["Jc"]) / prop_eff_params["Js"]
        ),
        prop_eff_params["max"],
        prop_eff_params["min"]
    )

    required_current = drag_power / (voltage(t).mean() * prop_efficiency)
    required_propto_rpm = np.softmax(
        required_current ** (1 / 3),
        1e-3,
        softness=0.01
    )
    required_propto_J = airspeed / required_propto_rpm

    # opti2.subject_to(
    #     propto_J == required_propto_J
    # )
    J_residuals = propto_J - required_propto_J
    opti2.minimize(np.mean(J_residuals ** 2))

    sol2 = opti2.solve(
        # verbose=False
    )

    # if np.any(np.abs(sol2(J_residuals))) > 1e-4:
    #     raise ValueError("Failed to find steady-state solution")

    return sol2(
        drag_power / prop_efficiency + avionics_power
    )


# print(steady_state_power(np.array([10.])))

########## Plot
fig, ax = plt.subplots()

##### Plot data

np.random.seed(0)
x_jitter = np.random.uniform(-1, 1, len(t)) * 0.1
y_jitter = np.random.uniform(-1, 1, len(t)) * 5

plt.plot(
    f(airspeed(t)) + x_jitter,
    steady_state_power(f(airspeed(t))) + residuals + y_jitter,
    ".",
    color="k",
    alpha=0.2 / np.log(len(t)),
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
plot_powers = steady_state_power(plot_speeds)

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
    zorder=3.5,
)

plt.annotate(
    text="Watermark: Plot made by Peter",
    xy=(0.02, 0.98),
    xycoords="axes fraction",
    ha="left",
    va="top",
    fontsize=18,
    alpha=0.4,
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
