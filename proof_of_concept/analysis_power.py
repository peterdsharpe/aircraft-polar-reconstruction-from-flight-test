import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pandas as pd
from scipy import signal, ndimage, interpolate

timestamp_0 = 539330432

raw_time_takeoff = 576
raw_time_landing = 838

t_max = raw_time_landing - raw_time_takeoff

data_sources = {
    "airspeed" : ("../data/flight3_airspeed_validated_0.csv", "calibrated_airspeed_m_s"),
    "barometer": ("../data/flight3_sensor_baro_0.csv", "pressure"),
    "baro_alt" : ("../data/flight3_vehicle_air_data_0.csv", "baro_alt_meter"),
    "voltage"  : ("../data/flight3_battery_status_1.csv", "voltage_v"),
    "current"  : ("../data/flight3_battery_status_1.csv", "current_a"),
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


# for name, (source, colname) in data_sources.items():
#     interpolator, time, data = read(name)
#
#     fig, ax = plt.subplots()
#     ax.plot(
#         time,
#         data,
#         label="Raw",
#     )
#     t = np.linspace(time[0], time[-1], 10000)
#     ax.plot(
#         t,
#         interpolator(t)
#     )
#     p.show_plot(
#         f"{name} vs. time",
#         xlabel="Time [s]",
#         ylabel=name,
#         dpi=400
#     )
#
#     p.qp(time, data)
#     p.qp(t, interpolator(t))

airspeed = read("airspeed")[0]
baro_alt = read("baro_alt")[0]
voltage = read("voltage")[0]
current = read("current")[0]

mass_total = 9.4

# avionics_power = 3
# prop_efficiency = 0.75

t = np.linspace(10, t_max, 50000)
dt = np.diff(t)[0]


def f(x):
    # return x
    return ndimage.gaussian_filter(x, sigma=7 / dt)


input = np.array([ 1.00788317e-07,  5.35121235e-01,  3.75497343e-01, -1.09455622e-01])

avionics_power = input[0]
pe = (
        input[1]
        + input[2] * (f(airspeed(t)) - 10)
        + input[3] * (f(current(t)) - 10)
)
prop_efficiency = np.blend(pe, 1, 0.01)

drag_power = (
        prop_efficiency * (f(voltage(t)) * f(current(t)) - avionics_power)
        - mass_total * f(airspeed(t)) * f(airspeed(t, 1))
        - mass_total * 9.81 * f(baro_alt(t, 1))
)

required_electrical_power = drag_power / prop_efficiency + avionics_power

np.random.seed(0)

x_jitter = np.random.uniform(-1, 1, len(t)) * 0.1
y_jitter = np.random.uniform(-1, 1, len(t)) * 5

fig, ax = plt.subplots()
plt.plot(
    f(airspeed(t)) + x_jitter,
    required_electrical_power + y_jitter,
    ".",
    color="k",
    alpha=0.09 / np.log(len(t)),
    markersize=3,
)
plt.plot(
    [],
    [],
    ".k",
    markersize=3,
    label="Flight Data, with total-energy\ncorrections, powertrain eff.",
)


def CD_func(CL, p):
    return (
            p["CD0"]
            + p["CD2"] * np.abs(CL - p["CLCD0"]) ** 2
            + p["CD3"] * np.abs(CL - p["CLCD0"]) ** 3
            + p["CD4"] * np.abs(CL - p["CLCD0"]) ** 4
            + p["CD5"] * np.abs(CL - p["CLCD0"]) ** 5
            + p["CD6"] * np.abs(CL - p["CLCD0"]) ** 6
    )


def model(x, p):
    q = 0.5 * 1.225 * x ** 2

    S = 1.499

    CL = mass_total * 9.81 / q / S

    CD = CD_func(CL, p)

    D = CD * q * S

    power_drag = D * x
    # power_electric = power_drag / prop_efficiency + avionics_power
    #
    # return power_electric

    pe_approx = (input[1] + input[2] * (x - 10))
    prop_eff_approx = np.blend(pe_approx, 1, 0)

    power_electric_approx = power_drag / prop_eff_approx + avionics_power

    pe = (
            input[1]
            + input[2] * (x - 10)
            + input[3] * (power_electric_approx / voltage(np.median(t)) - 10)
    )
    prop_eff = np.blend(pe, 1, 0.01)

    power_electric = power_drag / prop_eff + avionics_power

    return power_electric


fit = asb.FittedModel(
    model=model,
    x_data=f(airspeed(t))[::len(t) // 2000],
    y_data=required_electrical_power[::len(t) // 2000],
    parameter_guesses={
        "CD0"  : 0.07,
        "CLCD0": 0.5,
        "CD2"  : 0.1,
        "CD3"  : 0.03,
        "CD4"  : 0.01,
        "CD5"  : 0.003,
        "CD6"  : 0.001,
        # "CD2_max": 0.05,
        # "CL_max": 1.5,
        # "span_eff": 3.848,
    },
    parameter_bounds={
        "CD0"  : [0, 1],
        "CLCD0": [0, 2],
        "CD2"  : [0, None],
        "CD3"  : [0, None],
        "CD4"  : [0, None],
        "CD5"  : [0, None],
        "CD6"  : [0, None],
        # "CD2_max": [0, 0.5],
        # "CL_max": [0, None],
        # "span_eff": [0, None],
    },
    residual_norm_type="L1",
    verbose=False
)

print(fit.goodness_of_fit())

plot_speeds = np.linspace(0.01, 20, 1000)
plot_powers = fit(plot_speeds)

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

solar_power_input = 135

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

# ----------------------------------------------------------------------

# fig, ax = plt.subplots()
# CLs = np.linspace(0, 2, 1000)
# CDs = CD_func(CLs, fit.parameters)
# ax.plot(CDs, CLs)
# plt.xlim(left=0, right=0.2)
# plt.ylim(bottom=CLs.min(), top=CLs.max())
# p.show_plot(
#     "Solar Seaplane Drag Polar",
#     "$C_D$",
#     "$C_L$",
# )
#
# LD = CLs / CDs
