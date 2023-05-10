import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import pandas as pd
from scipy import signal, ndimage

df = pd.read_csv(
    "./AABV_flight3.csv"
)

dt = 0.01

time = df["Time"] * dt
groundspeed = df["Ground_Speed"] # GPS
airspeed = df["Indicated_Airspeed"] # Pitot
baro_alt = df["Barometer_Altitude"] # Barometer
gps_alt = df["GPS_Altitude"] / 1e3 # GPS
voltage = df["Battery_Voltage"]
current = df["Battery_Current"]

airspeed = ndimage.gaussian_filter(airspeed, sigma=1 / dt)

altitude = baro_alt
# altitude = gps_alt

atmo = asb.Atmosphere(altitude=baro_alt[0])

mask = airspeed > 6

mass_total = 9.57

total_energy = (
        0.5 * mass_total * airspeed ** 2 +
        mass_total * 9.81 * altitude
)

total_power = np.gradient(
    ndimage.gaussian_filter(
        total_energy,
        sigma = 1 / dt,
    )
) / dt

electrical_power = voltage * current
avionics_power = 3

prop_efficiency = 0.7

prop_air_power = (electrical_power - avionics_power) * prop_efficiency

drag_power = total_power - prop_air_power

required_electrical_power = -drag_power / prop_efficiency + avionics_power

required_electrical_power = ndimage.gaussian_filter(
    required_electrical_power,
    sigma=1 / dt,
)

# fig, ax = plt.subplots()
# plt.plot(
#     time[mask],
#     # groundspeed[mask]
#     airspeed[mask]
# )
# p.show_plot()


fig, ax = plt.subplots()

x_jitter = np.random.uniform(-1, 1, len(airspeed[mask])) * 0.2
y_jitter = np.random.uniform(-1, 1, len(airspeed[mask])) * 10


plt.plot(
    airspeed[mask] + x_jitter,
    # groundspeed[mask],
    # -input_power[mask],
    # electrical_power[mask],
    # total_power[mask],
    # total_power[mask] - prop_air_power[mask],
    required_electrical_power[mask] + y_jitter,
    ".k",
    alpha=0.025,
    markersize=3,
)
plt.plot(
    [],
    [],
    ".k",
    markersize=3,
    label="Data, with total-energy\ncorrection, denoising, powertrain eff.",
)

def model(x, p):
    q = 0.5 * 1.225 * x ** 2

    S = 1.499

    CL = mass_total * 9.81 / q / S

    CD = (
        p["CD0"]
        + p["CD2"] * (CL - p["CLCD0"]) ** 2
        # + p["CD2_max"] * np.maximum(0, CL - p["CL_max"]) ** 2
        + p["CD4"] * (CL - p["CLCD0"]) ** 4
    )

    D = CD * q * S

    power_drag = D * x
    power_electric = power_drag / prop_efficiency + avionics_power

    return power_electric

fit = asb.FittedModel(
    model=model,
    x_data=airspeed[mask][::10],
    y_data=required_electrical_power[mask][::10],
    parameter_guesses={
        "CD0": 0.02,
        "CLCD0": 0.5,
        "CD2": 0.02,
        "CD4": 0.001,
        # "CD2_max": 0.05,
        # "CL_max": 1.5,
        # "span_eff": 3.848,
    },
    parameter_bounds={
        "CD0": [0, 1],
        "CLCD0": [0, 2],
        "CD2": [0, None],
        "CD4": [0, None],
        # "CD2_max": [0, 0.5],
        # "CL_max": [0, None],
        # "span_eff": [0, None],
    },
    residual_norm_type="L1"
)

plot_speeds = np.linspace(0.01, 20, 1000)
plot_powers = fit(plot_speeds)

_line, = plt.plot(
    plot_speeds,
    plot_powers,
    linewidth=2.5,
    color=p.adjust_lightness("darkred", 1.2),
    alpha=0.7,
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
    alpha=0.25,
)




plt.xlim(6, 20)
plt.ylim(-100, 600)

p.set_ticks(1, 0.25, 100, 25)

plt.legend(
    loc="lower right",
)
p.show_plot(
    "Solar Seaplane Power Polar",
    "Cruise Airspeed [m/s]",
    "Required Electrical Power for Cruise [W]",
    legend=False
)