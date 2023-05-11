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
    "airspeed" : ("./data/flight3_airspeed_validated_0.csv", "calibrated_airspeed_m_s"),
    "barometer": ("./data/flight3_sensor_baro_0.csv", "pressure"),
    "baro_alt" : ("./data/flight3_vehicle_air_data_0.csv", "baro_alt_meter"),
    "voltage"  : ("./data/flight3_battery_status_1.csv", "voltage_filtered_v"),
    "current"  : ("./data/flight3_battery_status_1.csv", "current_filtered_a"),
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


def get_gof(input):

    t = np.linspace(10, t_max, 2000)
    dt = np.diff(t)[0]

    def f(x):
        return ndimage.gaussian_filter(x, sigma=6 / dt)

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

        pe_approx = (input[1] + input[2] * (x - 10))
        prop_eff_approx = np.blend(pe_approx, 1, 0.01)

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
        x_data=f(airspeed(t))[::len(t) // 500],
        y_data=required_electrical_power[::len(t) // 500],
        parameter_guesses={
            "CD0"  : 0.07,
            "CLCD0": 0.5,
            "CD2"  : 0.1,
            "CD3"  : 0.01,
            "CD4"  : 0.01,
            "CD5"  : 0.001,
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
        residual_norm_type="L2",
        verbose=False
    )

    return fit.goodness_of_fit()


x0 = np.array([ 0.        ,  0.49958197,  0.36098534, -0.1080622 ])
get_gof(x0)

from scipy import optimize

res = optimize.minimize(
    fun=lambda x: -get_gof(x),
    x0=x0,
    method="Nelder-Mead",
    bounds=[
        (0, None),
        (None, None),
        (None, None),
        (None, None),
    ],
    options=dict(
        disp=True,

    )
)
