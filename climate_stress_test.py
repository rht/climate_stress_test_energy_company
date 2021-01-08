# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + cellView="form"
#@title Climate Stress Test Energy Company
import math

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

# Hardcoded params
dg = 0.075
db = 0.15  # dg < db
initial_x = 0.1
rho_cg = 0.19
psi = 24.39  # $/tons
beta = 1
mu = 0.5
rho = 0.01
Tstart = 2020
DeltaT = 30
Ts = list(range(2021, 2021 + DeltaT))
full_Ts = [2020] + Ts
delta_t_tax = 30
t_tax = 2020 + delta_t_tax
xs0 = [min(1, initial_x) for i in range(DeltaT)]

# Scenario
scenario_description = widgets.Label(
    value='Which carbon tax scenario from Figure 1 '
          'below would you like to consider for the climate stress test?')
display(scenario_description)
scenario_list = [
    'Orderly transition',
    'Disorderly transition (late)',
    'Too little, too late transition',
    'No transition (hot house world)']
scenario = widgets.Dropdown(options=scenario_list)
display(scenario)
scenario_plot = widgets.Output()
display(scenario_plot)
with scenario_plot:
    # Orderly
    plt.plot(
        full_Ts,
        [i * 10 for i in range(len(full_Ts))],
        label=scenario_list[0],
        color=u'#2ca02c'  # green
    )
    # Disorderly
    taxes = [0]
    for t in Ts:
        if t > 2030:
            taxes.append(taxes[-1] + 35)
        else:
            taxes.append(0)
    plt.plot(
        full_Ts,
        taxes,
        label=scenario_list[1],
        color=u'#1f77b4'  # blue
    )
    # TLTL
    taxes = [0]
    for t in Ts:
        if t > 2030:
            taxes.append(taxes[-1] + 10)
        else:
            taxes.append(0)
    plt.plot(
        full_Ts,
        taxes,
        label=scenario_list[2],
        color=u'#ff7f0e'  # orange
    )
    # Hot house
    plt.plot(
        full_Ts,
        [0] * len(full_Ts),
        label=scenario_list[3],
        color=u'#d62728'  # red
    )
    plt.ylabel('USD/t $CO_2$')
    plt.xlabel('Time (years)')
    plt.title('Figure 1: Carbon tax development across scenarios')
    plt.legend()
    plt.show()

# For cost evolution visualization
montecarlo_plot = widgets.Output()
averaged_montecarlo_plot = widgets.Output()
def averaged_normal(mean, sigma, mcpaths=1000):
    return np.mean([np.random.normal(mean, sigma) for i in range(mcpaths)])

custom_mcpaths = 1
# evolve_cg() and evolve_cb() duplicate the code to calculate cost evolution
# inside calculate_utility(). They can be refactored later if necessary.
def evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial):
    omega_cg = averaged_normal(omega_hat, sigma_omega, custom_mcpaths)
    ut_greens = [averaged_normal(0, sigma_u, custom_mcpaths) for i in range(len(Ts))]
    c_greens = [cg_initial]
    for j, t in enumerate(Ts):
        ut = ut_greens[j]
        cg = c_greens[-1]
        # Wright's law
        try:
            ut_minus1 = ut_greens[-2]
        except IndexError:
            ut_minus1 = 0.0
        cg_next = cg * math.exp(-omega_cg + ut + rho_cg * ut_minus1)
        c_greens.append(cg_next)
    return c_greens

def evolve_cb(sigma_cb, cb_initial, kappa, phi_cb):
    epsilon_cb = [averaged_normal(0, sigma_cb, custom_mcpaths) for i in range(len(Ts))]
    c_browns = [cb_initial]
    for j, t in enumerate(Ts):
        cb = c_browns[-1]
        m_cb = kappa / (1 - phi_cb)
        cb_next = cb * math.exp((1 - phi_cb) * (m_cb - math.log(cb)) + epsilon_cb[j])
        c_browns.append(cb_next)
    return c_browns

# Brown params
params_oil = dict(
    kappa = 0.342,
    phi_cb = 0.846,
    sigma_cb = 0.252,
    cb_initial = 11.7,  # $/GJ
    # See section "CO2 emission" on notes.md
    # chi is tons emission of CO2 per GJ
    chi = 0.07547,  # tons/GJ
    alpha_b = 6.83  # $/GJ
)
params_coal = dict(
    kappa = 0.035,
    phi_cb = 0.95,
    sigma_cb = 0.090,
    cb_initial = 2.18,  # $/GJ
    chi = 0.1024,  # tons/GJ
    alpha_b = 1.61  # $/GJ
)
params_gas = dict(
    kappa = 0.21,
    phi_cb = 0.82,
    sigma_cb = 0.24,
    cb_initial = 3.0,  # $/GJ
    chi = 0.05545,
    alpha_b = 4.21  # $/GJ (DOE only)
)
display(widgets.Label(
    value='What type of brown energy company would you like to subject to a climate stress test?'
))
dropdown_brown = widgets.Dropdown(options=['oil', 'coal', 'gas'])
display(dropdown_brown)
brown_params = widgets.Output()
brown_params.value = params_oil  # default
display(brown_params)
with brown_params:
    display(brown_params.value)

# Green params
params_solar = dict(
    omega_hat=0.303,
    sigma_omega=0.047,
    sigma_eta=0.093,
    cg_initial=70 / 3.6,
    alpha_g=14 / 8760 / 0.0036
)
params_wind = dict(
    omega_hat=0.158,
    sigma_omega=0.045,
    sigma_eta=0.103,
    cg_initial=55 / 3.6,
    alpha_g = (30 + 99.5) / 2 / 8760 / 0.0036
)
display(widgets.Label(
    value='What type of green energy does the brown energy company invest (if it decides to invest in green) in the climate stress test?'
))
dropdown_green = widgets.Dropdown(options=['solar', 'wind'])
display(dropdown_green)
green_params = widgets.Output()
green_params.value = params_solar  # default
display(green_params)
with green_params:
    display(green_params.value)

# Cost evolution, brown params, and green params event handler
def plot_cost_evolution():
    np.random.seed(1337)
    # brown
    brown_tech = dropdown_brown.value
    sigma_cb = brown_params.value['sigma_cb']
    cb_initial = brown_params.value['cb_initial']
    kappa = brown_params.value['kappa']
    phi_cb = brown_params.value['phi_cb']
    # green
    green_tech = dropdown_green.value
    omega_hat = green_params.value['omega_hat']
    sigma_omega = green_params.value['sigma_omega']
    sigma_u = green_params.value['sigma_eta'] / np.sqrt(1 + rho_cg ** 2)
    cg_initial = green_params.value['cg_initial']
    all_c_browns = []
    all_c_greens = []
    montecarlo_plot.clear_output()
    averaged_montecarlo_plot.clear_output()
    with montecarlo_plot:
        for i in range(5):
            c_browns = np.array(evolve_cb(sigma_cb, cb_initial, kappa, phi_cb))
            all_c_browns.append(c_browns)
            plt.plot(full_Ts, c_browns, label=f'{brown_tech} {i + 1}')
        for i in range(5):
            c_greens = np.array(evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial))
            all_c_greens.append(c_greens)
            plt.plot(full_Ts, c_greens, label=f'{green_tech} {i + 1}')
        plt.xlabel('Time (years)')
        plt.ylabel('Cost ($/GJ)')
        plt.title('Figure 2: Evolution of Unit Cost of Energy')
        plt.legend()
        plt.show()
    with averaged_montecarlo_plot:
        ave_c_browns = np.mean(np.array(all_c_browns), axis=0)
        plt.plot(full_Ts, ave_c_browns, label=brown_tech)
        ave_c_greens = np.mean(np.array(all_c_greens), axis=0)
        plt.plot(full_Ts, ave_c_greens, label=green_tech)
        plt.xlabel('Time (years)')
        plt.ylabel('Cost ($/GJ)')
        plt.title('Figure 3: Averaged evolution of energy cost')
        plt.legend()
        plt.show()

def dropdown_brown_eventhandler(change):
    brown_params.clear_output()
    if change.new == 'oil':
        brown_params.value = params_oil
    elif change.new == 'coal':
        brown_params.value = params_coal
    else:  # gas
        brown_params.value = params_gas
    with brown_params:
        display(brown_params.value)
    plot_cost_evolution()
dropdown_brown.observe(dropdown_brown_eventhandler, names='value')

def dropdown_green_eventhandler(change):
    green_params.clear_output()
    if change.new == 'solar':
        green_params.value = params_solar
    else:  # wind
        green_params.value = params_wind
    with green_params:
        display(green_params.value)
    plot_cost_evolution()
dropdown_green.observe(dropdown_green_eventhandler, names='value')

# Brown energy percentage
display(widgets.Label(
    value='Does the brown energy company that you would like to '
          'subject to the climate stress test already have a small '
          'green portfolio or is it a fully brown company?'

))
style = {'description_width': 'initial'}
brown_energy_percentage = widgets.IntSlider(
    min=0,
    max=100,
    step=25,
    description='Brown energy percentage (%):',
    value=75,
    style=style,
    layout=widgets.Layout(width='50%')
)
display(brown_energy_percentage)

# alpha_b multiplier
display(widgets.Label(
    value='How much does the brown energy company expect its '
          'cost of maintaining its stock of brown energy to be '
))
display(widgets.Label(
    value='in the future (relative to its current maintenance '
          'cost of brown energy $\\alpha_b^{default}$)?'
))
alpha_b_multiplier = widgets.IntSlider(
    min=0,
    max=150,  # TODO find the max
    step=25,
    description='Maintenance cost of brown energy relative to $\\alpha_b^{default}$ in percentages (%):',
    value=100,
    style=style,
    layout=widgets.Layout(width='80%')
)
display(alpha_b_multiplier)
# TODO use alpha_b_multiplier

# alpha_g multiplier
display(widgets.Label(
    value='How much does the brown energy company expect its '
          'cost of maintaining its stocks of green energy to be '
))
display(widgets.Label(
    value='in the future (relative to its current maintenance '
          'cost of green energy $\\alpha_g^{default}$)?'
))
alpha_g_multiplier = widgets.IntSlider(
    min=0,
    max=150,  # TODO find the max
    step=25,
    description='Maintenance cost of green energy relative to $\\alpha_g^{default}$ in percentages (%):',
    value=100,
    style=style,
    layout=widgets.Layout(width='80%')
)
display(alpha_g_multiplier)
# TODO use alpha_g_multiplier

# omega_hat
display(widgets.Label(
    value='What expectations does the energy company have '
          'regarding the rate at which investing in green energy '
))
display(widgets.Label(
    value='becomes cheaper over time? Please select the technology '
          'scenario on the slider by specifying the rate of '
))
display(widgets.Label(
    value='decay $\hat{\omega}$ of the initial investment '
          'costs associated to your choice of green energy '
))
display(widgets.Label(
    value='$c_g$ (relative to the current decay value '
          '$\hat{\omega}^{default}$)'
))
omega_hat_multiplier = widgets.IntSlider(
    min=0,
    max=150,  # TODO find the max
    step=25,
    description='Investment cost of green energy relative to $c_g^{default}$ in percentages (%):',
    value=100,
    style=style,
    layout=widgets.Layout(width='80%')
)
display(omega_hat_multiplier)
# TODO use omega_hat_multiplier

# Display cost evolution here
display(montecarlo_plot)
display(averaged_montecarlo_plot)
plot_cost_evolution()

# Model
def calculate_cost_g(cg, x, delta_E, Eg):
    return cg * x * delta_E + green_params.value['alpha_g'] * (Eg ** beta)

def calculate_cost_b(cb, tax, x, delta_E, Eb):
    return (
        (cb + tax) * (1 - x) * delta_E +
        brown_params.value['alpha_b'] * (Eb ** beta) +
        tax * Eb
    )

def calculate_numerator(tau, x, delta_E, Eg, Eb, cg, cb, tax, p0):
    # Discounted profit associated with year t + tau
    cost_g = calculate_cost_g(cg, x, delta_E, Eg)
    cost_b = calculate_cost_b(cb, tax, x, delta_E, Eb)
    return math.exp(-rho * tau) * (
        max((p0 * Eg - cost_g + p0 * Eb - cost_b), 0)
    )

def calculate_utility(omega_cg, ut_greens, epsilon_cb, t_tax, plot_Evst=False, plot_tax=False, plot_cost=False):
    def _calc_U(xs):
        Us = []
        for i in range(1):
            full_xs = [initial_x] + list(xs)
            # Initialize first element of all the time series at t = 2020
            brown_fraction = brown_energy_percentage.value / 100
            # Time series of green energy
            E_greens = [(1 - brown_fraction) * 200]  # GJ/yr, useful energy at t0
            # Time series of cost of green energy
            c_greens = [green_params.value['cg_initial']]
            E_browns = [brown_fraction * 200]  # GJ/yr, useful energy at t0
            # Time series of cost of brown energy
            c_browns = [brown_params.value['cb_initial']]
            E_total = E_greens[0] + E_browns[0]
            # Time series of total depreciation of energy
            delta_Es = [dg * E_greens[0] + db * E_browns[0]]
            tax = 0.0
            taxes = [tax]

            # There is no need to discount the initial
            # denominator term because tau is 0 anyway.
            denominators = [(c_greens[0] * full_xs[0] + c_browns[0] * (1 - full_xs[0])) * delta_Es[0]]

            price0 = (1 + mu) * (
                calculate_cost_g(c_greens[0], full_xs[0], delta_Es[0], E_greens[0]) +
                calculate_cost_b(c_browns[0], tax, full_xs[0], delta_Es[0], E_browns[0])
            ) / E_total
            numerators = [calculate_numerator(0, full_xs[0], delta_Es[0], E_greens[0], E_browns[0], c_greens[0], c_browns[0], tax, price0)]

            assert len(full_xs) == (len(Ts) + 1), (len(full_xs), len(Ts) + 1)

            for j, t in enumerate(Ts):
                Eg = E_greens[-1]
                cg = c_greens[-1]
                ut = ut_greens[j]
                Eb = E_browns[-1]
                cb = c_browns[-1]
                delta_E = delta_Es[-1]
                x = full_xs[j + 1]

                assert abs(E_total - (Eg + Eb)) < 1e-9
                # Doyne equation 18
                E_green_next = Eg * (1 - dg) + x * delta_E
                # Doyne equation 19
                E_brown_next = Eb * (1 - db) + (1 - x) * delta_E
                delta_E_next = dg * E_green_next + db * E_brown_next

                if scenario.value == 'DEFAULT':
                    if t >= t_tax:
                        tax = psi * brown_params.value['chi']
                    else:
                        tax = 0
                elif scenario.value == 'Orderly transition':
                    # Allen 2020 page 11
                    # First scenario of NGFS (orderly).
                    # "That price increases by about $10/ton of CO2 per year until 2050"
                    if t <= 2050:
                        tax += 10.0 * brown_params.value['chi']
                elif scenario.value == 'Disorderly transition (late)':
                    # Allen 2020 page 11
                    # Second scenario of NGFS (disorderly).
                    # "In 2030, the carbon price is abruptly revised and
                    # increases by about $40/ton of CO2 per year afterwards to
                    # keep on track with climate commitments."
                    # We use NGFS paper's number which is $35/ton
                    if t > 2030:
                        tax += 35.0 * brown_params.value['chi']
                elif scenario.value == 'Disorderly transition (sudden)':
                    if t > 2025:
                        tax += 36.0 * brown_params.value['chi']
                elif scenario.value == 'No transition (hot house world)':
                    # Third scenario of NGFS (hot house).
                    pass
                elif scenario.value == 'Too little, too late transition':
                    if t > 2030:
                        tax += 10.0 * brown_params.value['chi']

                # Wright's law
                try:
                    ut_minus1 = ut_greens[-2]
                except IndexError:
                    ut_minus1 = 0
                cg_next = cg * math.exp(-omega_cg + ut + rho_cg * ut_minus1)
                # AR(1)
                phi_cb = brown_params.value['phi_cb']
                kappa = brown_params.value['kappa']
                m_cb = kappa / (1 - phi_cb)
                cb_next = cb * math.exp((1 - phi_cb) * (m_cb - math.log(cb)) + epsilon_cb[j])

                E_greens.append(E_green_next)
                c_greens.append(cg_next)
                E_browns.append(E_brown_next)
                c_browns.append(cb_next)
                delta_Es.append(delta_E_next)
                taxes.append(tax)
                numerator = calculate_numerator(t - Tstart, x, delta_E_next, E_green_next, E_brown_next, cg_next, cb_next, tax, price0)
                numerators.append(numerator)
                denominator = math.exp(-rho * (t - Tstart)) * (cg_next * x + (cb_next + tax) * (1 - x)) * delta_E_next
                denominators.append(denominator)
            # Rupert short paper equation 12
            # Since mu is a scale, it doesn't affect the final result, but we
            # set it anyway.
            U = math.log(mu * sum(numerators) / sum(denominators))
            Us.append(U)

        mean_U = np.mean(Us)
        # Reverse the sign because we only have `minimize` function
        out = -mean_U
        if not plot_Evst:
            return out
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(right=0.77)
        ax.stackplot(full_Ts, [E_browns, E_greens], labels=[f'Brown ({dropdown_brown.value})', f'Green ({dropdown_green.value})'], colors=['brown', 'green'])
        ax.set_title(scenario.value)
        ax.set_ylabel('Energy (GJ)')
        ax.set_xlabel('Time (years)')
        ax.set_ylim(0, int(1.01 * (E_browns[0] + E_greens[0])))

        ax2 = ax.twinx()
        ax2.plot(full_Ts, 100 * np.array([initial_x] + xs0),
                 label='Initial guess', color='gray', linewidth=2.0)
        ax2.plot(full_Ts, 100 * np.array([initial_x] + list(xs)),
                 label='Optimized', color='black', linewidth=2.0)
        ax2.set_ylabel("Investment in green energy x%")
        ax2.set_ylim(0, 101)
        fig.legend(loc=7)
        return out
    return _calc_U

def do_optimize(fn, xs0):
    method = 'SLSQP'
    bounds = Bounds([0.0 for i in range(DeltaT)], [1.0 for i in range(DeltaT)])
    result = minimize(fn, xs0, bounds=bounds, method=method)
    return result

# Plot
simulation_plot = widgets.Output()

# Run button
btn = widgets.Button(description='Run')
display(btn)
def btn_eventhandler(obj):
    simulation_plot.clear_output()
    with simulation_plot:
        # For deterministic result
        np.random.seed(1337)
        omega_cg = averaged_normal(green_params.value['omega_hat'], green_params.value['sigma_omega'])
        sigma_u = green_params.value['sigma_eta'] / np.sqrt(1 + rho_cg ** 2)
        ut_greens = [averaged_normal(0, sigma_u) for i in range(len(Ts))]
        epsilon_cb = [averaged_normal(0, brown_params.value['sigma_cb']) for i in range(len(Ts))]
        print(scenario.value)
        fn = calculate_utility(omega_cg, ut_greens, epsilon_cb, t_tax)
        result = do_optimize(fn, xs0)

        fn_with_plot = calculate_utility(omega_cg, ut_greens, epsilon_cb, t_tax, plot_Evst=True)
        fn_with_plot(result.x)
        plt.show()
btn.on_click(btn_eventhandler)

display(simulation_plot)
