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
#@title Illustrative Exercise: Stress Testing an Energy Company

import math

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

# Hardcoded params
dg = 0.05  # DEVIATION
db = 0.06  # dg < db  # DEVIATION
initial_x = 0.1
rho_cg = 0.19
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
total_energy = 200_000_000  # GJ
green_tech = 'solar'
brown_energy_percentage = 75

# Brown params
display(widgets.HTML("<h1>Select type of brown energy company:</h1>"))
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
    value='1. What type of brown energy company would you like to subject to a climate stress test?'
))
dropdown_brown = widgets.Dropdown(options=['oil', 'coal', 'gas'], value='coal')
display(dropdown_brown)
display(widgets.HTML(
    'If the brown company decides to invest in green energy, we assume it invests in solar power.'
))
brown_params = widgets.Output()
brown_params.value = params_coal  # default
# display(brown_params)
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
green_params = widgets.Output()
green_params.value = params_solar  # default
# display(green_params)
with green_params:
    display(green_params.value)

# Empty line for a breather
display(widgets.Label('\n\n'))
# Scenario
display(widgets.HTML("<h1>Select transition scenario:</h1>"))
display(widgets.HTML("2. Which carbon tax scenario from Figure 1 below would you like to consider for the climate stress test?"))
scenario_list = [
    'Orderly transition',
    'Disorderly transition (late)',
    'Too little, too late transition',
    'No transition (hot house world)']
scenario = widgets.Dropdown(options=scenario_list, value='No transition (hot house world)')
display(scenario)
scenario_plot = widgets.Output()
display(scenario_plot)
display(widgets.HTML(
    "We assume the transition scenario consists solely of the carbon tax scenario. Figure 1 shows by how many dollars the carbon tax per ton of CO2 emissions increases per year."
))
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
averaged_montecarlo_plot = widgets.Output()
MCPATHS = 1000
def evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial):
    # Rupert appendix p38
    # We generate the cost evolution for every monte carlo
    # path, and then we average the path for every point in
    # time.
    c_greens_all = []
    for n in range(MCPATHS):
        omega_cg = np.random.normal(omega_hat, sigma_omega)
        ut_greens = np.random.normal(0, sigma_u, len(Ts))
        c_greens = [cg_initial]
        for j in range(len(Ts)):
            ut = ut_greens[j]
            cg = c_greens[-1]
            # Wright's law
            if (j - 1) == -1:
                ut_minus1 = 0
            else:
                ut_minus1 = ut_greens[j - 1]
            cg_next = cg * math.exp(-omega_cg + ut + rho_cg * ut_minus1)
            c_greens.append(cg_next)
        c_greens_all.append(c_greens)
    c_greens_ave = np.mean(c_greens_all, axis=0)
    return c_greens_ave

def evolve_cb(sigma_cb, cb_initial, kappa, phi_cb):
    c_browns_all = []
    for n in range(MCPATHS):
        epsilon_cb = np.random.normal(0, sigma_cb, len(Ts))
        c_browns = [cb_initial]
        for j in range(len(Ts)):
            cb = c_browns[-1]
            # AR(1)
            # Equation 25 of Rupert appendix
            m_cb = kappa / (1 - phi_cb)
            cb_next = cb * math.exp((1 - phi_cb) * (m_cb - math.log(cb)) + epsilon_cb[j])
            c_browns.append(cb_next)
        c_browns_all.append(c_browns)
    c_browns_ave = np.mean(c_browns_all, axis=0)
    return c_browns_ave

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
    omega_hat = green_params.value['omega_hat'] * omega_hat_multiplier.value / 100
    sigma_omega = green_params.value['sigma_omega']
    sigma_u = green_params.value['sigma_eta'] / np.sqrt(1 + rho_cg ** 2)
    cg_initial = green_params.value['cg_initial']
    averaged_montecarlo_plot.clear_output()
    with averaged_montecarlo_plot:
        ave_c_browns = evolve_cb(sigma_cb, cb_initial, kappa, phi_cb)
        plt.plot(full_Ts, ave_c_browns, label=brown_tech)
        ave_c_greens = evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial)
        plt.plot(full_Ts, ave_c_greens, label=green_tech)
        plt.xlabel('Time (years)')
        plt.ylabel('Cost ($/GJ)')
        plt.title('Figure 2: Averaged evolution of energy cost')
        plt.legend()
        plt.show()

display(widgets.HTML("<h1>Select technology scenario:</h1>"))

# omega_hat
# Empty line for a breather
display(widgets.Label('\n\n'))
display(widgets.HTMLMath(
    '''
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
    '''
    '<script type="text/javascript" id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>'
    '3. What expectations does the energy company have '
    'regarding the rate at which investing in green energy '
    'becomes cheaper over time? Please select the rate of '
    'decay $\hat{\omega}$ of the initial investment costs $c_g$ '
    'relative to its default value $\hat{\omega}^{default}$.'
))
style = {'description_width': 'initial'}
omega_hat_multiplier = widgets.IntSlider(
    min=10,
    max=130,
    step=10,
    description='Rate of decay relative to default (%):',
    value=100,
    style=style,
    layout=widgets.Layout(width='60%')
)
display(omega_hat_multiplier)

# Display cost evolution here
display(averaged_montecarlo_plot)
plot_cost_evolution()

# Model
def calculate_cost_g(cg, x, delta_E, Eg):
    alpha_g = green_params.value['alpha_g']
    return cg * x * delta_E + alpha_g * (Eg ** beta)

def calculate_cost_b(cb, tax, x, delta_E, Eb):
    alpha_b = brown_params.value['alpha_b']
    return (
        (cb + tax) * (1 - x) * delta_E +
        alpha_b * (Eb ** beta) +
        tax * Eb
    )

def calculate_numerator(tau, x, delta_E, Eg, Eb, cg, cb, tax, p0):
    # Discounted profit associated with year t + tau
    cost_g = calculate_cost_g(cg, x, delta_E, Eg)
    cost_b = calculate_cost_b(cb, tax, x, delta_E, Eb)
    return math.exp(-rho * tau) * (
        max((p0 * Eg - cost_g + p0 * Eb - cost_b), 0)
    )

def calculate_utility(c_greens, c_browns, t_tax, plot_Evst=False, initial=False):
    def _calc_U(xs):
        Us = []
        Vs = []
        for i in range(1):
            full_xs = [initial_x] + list(xs)
            # Initialize first element of all the time series at t = 2020
            brown_fraction = brown_energy_percentage / 100
            # Time series of green energy
            E_greens = [(1 - brown_fraction) * total_energy]  # GJ/yr, useful energy at t0
            E_browns = [brown_fraction * total_energy]  # GJ/yr, useful energy at t0
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
                cg = c_greens[j]
                Eb = E_browns[-1]
                cb = c_browns[j]
                delta_E = delta_Es[-1]
                x = full_xs[j + 1]

                assert abs(E_total - (Eg + Eb)) / E_total < 1e-9
                # Doyne equation 18
                E_green_next = Eg * (1 - dg) + x * delta_E
                # Doyne equation 19
                E_brown_next = Eb * (1 - db) + (1 - x) * delta_E
                delta_E_next = dg * E_green_next + db * E_brown_next

                if scenario.value == 'Orderly transition':
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

                cg_next = c_greens[j + 1]
                cb_next = c_browns[j + 1]

                E_greens.append(E_green_next)
                E_browns.append(E_brown_next)
                delta_Es.append(delta_E_next)
                taxes.append(tax)
                numerator = calculate_numerator(t - Tstart, x, delta_E_next, E_green_next, E_brown_next, cg_next, cb_next, tax, price0)
                numerators.append(numerator)
                denominator = math.exp(-rho * (t - Tstart)) * (cg_next * x + (cb_next + tax) * (1 - x)) * delta_E_next
                denominators.append(denominator)
            sum_numerators = sum(numerators)
            U = math.log(sum_numerators / sum(denominators))
            Us.append(U)
            Vs.append(sum_numerators)

        mean_U = np.mean(Us)
        # Reverse the sign because we only have `minimize` function
        out = -mean_U
        if not plot_Evst:
            return out

        # Else plot E vs t
        # First, print out the value of numerators
        print('$', round(np.mean(Vs) / 1000_000_000, 2), 'billion')

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(right=0.77)
        ax.stackplot(full_Ts, [E_browns, E_greens], labels=[f'Brown ({dropdown_brown.value})', f'Green ({green_tech})'], colors=['tab:brown', 'tab:green'])
        ax.set_ylabel('Energy (GJ)')
        ax.set_xlabel('Time (years)')
        ax.set_ylim(0, int(1.01 * (E_browns[0] + E_greens[0])))

        ax2 = ax.twinx()
        if not initial:
            ax2.plot(full_Ts, 100 * np.array([initial_x] + xs0),
                     label='Initial guess', color='gray', linewidth=2.0)
            ax2.plot(full_Ts, 100 * np.array([initial_x] + list(xs)),
                     label='Optimized', color='black', linewidth=2.0)
        else:
            ax2.plot(full_Ts, 100 * np.array([initial_x] + xs0),
                     label='Current', color='black', linewidth=2.0)
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
display(widgets.HTML("<h1>Press run:</h1>"))
display(widgets.HTML(
    'To generate the outputs of the climate stress test of your '
    'energy company given your selected transition and technology scenario, '
    'press "Run".'
))
btn = widgets.Button(description='Run')
display(btn)

# Event handlers
def dropdown_brown_eventhandler(change):
    simulation_plot.clear_output()
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

scenario.observe(lambda x: simulation_plot.clear_output(), names='value')

def omega_hat_multiplier_eventhandler(change):
    simulation_plot.clear_output()
    plot_cost_evolution()
omega_hat_multiplier.observe(omega_hat_multiplier_eventhandler, names='value')

def btn_eventhandler(obj):
    simulation_plot.clear_output()
    with simulation_plot:
        # For deterministic result
        np.random.seed(1337)
        omega_hat = green_params.value['omega_hat'] * omega_hat_multiplier.value / 100
        sigma_omega = green_params.value['sigma_omega']
        sigma_u = green_params.value['sigma_eta'] / np.sqrt(1 + rho_cg ** 2)
        cg_initial = green_params.value['cg_initial']

        sigma_cb = brown_params.value['sigma_cb']
        cb_initial = brown_params.value['cb_initial']
        kappa = brown_params.value['kappa']
        phi_cb = brown_params.value['phi_cb']

        c_greens = evolve_cg(omega_hat, sigma_omega, sigma_u, cg_initial)
        c_browns = evolve_cb(sigma_cb, cb_initial, kappa, phi_cb)

        fn = calculate_utility(c_greens, c_browns, t_tax)
        result = do_optimize(fn, xs0)

        display(widgets.HTML(
            '<b>Output 1:</b> Value of the energy company given its current '
            'business strategy of directing 10% of its investments towards green energy projects:'
        ))
        fn_with_plot_initial = calculate_utility(c_greens, c_browns, t_tax, plot_Evst=True, initial=True)
        fn_with_plot_initial(xs0)
        plt.title('Figure 3: ' + scenario.value)
        display(widgets.HTML(
            '<b>Output 2:</b> Figure 3 shows the portfolio of the energy '
            'company over time given its current business strategy of directing 10% of its '
            'investments towards green energy projects:'
        ))
        plt.show()
        display(widgets.HTML(
            "* The 'current' (black line) represents the investment percentage in green energy under the <b>current business model</b>. This percentage is assumed to be 10%."
        ))

        display(widgets.HTML(
            '<b>Output 3:</b> Value of the energy company given its optimally '
            'adapted business strategy:'
        ))
        fn_with_plot = calculate_utility(c_greens, c_browns, t_tax, plot_Evst=True)
        fn_with_plot(result.x)
        plt.title('Figure 4: ' + scenario.value)
        display(widgets.HTML(
            '<b>Output 4:</b> Figure 4 shows the energy company transition '
            'towards a green business model (if at all) given its optimally adapted business strategy:'
        ))
        plt.show()
        display(widgets.HTML(
            "* The 'optimized' (black line) represents the investment percentage in green energy under the <b>optimally adapted business model.</b>"
            "<br> ** The 'initial guess' (grey line) represents the initial guess that is provided to the optimization algorithm regarding the optimal investment % in green energy."
        ))
btn.on_click(btn_eventhandler)

display(simulation_plot)
