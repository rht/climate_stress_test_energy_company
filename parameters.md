|name|description|value|
|----|-----|------|
dg | depreciation rate (green) | 0.05 |
db | depreciation rate (brown) | 0.06 |
rho_cg | autocorrelation parameter | 0.19 |
sigma_u | noise shock | sigma_eta / np.sqrt(1 + rho_cg ** 2) |
beta | economy of scale | 1 |
rho | discount rate | 0.01 |
Tstart | starting year | 2020 |
DeltaT | time span (year) | 30 |
total_energy (GJ) | total energy | 200,000,000 |
initial_x | initial policy | 0.1 |
mu | price markup | 0.5 |


|name|description|solar|wind|
|-|-----|----|------|
omega_hat | decay rate | 0.303 | 0.158 |
sigma_omega | noise of decay rate | 0.047 | 0.045 |
sigma_eta | N/A | 0.093 | 0.103 |
cg_initial ($/GJ) | investment cost (in 2020) | 70 / 3.6 | 55 / 3.6 |
alpha_g ($/GJ) | operating cost | 0.44393708777270424 | 2.053209030948757 |

|name|description|oil|coal|gas|
|-|---|----|---|------|
kappa | N/A | 0.342 | 0.035 | 0.21 |
phi_cb | N/A | 0.846 | 0.95 | 0.82 |
sigma_cb | cost noise | 0.252 | 0.090 | 0.24 |
cb_initial ($/GJ) | investment cost (in 2020) | 11.7 | 2.18 | 3.0 |
chi (tons/GJ) | (tons of CO2 emitted)/GJ | 0.07547 | 0.1024 | 0.05545 |
alpha_b ($/GJ) | operating cost | 6.83 | 1.61 | 4.21 |

