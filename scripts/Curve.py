import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
# Ensure all pressures are in GPa
P = np.array([0.0001, 0.000101325, 1, 5, 10, 15, 50, 100])        # GPa
P_96 = np.array([0.0001, 0.000101325, 1, 5, 10, 15, 50, 100])     # GPa
P_32 = np.array([0.000101325, 1*1.01325, 5*1.01325, 10*1.01325, 15*1.01325, 30*1.01325, 50*1.01325, 100*1.01325]) # Convert to GPa
P_108 = np.array([0.0001, 0.000101325, 1, 5, 10, 15, 50, 100])    # GPa
P_256 = np.array([1, 5, 10, 15, 50, 100])                         # GPa
P_150 = np.array([5, 50, 100])                                    # GPa
P_216 = np.array([5, 10, 15, 50, 100])                            # GPa

# Temperatures (K)
T_plus_27 = np.array([123.40438499160605, 123.58013615667511, 391.3026613995936, 1031.1395878787039, 1633.574131835625, 2169.0002733751667, 5505.018041488894, 9792.651503983141])
T_plus_96 = np.array([105.30338182419992, 105.0495642618424, 302.4078411337586, 775.8459979106822, 1219.310102458083, 1625.9582371200256, 4073.016427186415, 7457.843685180374])
T_plus_32 = np.array([110,330,855,1340,1797.2935884104504, 2974.135686685555, 4483.105122715527, 8200.851335135036])
T_plus_108 = np.array([106.53812872727019, 107.14131211275075, 302.45685871342874, 782.2604696609754, 1234.3227450502068, 1616, 4141, 7515])
T_plus_256 = np.array([247, 690, 942, 1504, 3411, 5556])
T_plus_150 = np.array([430, 3350, 5090])
T_plus_216 = np.array([535, 1359, 1550, 3275, 4475])

# Calculate T_M from T_plus
T_M_factor = (1 + np.log(2**(1/3)))
T_M_27 = T_plus_27 / T_M_factor
T_M_96 = T_plus_96 / T_M_factor
T_M_32 = T_plus_32 / T_M_factor
T_M_108 = T_plus_108 / T_M_factor
T_M_256 = T_plus_256 / T_M_factor
T_M_150 = T_plus_150 / T_M_factor
T_M_216 = T_plus_216 / T_M_factor

# Define the Kechin function
def Kechin_TM_P(x, T_0, a, b, c):
    return T_0 * (1 + x / a) ** b * np.exp(-c * x)

# Prepare pressure range for fitted curves
P_fit = np.linspace(0.0001, 100, 1000)

# Function to perform safe curve fitting
def safe_curve_fit(func, xdata, ydata, p0):
    if len(xdata) >= len(p0):
        try:
            popt, _ = curve_fit(func, xdata, ydata, p0=p0)
            return popt
        except RuntimeError:
            return None
    else:
        return None

# Perform curve fitting for each dataset
# Rhombic cells
popt_27 = safe_curve_fit(Kechin_TM_P, P, T_M_27, p0=[80.4, .201, 1/1.552, .634e-6])
popt_96 = safe_curve_fit(Kechin_TM_P, P_96, T_M_96, p0=[80.4, .201, 1/1.552, .634e-6])
popt_150 = safe_curve_fit(Kechin_TM_P, P_150, T_M_150, p0=[80.4, .201, 1/1.552, .634e-6])

# Cubic cells
popt_32 = safe_curve_fit(Kechin_TM_P, P_32, T_M_32, p0=[80.4, .201, 1/1.552, .634e-6])
popt_108 = safe_curve_fit(Kechin_TM_P, P_108, T_M_108, p0=[80.4, .201, 1/1.552, .634e-6])
popt_256 = safe_curve_fit(Kechin_TM_P, P_256, T_M_256, p0=[80.4, .201, 1/1.552, .634e-6])
popt_216 = safe_curve_fit(Kechin_TM_P, P_216, T_M_216, p0=[80.4, .201, 1/1.552, .634e-6])

# Calculate fitted curves where fitting was successful
if popt_27 is not None:
    T_M_27_fit = Kechin_TM_P(P_fit, *popt_27)
if popt_96 is not None:
    T_M_96_fit = Kechin_TM_P(P_fit, *popt_96)
if popt_150 is not None:
    T_M_150_fit = Kechin_TM_P(P_fit, *popt_150)
if popt_32 is not None:
    T_M_32_fit = Kechin_TM_P(P_fit, *popt_32)
if popt_108 is not None:
    T_M_108_fit = Kechin_TM_P(P_fit, *popt_108)
if popt_256 is not None:
    T_M_256_fit = Kechin_TM_P(P_fit, *popt_256)
if popt_216 is not None:
    T_M_216_fit = Kechin_TM_P(P_fit, *popt_216)

# Experimental data
P_exp = np.array([64,65,66,70,74,77])
T_M_exp = np.array([2800, 3000, 2900, 3050, 3000, 3150])
T_M_exp_error = np.array([80,120,100,100,100,100])

# Reference curve
P_E = np.linspace(0.0001, 100, 1000)
T_M_E = Kechin_TM_P(P_E, 80.4, .201, 1/1.552, .634e-6)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot rhombic cells
ax1.plot(P, T_M_27, 'o', color='red', label='27 atoms')
ax1.plot(P_96, T_M_96, 'o', color='blue', label='96 atoms')
ax1.plot(P_150, T_M_150, 'o', color='green', label='150 atoms')

if popt_27 is not None:
    ax1.plot(P_fit, T_M_27_fit, '-', color='red', label='27 atoms fit')
if popt_96 is not None:
    ax1.plot(P_fit, T_M_96_fit, '-', color='blue', label='96 atoms fit')
if popt_150 is not None:
    ax1.plot(P_fit, T_M_150_fit, '-', color='green', label='150 atoms fit')

ax1.errorbar(P_exp, T_M_exp, yerr=T_M_exp_error, fmt='.', color='grey', label='Experimental Results')
ax1.plot(P_E, T_M_E, '-', color='black', label='Reference Curve')

ax1.set_title('Rhombic Cells')
ax1.set_xlabel('Pressure (GPa)')
ax1.set_ylabel('Melting Temperature (K)')
ax1.legend()
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 8500)

# Plot cubic cells
ax2.plot(P_32, T_M_32, 'o', color='red', label='32 atoms')
ax2.plot(P_108, T_M_108, 'o', color='blue', label='108 atoms')
ax2.plot(P_256, T_M_256, 'o', color='green', label='256 atoms')
ax2.plot(P_216, T_M_216, 'o', color='purple', label='216 atoms')

if popt_32 is not None:
    ax2.plot(P_fit, T_M_32_fit, '-', color='red', label='32 atoms fit')
if popt_108 is not None:
    ax2.plot(P_fit, T_M_108_fit, '-', color='blue', label='108 atoms fit')
if popt_256 is not None:
    ax2.plot(P_fit, T_M_256_fit, '-', color='green', label='256 atoms fit')
if popt_216 is not None:
    ax2.plot(P_fit, T_M_216_fit, '-', color='purple', label='216 atoms fit')

ax2.errorbar(P_exp, T_M_exp, yerr=T_M_exp_error, fmt='.', color='grey', label='Experimental Results')
ax2.plot(P_E, T_M_E, '-', color='black', label='Reference Curve')

ax2.set_title('Cubic Cells')
ax2.set_xlabel('Pressure (GPa)')
ax2.set_ylabel('Melting Temperature (K)')
ax2.legend()
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 8500)

plt.tight_layout()
plt.show()
