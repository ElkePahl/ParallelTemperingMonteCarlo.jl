import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Data
#Ar 27 Rhombic Cell


P = [0.0001, 0.000101325, 1, 5, 10, 15, 50, 100]        #Gpa
P_96 = [0.0001, 0.000101325, 1, 5, 10, 15, 50, 100] 
P_32 = [0.000101325, 1*1.01325, 5*1.01325, 10*1.01325, 15*1.01325, 30*1.01325, 50*1.01325, 100*1.01325]
P_108 = [0.0001, 0.000101325, 1, 5, 10, 15, 50, 100]
P_256 = [1, 5, 10, 15, 50, 100]
P_150 = [10, 15, 25, 50, 100] #50, 60,
P_216 = [5, 15, 50, 100]
P_HCP = [10, 15, 100]


P_exp = [64,65,66,70,74,77]

T_plus_27 = [123.40438499160605, 123.58013615667511, 391.3026613995936, 1031.1395878787039, 1633.574131835625, 2169.0002733751667, 5505.018041488894, 9792.651503983141]   #Raw Melting Temp (K)
T_plus_96 = [105.30338182419992, 105.0495642618424, 302.4078411337586, 775.8459979106822, 1219.310102458083, 1625.9582371200256, 4073.016427186415, 7457.843685180374]
T_plus_32 = [110,330,855,1340,1797.2935884104504, 2974.135686685555, 4483.105122715527, 8200.851335135036]
T_plus_108 = [106.53812872727019, 107.14131211275075, 302.45685871342874, 782.2604696609754, 1234.3227450502068, 1616, 4141, 7515] 
T_plus_256 = [247, 690, 942, 1504, 3411, 5556]
T_plus_150 = [1045, 1221, 1960, 3400,  5052] # 2507, 2503,
T_plus_216 = [525, 1359, 3344, 4610]
T_plus_HCP = [530, 1883, 4625]

T_M_27 = T_plus_27/(1 + np.log(2**(1/3)))  #T_+/T_M = 1 + Ln(2^(1/3))
T_M_96 = T_plus_96/(1 + np.log(2**(1/3)))  #T_+/T_M = 1 + Ln(2^(1/3))
T_M_32 = T_plus_32/(1 + np.log(2**(1/3)))
T_M_108 = T_plus_108/(1 + np.log(2**(1/3)))
T_M_256 = T_plus_256/(1 + np.log(2**(1/3)))
T_M_150 = T_plus_150/(1 + np.log(2**(1/3)))
T_M_216 = T_plus_216/(1 + np.log(2**(1/3)))
T_M_HCP = T_plus_HCP/(1 + np.log(2**(1/3)))
 
def Kechin_TM_P(x, T_0, a, b, c):
    return T_0*(1 + x/a)**b*np.exp(-c*x)

popt_27, pcov_27 = curve_fit(Kechin_TM_P, P, T_M_27, p0 = [80.4, .201, 1/1.552, .634*10**-6])
popt_96, pcov_96 = curve_fit(Kechin_TM_P, P_96, T_M_96, p0 = [80.4, .201, 1/1.552, .634*10**-6])
popt_32, pcov_32 = curve_fit(Kechin_TM_P, P_32, T_M_32, p0 = [80.4, .201, 1/1.552, .634*10**-6])
popt_108, pcov_108 = curve_fit(Kechin_TM_P, P_108, T_M_108, p0 = [80.4, .201, 1/1.552, .634*10**-6])
popt_256, pcov_256 = curve_fit(Kechin_TM_P, P_256, T_M_256, p0 = [80.4, .201, 1/1.552, .634*10**-6])
popt_150, pcov_150 = curve_fit(Kechin_TM_P, P_150, T_M_150, p0 = [80.4, .201, 1/1.552, .634*10**-6])
popt_216, pcov_216 = curve_fit(Kechin_TM_P, P_216, T_M_216, p0 = [80.4, .201, 1/1.552, .634*10**-6])

P_fit = np.linspace(.0001, 100, num = 100)
T_M_27_fit = Kechin_TM_P(P_fit, popt_27[0], popt_27[1], popt_27[2], popt_27[3])
T_M_96_fit = Kechin_TM_P(P_fit, popt_96[0], popt_96[1], popt_96[2], popt_96[3])
T_M_32_fit = Kechin_TM_P(P_fit, popt_32[0], popt_32[1], popt_32[2], popt_32[3])
T_M_108_fit = Kechin_TM_P(P_fit, popt_108[0], popt_108[1], popt_108[2], popt_108[3])
T_M_256_fit = Kechin_TM_P(P_fit, popt_256[0], popt_256[1], popt_256[2], popt_256[3])
T_M_150_fit = Kechin_TM_P(P_fit, popt_150[0], popt_150[1], popt_150[2], popt_150[3])
T_M_216_fit = Kechin_TM_P(P_fit, popt_216[0], popt_216[1], popt_216[2], popt_216[3])

P_E = np.linspace(.0001, 100, num = 100)
T_M_E = Kechin_TM_P(P_E, 80.4, .201, 1/1.552, .634*10**-6)
T_M_exp = [2800, 3000, 2900, 3050, 3000, 3150]
T_M_exp_error = [80,120,100,100,100,100]


plt.figure(figsize=(6, 8))
plt.plot(P, T_M_27, '.', color = 'red')
plt.plot(P_32, T_M_32, '.', color = 'blue' )

plt.plot(P_96, T_M_96, '.', color = 'red')
plt.plot(P_108, T_M_108, '.', color = 'blue')

# plt.plot(P_150, T_M_150, 'o', color = 'red', label = '150 Result')
# plt.plot(P_256, T_M_256, 'o', color = 'blue', label = '256 Result')

plt.plot(P_216, T_M_216, 'o', color = 'purple', label = '216 FCC Result')
# plt.plot(P_HCP, T_M_HCP, 'x', color = 'purple', label = '216 HCP Result')

plt.plot(P_fit, T_M_27_fit, ':', color = 'red', label = '27 Kechin Fit')
plt.plot(P_fit, T_M_32_fit, ':', color = 'blue', label = '32 Kechin Fit')

plt.plot(P_fit, T_M_96_fit, '--', color = 'red', label = '96 Kechin Fit')
plt.plot(P_fit, T_M_108_fit, '--', color = 'blue', label = '108 Kechin Fit')

# plt.plot(P_fit, T_M_150_fit, '-', color = 'red', label = '150 Kechin Fit')
# plt.plot(P_fit, T_M_256_fit, '-', color = 'blue', label = '256 Kechin Fit')

plt.plot(P_fit, T_M_216_fit, '-', color = 'purple', label = '216 Kechin Fit')

plt.errorbar(P_exp, T_M_exp, yerr=T_M_exp_error, fmt='.',  color = 'grey', label = 'Experimental Results Ref.')
plt.plot(P_E, T_M_E, '-', color = 'black', label = 'Ref. Work 256 Cubic')
plt.plot()
plt.xlim(0,100)
plt.ylim(0, 8500)
plt.xlabel("Pressure (GPa)")
plt.ylabel("Melting Temperature")
plt.legend()
plt.show()