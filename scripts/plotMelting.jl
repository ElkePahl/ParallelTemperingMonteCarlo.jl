using Plots
using LsqFit

# Data arrays (same as before)
# Pressures (GPa)
P = [0.0001, 0.000101325, 1, 5, 10, 15, 50, 100]
P_27 = P  # Define P_27
P_96 = [0.0001, 0.000101325, 1, 5, 10, 15, 50, 100]
P_32 = [0.000101325, 1*1.01325, 5*1.01325, 10*1.01325,
        15*1.01325, 30*1.01325, 50*1.01325, 100*1.01325]
P_108 = [0.000101325, 1, 10, 15, 25, 50, 100]
P_150 = [0.000101325, 1, 5, 10, 15, 50, 100]
P_216 = [0.000101325, 1, 25, 35, 50, 100]
P_HCP = [0.000101325, 5, 10, 15, 50, 100]
P_256 = [1, 5, 10, 15, 100]

# Temperatures (K)
T_plus_27 = [123.40438499160605, 123.58013615667511,
             391.3026613995936, 1031.1395878787039,
             1633.574131835625, 2169.0002733751667,
             5505.018041488894, 9792.651503983141]
T_plus_96 = [105.30338182419992, 105.0495642618424,
             302.4078411337586, 775.8459979106822,
             1219.310102458083, 1625.9582371200256,
             4073.016427186415, 7457.843685180374]
T_plus_32 = [110, 330, 855, 1340, 1797.2935884104504,
             2974.135686685555, 4483.105122715527,
             8200.851335135036]
T_plus_108 = [108, 303, 1230, 1624, 2412, 4092, 7571]
T_plus_150 = [107, 307, 804, 1232, 1655, 4172, 7677]
T_plus_216 = [110, 312, 2407, 3157, 4150, 7701]
T_plus_HCP = [108, 780, 1250, 1637, 3880, 7180]
T_plus_256 = [314, 803, 1255, 1672, 7782]


# Calculate T_M from T_plus
T_M_factor = 1 + log(2^(1/3))
T_M_27 = T_plus_27 ./ T_M_factor
T_M_96 = T_plus_96 ./ T_M_factor
T_M_32 = T_plus_32 ./ T_M_factor
T_M_108 = T_plus_108 ./ T_M_factor
T_M_150 = T_plus_150 ./ T_M_factor
T_M_216 = T_plus_216 ./ T_M_factor
T_M_HCP = T_plus_HCP ./ T_M_factor
T_M_256 = T_plus_256 ./ T_M_factor

# Define the model function
function Kechin_TM_P(x, p)
    T_0, a, b, c = p
    return T_0 .* (1 .+ x ./ a) .^ b .* exp.(-c .* x)
end

# Initial parameter guess
p0 = [80.4, 0.201, 1 / 1.552, 0.634e-6]

# Perform curve fitting
fit_27 = curve_fit(Kechin_TM_P, P_27, T_M_27, p0)
fit_96 = curve_fit(Kechin_TM_P, P_96, T_M_96, p0)
fit_32 = curve_fit(Kechin_TM_P, P_32, T_M_32, p0)
fit_108 = curve_fit(Kechin_TM_P, P_108, T_M_108, p0)
fit_150 = curve_fit(Kechin_TM_P, P_150, T_M_150, p0)
fit_216 = curve_fit(Kechin_TM_P, P_216, T_M_216, p0)
fit_HCP = curve_fit(Kechin_TM_P, P_HCP, T_M_HCP, p0)
fit_256 = curve_fit(Kechin_TM_P, P_256, T_M_256, p0)

# Generate P_fit and compute fitted temperatures
P_fit = range(0.0001, stop=100, length=100)
T_M_27_fit = Kechin_TM_P(P_fit, fit_27.param)
T_M_96_fit = Kechin_TM_P(P_fit, fit_96.param)
T_M_32_fit = Kechin_TM_P(P_fit, fit_32.param)
T_M_108_fit = Kechin_TM_P(P_fit, fit_108.param)
T_M_150_fit = Kechin_TM_P(P_fit, fit_150.param)
T_M_216_fit = Kechin_TM_P(P_fit, fit_216.param)
T_M_HCP_fit = Kechin_TM_P(P_fit, fit_HCP.param)
T_M_256_fit = Kechin_TM_P(P_fit, fit_256.param)

# Experimental data
P_exp = [64, 65, 66, 70, 74, 77]
T_M_exp = [2800, 3000, 2900, 3050, 3000, 3150]
T_M_exp_error = [80, 120, 100, 100, 100, 100]

# Reference data
P_E = P_fit
T_M_E = Kechin_TM_P(P_E, [80.4, 0.201, 1 / 1.552, 0.634e-6])

# Create the main plot
plt = plot(size=(600, 600), legend=:bottomright, framestyle=:box)

# Plot data points on the main plot
#scatter!(plt, P_27, T_M_27, label="27 Result", color="orange", marker=:circle)
#scatter!(plt, P_32, T_M_32, label="32 Result", color="cyan", marker=:circle)
#scatter!(plt, P_96, T_M_96, label="96 Result", color="green", marker=:circle)
#scatter!(plt, P_108, T_M_108, label="108 Result", color="blue", marker=:circle)
#scatter!(plt, P_150, T_M_150, label="150 Result", color="magenta", marker=:circle)
scatter!(plt, P_216, T_M_216, label="216 FCC Result", color="red", marker=:circle)
scatter!(plt, P_HCP, T_M_HCP, label="216 HCP Result", color="purple", marker=:circle)

# Plot fitted curves on the main plot
# plot!(plt, P_fit, T_M_27_fit, label="27 Kechin Fit", color="red", linestyle=:dot)
# plot!(plt, P_fit, T_M_32_fit, label="32 Kechin Fit", color="blue", linestyle=:dot)
# plot!(plt, P_fit, T_M_96_fit, label="96 Kechin Fit", color="red", linestyle=:dashdot)
# plot!(plt, P_fit, T_M_108_fit, label="108 Kechin Fit", color="blue", linestyle=:dashdot)
# plot!(plt, P_fit, T_M_150_fit, label="150 Kechin Fit", color="red", linestyle=:dash)
plot!(plt, P_fit, T_M_216_fit, label="216 FCC Kechin Fit", color="red")
plot!(plt, P_fit, T_M_HCP_fit, label="216 HCP Kechin Fit", color="purple")
plot!(plt, P_fit, T_M_256_fit, label="256 Kechin Fit", color="blue")

plot!(plt, P_E, T_M_E, label="Ref. PTMC", color="black")
# Plot experimental data with error bars on the main plot
scatter!(plt, P_exp, T_M_exp, yerror=T_M_exp_error, label="Ref. Experimental Results", color="grey", marker=:diamond)

# Customize the main plot
xlims!(plt, 0, 100)
ylims!(plt, 0, 7000)
xlabel!(plt, "Pressure (GPa)")
ylabel!(plt, "Melting Temperature (K)")
#title!(plt, "Melting Temperature vs. Pressure")

# Now, add the inset plot
# Plot the inset data using subplot=2 and inset=bbox(...)
scatter!(
    plt,
    P_216, T_M_216,
    label="",
    color="red",
    marker=:circle,
    framestyle=:box,
    inset=bbox(0.2, 0.05, 0.3, 0.3),
    subplot=2,
    xtickfontsize=6,
    ytickfontsize=6,
    xguidefontsize=8,
    yguidefontsize=8
)

# Add other datasets and fits to the inset plot
#scatter!(plt, P_32, T_M_32, label="", color="cyan", marker=:circle, subplot=2)
#scatter!(plt, P_96, T_M_96, label="", color="green", marker=:circle, subplot=2)
#scatter!(plt, P_108, T_M_108, label="", color="blue", marker=:circle, subplot=2)
#scatter!(plt, P_150, T_M_150, label="", color="magenta", marker=:circle, subplot=2)
#scatter!(plt, P_216, T_M_216, label="", color="red", marker=:circle, subplot=2)
scatter!(plt, P_HCP, T_M_HCP, label="", color="purple", marker=:circle, subplot=2)

# Plot fitted curves on the inset plot
#plot!(plt, P_fit, T_M_27_fit, label="", color="red", linestyle=:dot, subplot=2)
#plot!(plt, P_fit, T_M_32_fit, label="", color="blue", linestyle=:dot, subplot=2)
#plot!(plt, P_fit, T_M_96_fit, label="", color="red", linestyle=:dashdot, subplot=2)
#plot!(plt, P_fit, T_M_108_fit, label="", color="blue", linestyle=:dashdot, subplot=2)
plot!(plt, P_E, T_M_E, label="", color="black", subplot=2)
#plot!(plt, P_fit, T_M_150_fit, label="", color="red", linestyle=:dash, subplot=2)
plot!(plt, P_fit, T_M_216_fit, label="", color="red", subplot=2)
plot!(plt, P_fit, T_M_HCP_fit, label="", color="purple", subplot=2)
plot!(plt, P_fit, T_M_256_fit, label="", color="blue", subplot=2)

# Plot experimental data with error bars in the inset plot
scatter!(plt, P_exp, T_M_exp, yerror=T_M_exp_error, label="", color="grey", marker=:diamond, subplot=2)

# Customize the inset plot
# xlims!(plt, 80, 100, subplot=2)
# ylims!(plt, 4000, 6500, subplot=2)
xlims!(plt, 80, 100, subplot=2)
ylims!(plt, 4000, 6500, subplot=2)
# framestyle!(plt, :box, subplot=2)
# legend!(plt, false, subplot=2)

# Display the plot
display(plt)
