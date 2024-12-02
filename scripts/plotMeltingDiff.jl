using Plots
using LsqFit

# Pressure (GPa) range
P_fit = range(0.0001, stop=100, length=100)

# Kechin model function
function Kechin_TM_P(x, p)
    T_0, a, b, c = p
    return T_0 .* (1 .+ x ./ a) .^ b .* exp.(-c .* x)
end

# Given pressure and temperature data for 216 FCC, 216 HCP
P_216 = [0.000101325, 1, 25, 35, 50, 100]
P_HCP = [0.000101325, 5, 10, 15, 60, 75, 100]
T_M_216 = [110, 312, 2407, 3157, 4150, 7701] ./ (1 + log(2^(1 / 3)))
T_M_HCP = [108, 780, 1250, 1637, 4800, 5820, 7780] ./ (1 + log(2^(1 / 3)))

# Reference Kechin curve parameters
reference_params = [80.4, 0.201, 1 / 1.552, 0.634e-6]
T_M_E = Kechin_TM_P(P_fit, reference_params)  # Reference curve

# Experimental points
P_exp = [64, 65, 66, 70, 74, 77]
T_M_exp = [2800, 3000, 2900, 3050, 3000, 3150]
T_M_exp_error = [80, 120, 100, 100, 100, 100]

# Fit the Kechin model to 216 FCC and 216 HCP data
p0 = [80.4, 0.201, 1 / 1.552, 0.634e-6]  # Initial guess
fit_216 = curve_fit(Kechin_TM_P, P_216, T_M_216, p0)
fit_HCP = curve_fit(Kechin_TM_P, P_HCP, T_M_HCP, p0)

# Generate fitted curves for 216 FCC and HCP
T_M_216_fit = Kechin_TM_P(P_fit, fit_216.param)
T_M_HCP_fit = Kechin_TM_P(P_fit, fit_HCP.param)

# Calculate the difference between FCC and HCP
diff_216_HCP = T_M_216_fit - T_M_HCP_fit

# Correct the reference curve by subtracting the FCC-HCP difference
T_M_corrected_ref = T_M_E .- diff_216_HCP

# Create the main plot
plt = plot(size=(600, 600), legend=:bottomright, framestyle=:box)

# Plot the main curves
plot!(plt, P_fit, T_M_216_fit, label="216 FCC Fit", color="red", linewidth=2)
plot!(plt, P_fit, T_M_HCP_fit, label="216 HCP Fit", color="purple", linewidth=2)
plot!(plt, P_fit, T_M_E, label="Ref. PTMC", color="black", linestyle=:solid, linewidth=2)
plot!(plt, P_fit, T_M_corrected_ref, label="Ref. PTMC with HCP Correction", color="black", linestyle=:dash, linewidth=2)

# Add experimental points with error bars
scatter!(plt, P_exp, T_M_exp, yerror=T_M_exp_error, label="Ref. Experimental Results", color="black", marker=:diamond, ms=6)

# Customize the main plot
xlabel!(plt, "Pressure (GPa)")
ylabel!(plt, "Melting Temperature (K)")
xlims!(plt, 0, 100)
ylims!(plt, 0, 7000)
#title!(plt, "Melting Curves and Corrected Reference")

# Add an inset plot
plot!(
    plt,
    P_fit, T_M_216_fit,
    label="",
    color="red",
    linewidth=2,
    inset=bbox(0.2, 0.05, 0.3, 0.3),
    subplot=2,
    framestyle=:box,
    xtickfontsize=6,
    ytickfontsize=6,
    xguidefontsize=8,
    yguidefontsize=8,
)
plot!(plt, P_fit, T_M_HCP_fit, label="", color="purple", linewidth=2, subplot=2)
plot!(plt, P_fit, T_M_E, label="", color="black", linewidth=2, subplot=2)
plot!(plt, P_fit, T_M_corrected_ref, label="", color="black", linestyle=:dash, linewidth=2, subplot=2)
scatter!(plt, P_exp, T_M_exp, yerror=T_M_exp_error, label="", color="black", marker=:diamond, ms=:6, subplot=2)

# Customize the inset plot
xlims!(plt, 60, 80, subplot=2)
ylims!(plt, 2500, 4500, subplot=2)

# Display the plot
display(plt)
