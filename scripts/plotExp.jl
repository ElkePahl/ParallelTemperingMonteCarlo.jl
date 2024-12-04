using Plots
using CurveFit

# Experimental data points
pressure_exp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160]
temperature_exp = [1200, 1500, 1800, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500, 3700, 3900]

# Define the Birch-Murnaghan EOS function
function BM_EOS(P, T0, B0, B0_prime)
    T0 * (1 + (3/2) * (B0_prime - 4) * ((P / B0) ^ (2/3)) - (3/2) * ((P / B0) ^ (1/3)))
end

# Perform a fit to the experimental data
initial_guess = [1200, 100, 4.0] # Initial guesses for T0, B0, and B0_prime
fit_result = curve_fit(BM_EOS, pressure_exp, temperature_exp, initial_guess)

# Extract fitted parameters
T0_fit, B0_fit, B0_prime_fit = fit_result.param

# Generate the BM EOS curve using the fitted parameters
pressure_fit = range(10, 160, length=100)
temperature_fit = [BM_EOS(P, T0_fit, B0_fit, B0_prime_fit) for P in pressure_fit]

# Plot the experimental data points
scatter(
    pressure_exp,
    temperature_exp,
    label="Experimental Data",
    color=:red,
    markersize=6,
    marker=:circle,
)

# Plot the fitted BM EOS curve
plot!(
    pressure_fit,
    temperature_fit,
    label="BM EOS Fit",
    color=:blue,
    linewidth=2,
)

# Add labels and title
xlabel!("Pressure (GPa)")
ylabel!("Temperature (K)")
title!("Birch-Murnaghan EOS Fit to Experimental Data")
xlims!(0, 160)
ylims!(0, 4000)

# Display the plot
plot!()
