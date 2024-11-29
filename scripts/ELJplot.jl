using Plots

# Define the standard Lennard-Jones potential function
function lennard_jones_potential(r, epsilon, sigma)
    4 * epsilon * ((sigma / r)^12 - (sigma / r)^6)
end

# Define the extended Lennard-Jones potential function
function extended_lennard_jones_potential(r, coeff)
    V = 0.0
    exponents = [6, 8, 10, 12, 14, 16]
    for n in 1:length(coeff)
        V += coeff[n] / r^exponents[n]
    end
    return V
end

# Parameters
epsilon = 4.0  # Depth of the potential well for standard LJ
sigma = 1.0    # Distance at which the potential is zero

# Coefficients for argon (provided)
coeff = [
    -123.635101619510,
    21262.8963716972,
    -3239750.64086661,
    189367623.844691,
    -4304257347.72069,
    35315085074.3605
]

# Generate r values for the main plot
r_values_main = 1:0.01:30.0

# Compute potentials for each r
lj_values_main = [lennard_jones_potential(r, epsilon, sigma) for r in r_values_main]
elj_values_main = [extended_lennard_jones_potential(r, coeff) for r in r_values_main]

# Create the main plot
plot(
    r_values_main,
    elj_values_main,
    xlims=(0.5, 1),
    ylims=(-5, 10),
    label="Standard Lennard-Jones",
    xlabel="Distance (r)",
    ylabel="Potential Energy (V(r))",
    linewidth=2,
    color=:blue
)

# Add the extended Lennard-Jones potential to the main plot
plot!(
    r_values_main,
    elj_values_main,
    label="Extended Lennard-Jones (Argon)",
    linewidth=2,
    color=:green
)

# # Add a horizontal line at zero for reference
# hline!([0], linestyle=:dash, color=:red, label="")

# # Create r values for the inset plot
# r_values_inset = 0.9:0.001:1.5

# # Compute potentials for the inset plot
# lj_values_inset = [lennard_jones_potential(r, epsilon, sigma) for r in r_values_inset]
# elj_values_inset = [extended_lennard_jones_potential(r, coeff) for r in r_values_inset]

# # Create the inset plot
# p_inset = plot(
#     r_values_inset,
#     lj_values_inset,
#     xlims=(0.9, 1.5),
#     ylims=(-5, 5),
#     label="Standard LJ",
#     linewidth=2,
#     color=:blue,
#     legend=false,
#     framestyle=:box,
#     xtickfontsize=6,
#     ytickfontsize=6
# )

# # Add the extended Lennard-Jones potential to the inset plot
# plot!(
#     p_inset,
#     r_values_inset,
#     elj_values_inset,
#     label="Extended LJ",
#     linewidth=2,
#     color=:green
# )

# # Add the inset to the main plot
# plot!(
#     inset = (p_inset, bbox(0.5, 0.3, 0.4, 0.4))
# )

# Add a legend to the main plot
# plot!(
#     legend = :topright,
#     title = "Lennard-Jones Potential Comparison"
# )
