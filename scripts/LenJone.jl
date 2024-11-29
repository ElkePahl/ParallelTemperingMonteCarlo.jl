using Plots

# Define the Lennard-Jones potential function
function lennard_jones_potential(r, epsilon, sigma)
    4 * epsilon * ((sigma / r)^12 - (sigma / r)^6)
end

# Parameters
epsilon = 4.0
sigma = 1.0

# Main plot data
r_values_main = 0.8:0.01:3.0
lj_values_main = lennard_jones_potential.(r_values_main, epsilon, sigma)

# Inset plot data
r_values_inset = 0.8:0.01:3.0
lj_values_inset = lennard_jones_potential.(r_values_inset, epsilon, sigma)

# Create the main plot
plot(
    r_values_main,
    lj_values_main,
    xlims=(0.5, 2.5),
    ylims=(-5, 40),
    label="",
    xlabel="Distance (r)",
    ylabel="Potential Energy (V(r))",
    linewidth=2,
    framestyle=:box,
    color=:blue
)
hline!([0], linestyle=:dash, color=:red, legend = false)

# Create the inset plot
plot!(
    r_values_main,
    lj_values_main,
    xlims=(0.9, 1.5),
    ylims=(-5, 3),
    label="",
    # xlabel="r",
    # ylabel="V(r)",
    inset=bbox(0.5, 0.1, 0.4, 0.4),
    subplot=2,
    linewidth=2,
    color=:blue,
    legend=false,
    framestyle=:box,
    xtickfontsize=6,
    ytickfontsize=6,
    xguidefontsize=8,
    yguidefontsize=8
)
#plot!(p_inset, inset = (p_inset, bbox(0.5, 0.3, 0.4, 0.4)))

# Combine the plots using layout
# plot(
#     p_main,
#     inset = (p_inset, bbox(0.5, 0.3, 0.4, 0.4)),
#     size=(800, 600)
# )
