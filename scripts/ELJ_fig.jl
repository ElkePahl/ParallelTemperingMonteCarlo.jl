# Define the ELJ dimer potential function
function dimer_energy(coeff, r2)
    r = sqrt(r2) # Convert r^2 to r
    V = 0.0
    for (n, c) in enumerate(coeff)
        exponent = 2 * n + 4 # Exponent for each term
        V += c * r^(-exponent)
    end
    return V
end

# Define squared distance (no periodic boundaries for simplicity in this example)
function distance2(a, b)
    dx = a[1] - b[1]
    dy = a[2] - b[2]
    dz = a[3] - b[3]
    return dx^2 + dy^2 + dz^2
end

# Define the total ELJ potential function for a pair of atoms
function lj_elj(x, coeff)
    # For simplicity, consider only one pair of atoms
    a = [x[1], x[2], x[3]]    # Position of atom 1
    b = [x[4], x[5], x[6]]    # Position of atom 2
    d2 = distance2(a, b)
    return dimer_energy(coeff, d2)
end

# Set up coefficients and parameters
coeff = [-123.635101619510, 21262.8963716972, -3239750.64086661, 
         189367623.844691, -4304257347.72069, 35315085074.3605]

# Define simulation parameters
x = [0.0, 0.0, 0.0, 3.7782, 0.0, 0.0] # Positions of 2 Argon atoms

# Plot the ELJ potential for varying interatomic distances
using Plots

# Generate a range of distances
r_values = 3.0:0.01:10.0 # Interatomic distances in Å
V_values = [dimer_energy(coeff, r^2) for r in r_values]

# Plotting the ELJ potential
plot(r_values, V_values,
    xlabel="Distance (r) [Å]",
    ylabel="Potential Energy (V)",
    title="ELJ Potential for Argon Dimer",
    legend=false)
