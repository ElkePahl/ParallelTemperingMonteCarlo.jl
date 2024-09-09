using Glob
using Optim
using ParallelTemperingMonteCarlo  

# Specify the directories for reading and writing files
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"
output_dir_initial = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/xyz"
output_dir_minimized = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized"

# Function to read structure from a file and extract atom positions and box length
function read_structure(file_path)
    atoms = []
    box_length = 0.0
    open(file_path, "r") do f
        readline(f)  # Skip the first empty line
        info_line = readline(f)  # Read the second line which contains the box length
        # Extract the box length from the info line using a regular expression.
        box_length = parse(Float64, match(r"Box Length: (\d+\.\d+)", info_line).captures[1])
        # Read the remaining lines, each representing an atom, and store the information
        for line in eachline(f)
            atom, x, y, z = split(line)
            push!(atoms, (atom, parse(Float64, x), parse(Float64, y), parse(Float64, z)))
        end
    end
    return atoms, box_length
end

function wrap_positions!(pos, bc::EMCubicBC)
    # Wrap each coordinate to be within the box limits [0, bc.box_length]
    for i in 1:3
        pos[i] -= bc.box_length * floor(pos[i] / bc.box_length)
    end
end

# Function to flatten atom coordinates for optimization and rebuild atoms list from flat array
flatten_atoms(atoms) = [coord for atom in atoms for coord in atom[2:end]]

function rebuild_atoms(flat_coords, atoms, bc::EMCubicBC)
    new_atoms = []
    for i in 1:length(atoms)
        # Extract and wrap coordinates
        pos = [flat_coords[3*i-2], flat_coords[3*i-1], flat_coords[3*i]]
        wrap_positions!(pos, bc)
        push!(new_atoms, (atoms[i][1], pos...))
    end
    return new_atoms
end

struct EMCubicBC
    box_length::Float64
end

function distance2(a, b, bc::EMCubicBC)
    dx = a[1] - b[1]
    dy = a[2] - b[2]
    dz = a[3] - b[3]

    dx -= bc.box_length * round(dx / bc.box_length)
    dy -= bc.box_length * round(dy / bc.box_length)
    dz -= bc.box_length * round(dz / bc.box_length)

    return dx^2 + dy^2 + dz^2
end

# Function to write structures to .xyz files
function write_xyz(atoms, file_path, comment="")
    open(file_path, "w") do f
        println(f, length(atoms))  # Write the number of atoms
        println(f, comment)  # Write a comment line
        # Write the atomic positions.
        for (atom, x, y, z) in atoms
            println(f, "$atom $x $y $z")
        end
    end
end

# Define the ELJ potential
function elj_potential_even(coeffs, r2)
    r6 = r2^3
    r12 = r6^2
    return coeffs[1] + coeffs[2] * r2 + coeffs[3] * r6 + coeffs[4] * r12 + coeffs[5] * r6^2 + coeffs[6] * r12^2
end

# Function to calculate the total energy using the new ELJ potential
function lj_elj(x, N, coeff, bc::EMCubicBC)
    E = 0.0
    for i in 1:N-1
        for j in i+1:N
            a = [x[3*i-2], x[3*i-1], x[3*i]]
            b = [x[3*j-2], x[3*j-1], x[3*j]]
            d2 = distance2(a, b, bc)
            E += elj_potential_even(coeff, d2)  # Use the new potential
        end
    end
    return E
end

# Optimization setup to use ELJ potential with new coefficients
function optimize_structure(atoms, coeff, box_length)
    N = length(atoms)
    x = flatten_atoms(atoms)
    bc = EMCubicBC(box_length)
    lj_wrapper(x) = lj_elj(x, N, coeff, bc)
    
    # Log the initial energy
    initial_energy = lj_wrapper(x)
    println("Initial energy: $initial_energy")

    # Set optimizer options
    options = Optim.Options(g_tol=1e-6, iterations=1000)

    # Optimization
    result = optimize(lj_wrapper, x, LBFGS(), options)

    # Log the final energy
    final_energy = lj_wrapper(Optim.minimizer(result))
    println("Final energy: $final_energy")

    minimized_atoms = rebuild_atoms(Optim.minimizer(result), atoms, bc)  # Pass bc to ensure wrapping
    return minimized_atoms
end

# Main loop with new potential coefficients
coeff = [-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]  # New coefficients for ELJ potential
sigma = 2.7700 * 1.8897259886  # Not be necessary if using ELJ potential?

for file_path in glob("*.dat", input_dir)
    local atoms, box_length
    atoms, box_length = read_structure(file_path)
    
    # Write the initial structure to an .xyz file 
    initial_xyz_path = joinpath(output_dir_initial, splitext(basename(file_path))[1] * ".xyz")
    write_xyz(atoms, initial_xyz_path, "Initial structure")

    # Perform energy minimization on the structure.
    minimized_atoms = optimize_structure(atoms, coeff, box_length)

    # Write the minimized structure to a new .xyz file
    minimized_xyz_path = joinpath(output_dir_minimized, "minimized_" * splitext(basename(file_path))[1] * ".xyz")
    write_xyz(minimized_atoms, minimized_xyz_path, "Minimized structure")
    
    println("Processed file: ", basename(file_path), " with Box Length: ", box_length)
end

