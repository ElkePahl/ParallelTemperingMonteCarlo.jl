using Glob
using Optim
using ParallelTemperingMonteCarlo  

# Specify the directories for reading and writing files
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"
output_dir_initial = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/xyz"
output_dir_minimized = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"

# Define the EMCubicBC struct before its usage
struct EMCubicBC
    box_length::Float64
end

# Function to read multiple structures from a file and extract atom positions, box length, etc.
function read_structures(file_path)
    structures = []
    atoms = []
    box_length = 0.0
    cycle_number = 0
    for line in eachline(file_path)
        if startswith(line, "#")  # Detect metadata lines
            if !isempty(atoms)  # Save the previous structure before starting a new one
                push!(structures, (atoms, box_length, cycle_number))
                atoms = []  # Reset for the new structure
            end
            # Extract cycle number and box length from metadata line
            if occursin(r"Cycle: (\d+), Box Length: (\d+\.\d+)", line)
                match_data = match(r"Cycle: (\d+), Box Length: (\d+\.\d+)", line)
                cycle_number = parse(Int, match_data.captures[1])
                box_length = parse(Float64, match_data.captures[2])
            end
        elseif !isempty(line)  # Detect atom lines
            atom_data = split(line)
            atom, x, y, z = atom_data[1], parse(Float64, atom_data[2]), parse(Float64, atom_data[3]), parse(Float64, atom_data[4])
            push!(atoms, (atom, x, y, z))
        end
    end
    # Don't forget to save the last structure
    if !isempty(atoms)
        push!(structures, (atoms, box_length, cycle_number))
    end
    return structures
end

# Wrap positions function
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

# Function to calculate the squared distance between atoms with periodic boundary conditions
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
# Function to calculate the total energy using ELJ potential
function lj_elj(x, N, pot, bc::EMCubicBC)
    E = 0.0
    for i in 1:N-1
        for j in i+1:N
            a = [x[3*i-2], x[3*i-1], x[3*i]]
            b = [x[3*j-2], x[3*j-1], x[3*j]]
            d2 = distance2(a, b, bc)
            E += dimer_energy(pot, d2)
        end
    end
    return E
end

# Optimization setup to use ELJ potential
function optimize_structure(atoms, coeff, box_length)
    N = length(atoms)
    pot = ELJPotentialEven(coeff)  # Using the ELJ potential
    x = flatten_atoms(atoms)
    bc = EMCubicBC(box_length)
    lj_wrapper(x) = lj_elj(x, N, pot, bc)
    od = OnceDifferentiable(lj_wrapper, x; autodiff=:forward)
    result = Optim.minimizer(optimize(od, x, ConjugateGradient(), Optim.Options(g_tol=1e-8)))
    minimized_atoms = rebuild_atoms(result, atoms, bc)  # Pass bc to ensure wrapping
    return minimized_atoms
end

# Main loop
coeff = [-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]  # Coefficients for ELJ potential
sigma = 2.7700 * 1.8897259886  # Not be necessary if using ELJ potential?

for file_path in glob("*.dat", input_dir)
    structures = read_structures(file_path)  # Get all structures in the file
    for (atoms, box_length, cycle_number) in structures
        # Create a unique filename using the cycle number
        file_base_name = basename(file_path) * "_cycle_$cycle_number"
        
        # Write the initial structure to an .xyz file in the initial output directory
        initial_xyz_path = joinpath(output_dir_initial, file_base_name * "_initial.xyz")
        write_xyz(atoms, initial_xyz_path, "Initial structure for cycle $cycle_number")

        # Perform energy minimization on the structure
        minimized_atoms = optimize_structure(atoms, coeff, box_length)

        # Write the minimized structure to a new .xyz file in the minimized output directory
        minimized_xyz_path = joinpath(output_dir_minimized, file_base_name * "_minimized.xyz")
        write_xyz(minimized_atoms, minimized_xyz_path, "Minimized structure for cycle $cycle_number")
        
        println("Processed cycle: ", cycle_number, " with Box Length: ", box_length)
    end
end
