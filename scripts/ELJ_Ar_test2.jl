using Glob
using Optim
using ParallelTemperingMonteCarlo

# Specify the directories for reading and writing files
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"
output_dir_minimized = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/Minimized"

# Structure for handling periodic boundary conditions
struct EMCubicBC
    box_length::Float64
end

# Function to read structure from a .dat file and extract atom positions and box length
function read_structure(file_path::String)
    atoms = []
    local box_length = 0.0  # Declare local variable to avoid scope issues
    open(file_path, "r") do f
        readline(f)  # Skip the first empty line
        info_line = readline(f)  # Read the second line, which contains the box length and possibly other info
        
        # Try to match the box length using a regular expression
        m = match(r"Box Length: (\d+\.\d+)", info_line)
        if m !== nothing
            box_length = parse(Float64, m.captures[1])
        else
            error("Box length not found in file: $file_path")
        end
        
        # Read the remaining lines, each representing an atom, and store the information
        for line in eachline(f)
            atom_info = split(line)
            if length(atom_info) == 4
                atom, x, y, z = atom_info
                push!(atoms, (atom, parse(Float64, x), parse(Float64, y), parse(Float64, z)))
            else
                error("Incorrect atom format in file: $file_path")
            end
        end
    end
    return atoms, box_length
end

# Function to wrap positions within the simulation box
function wrap_positions!(pos, bc::EMCubicBC)
    for i in 1:3
        pos[i] -= bc.box_length * floor(pos[i] / bc.box_length)
    end
end

# Flatten and rebuild atom coordinates for optimization
flatten_atoms(atoms) = [coord for atom in atoms for coord in atom[2:end]]

function rebuild_atoms(flat_coords, atoms, bc::EMCubicBC)
    new_atoms = []
    for i in 1:length(atoms)
        pos = [flat_coords[3*i-2], flat_coords[3*i-1], flat_coords[3*i]]
        wrap_positions!(pos, bc)
        push!(new_atoms, (atoms[i][1], pos...))
    end
    return new_atoms
end

# Calculate the squared distance between two atoms under periodic boundary conditions
function distance2(a, b, bc::EMCubicBC)
    dx = a[1] - b[1]
    dy = a[2] - b[2]
    dz = a[3] - b[3]
    dx -= bc.box_length * round(dx / bc.box_length)
    dy -= bc.box_length * round(dy / bc.box_length)
    dz -= bc.box_length * round(dz / bc.box_length)
    return dx^2 + dy^2 + dz^2
end

# Write minimized structures to .xyz files
function write_xyz(atoms, file_path, comment="")
    open(file_path, "w") do f
        println(f, length(atoms))  # Write the number of atoms
        println(f, comment)  # Write a comment line
        for (atom, x, y, z) in atoms
            println(f, "$atom $x $y $z")
        end
    end
end

# Write minimized structures to .dat files (including box length information)
function write_dat(atoms, box_length, file_path)
    open(file_path, "w") do f
        println(f, "# Minimized structure")  # Add a header line
        println(f, "Box Length: $box_length")  # Write the box length
        for (atom, x, y, z) in atoms
            println(f, "$atom $x $y $z")
        end
    end
end

# Define the ELJ potential
function lj_elj(x, N, pot, bc::EMCubicBC)
    E = 0.0
    for i = 1:N-1
        for j = i+1:N
            a = [x[3*i-2], x[3*i-1], x[3*i]]
            b = [x[3*j-2], x[3*j-1], x[3*j]]
            d2 = distance2(a, b, bc)
            E += dimer_energy(pot, d2)
        end
    end
    return E
end

# Perform optimization using the ELJ potential and show energy at each iteration
function optimize_structure(atoms, coeff, box_length)
    N = length(atoms)
    pot = ELJPotentialEven(coeff)  # Using the ELJ potential
    x = flatten_atoms(atoms)
    bc = EMCubicBC(box_length)
    lj_wrapper(x) = lj_elj(x, N, pot, bc)
    
    # Set options to show trace and set the gradient tolerance
    options = Optim.Options(g_tol=1e-8, show_trace=true)
    opt_result = optimize(lj_wrapper, x, ConjugateGradient(), options)
    
    # Print final energy and return minimized atoms
    println("Final minimized energy: ", opt_result.minimum)
    minimized_atoms = rebuild_atoms(opt_result.minimizer, atoms, bc)
    return minimized_atoms
end

# Main loop for processing .dat files and writing minimized structures
coeff = [-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]

for file_path in glob("*.dat", input_dir)
    atoms, box_length = read_structure(file_path)
    minimized_atoms = optimize_structure(atoms, coeff, box_length)

    # Modify the filename to remove the ".dat" extension and write both ".xyz" and ".dat"
    base_name = replace(basename(file_path), ".dat" => "")
    minimized_xyz_path = joinpath(output_dir_minimized, "minimized_$base_name.xyz")
    minimized_dat_path = joinpath(output_dir_minimized, "minimized_$base_name.dat")
    
    # Write the minimized structure to both .xyz and .dat files
    write_xyz(minimized_atoms, minimized_xyz_path, "Minimized structure")
    write_dat(minimized_atoms, box_length, minimized_dat_path)
    
    println("Processed file: ", basename(file_path), " with Box Length: ", box_length)
end
