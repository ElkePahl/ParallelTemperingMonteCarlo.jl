using Glob
using Optim
using ParallelTemperingMonteCarlo

# Specify the directories for reading and writing files
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ne"
output_dir_initial = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ne/xyz"
output_dir_minimized = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ne/Optim"

struct EMCubicBC
    box_length::Float64
end

function wrap_position!(pos, bc::EMCubicBC)
    # Adjust position components to remain within the periodic boundaries
    for i in 1:3
        pos[i] -= bc.box_length * floor(pos[i] / bc.box_length)
    end
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

function read_structure(file_path)
    atoms = []
    box_length = 0.0
    open(file_path, "r") do f
        readline(f)  # Skip the first empty line
        info_line = readline(f)  # Read the second line which contains the box length
        box_length = parse(Float64, match(r"Box Length: (\d+\.\d+)", info_line).captures[1])
        for line in eachline(f)
            atom, x, y, z = split(line)
            push!(atoms, (atom, parse(Float64, x), parse(Float64, y), parse(Float64, z)))
        end
    end
    return atoms, box_length
end

function flatten_atoms(atoms)
    [coord for atom in atoms for coord in atom[2:end]]
end

function rebuild_atoms(flat_coords, atoms)
    [(atoms[i][1], flat_coords[3*i-2], flat_coords[3*i-1], flat_coords[3*i]) for i in 1:length(atoms)]
end

function write_xyz(atoms, file_path, comment="")
    open(file_path, "w") do f
        println(f, length(atoms))
        println(f, comment)
        for (atom, x, y, z) in atoms
            println(f, "$atom $x $y $z")
        end
    end
end

function lj_elj(x, N, pot, bc::EMCubicBC)
    E = 0.0
    for i in 1:N-1
        for j = i+1:N
            a = [x[3*i-2], x[3*i-1], x[3*i]]
            b = [x[3*j-2], x[3*j-1], x[3*j]]
            d2 = distance2(a, b, bc)
            E += dimer_energy(pot, d2)
        end
    end
    return E
end

function optimize_structure(atoms, coeff, box_length)
    N = length(atoms)
    pot = ELJPotentialEven(coeff)
    x = flatten_atoms(atoms)
    bc = EMCubicBC(box_length)
    function lj_wrapper(x)
        E = lj_elj(x, N, pot, bc)
        # Ensure atom positions are wrapped during the optimization process
        for i in 1:N
            wrap_position!(view(x, (3*i-2):(3*i)), bc)
        end
        return E
    end
    od = OnceDifferentiable(lj_wrapper, x; autodiff=:forward)
    result = Optim.minimizer(optimize(od, x, ConjugateGradient(), Optim.Options(g_tol=1e-8)))
    # Apply wrapping to the final result to ensure all atoms are within the box
    final_coords = result[:]
    for i in 1:N
        wrap_position!(view(final_coords, (3*i-2):(3*i)), bc)
    end
    minimized_atoms = rebuild_atoms(final_coords, atoms)
    return minimized_atoms
end

# Main loop to process each file
coeff = [-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
for file_path in glob("*.dat", input_dir)
    atoms, box_length = read_structure(file_path)
    
    initial_xyz_path = joinpath(output_dir_initial, basename(file_path) * ".xyz")
    write_xyz(atoms, initial_xyz_path, "Initial structure")
    
    minimized_atoms = optimize_structure(atoms, coeff, box_length)
    
    minimized_xyz_path = joinpath(output_dir_minimized, "minimized_" * basename(file_path) * ".xyz")
    write_xyz(minimized_atoms, minimized_xyz_path, "Minimized structure")
    
    println("Processed file: ", basename(file_path), " with Box Length: ", box_length)
end
