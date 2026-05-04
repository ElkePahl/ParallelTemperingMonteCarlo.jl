"""
    AbstractMachineLearningPotential <: AbstractPotential
Abstract type for all Machine Learning Potentials.
"""
abstract type AbstractMachineLearningPotential <: AbstractPotential end

"""
    RuNNerPotential <: AbstractMachineLearningPotential
Contains the important structs required for a neural network potential defined in the MachineLearningPotential package:
-   Fields are:
    -   `nnp::NeuralNetworkPotential` -- a struct containing the weights, biases and neural network parameters.
    -   `radsymfunctions::StructVector{RadialType2{Float64}}` -- a vector containing the hyperparameters used to calculate symmetry function values
    -   `angsymfunctions::StructVector{AngularType3{Float64}}` -- a vector containing the hyperparameters used to calculate symmetry function values
    -   `r_cut::Float64` -- every symmetry function has an r_cut, but saving it here saves annoying memory unpacking
"""
struct RuNNerPotential{Nrad,Nang} <: AbstractMachineLearningPotential
    nnp::NeuralNetworkPotential
    radsymfunctions::StructVector{RadialType2{Float64}} #SVector{Nrad,RadialType2}
    angsymfunctions::StructVector{AngularType3{Float64}}
    r_cut::Float64
    boundary::Float64
end
"""
    RuNNerPotential(nnp::NeuralNetworkPotential,radsymvec,angsymvec)
RuNNerPotential constructor/initializer function, given a neural network potential `nnp` and the symmetry functions `radsymvec,angsymvec` and the cutoff radius `r_cut`.
"""
function RuNNerPotential(
    nnp::NeuralNetworkPotential,
    radsymvec::Vector{RadialType2{N}},
    angsymvec::Vector{AngularType3{N}},
) where {N<:Real}
    r_cut = radsymvec[1].r_cut
    nrad = length(radsymvec)
    nang = length(angsymvec)
    radvec = StructVector([rsymm for rsymm in radsymvec])
    angvec = StructVector([asymm for asymm in angsymvec])
    return RuNNerPotential{nrad,nang}(nnp, radvec, angvec, r_cut, 0.0)
end

function RuNNerPotential(nnp, radsymvec, angsymvec, boundary)
    r_cut = radsymvec[1].r_cut
    nrad = length(radsymvec)
    nang = length(angsymvec)
    radvec = StructVector([rsymm for rsymm in radsymvec])
    angvec = StructVector([asymm for asymm in angsymvec])
    return RuNNerPotential{nrad,nang}(nnp, radvec, angvec, r_cut, boundary * boundary)
end

function set_variables(
    config::Config{T}, dist2_mat::Matrix{Float64}, pot::RuNNerPotential{nrad,nang}
) where {T,nrad,nang}
    N = length(config)
    f_matrix = cutoff_function.(sqrt.(dist2_mat), Ref(pot.r_cut))
    g_matrix = total_symm_calc(
        config, dist2_mat, f_matrix, pot.radsymfunctions, pot.angsymfunctions, nrad, nang
    )

    return NNPVariables{T}(zeros(N), zeros(N), g_matrix, f_matrix, copy(g_matrix), zeros(N))
end

function initialise_energy(
    config::Config,
    dist2_mat::Matrix{Float64},
    potential_variables::AbstractPotentialVariables,
    ensemble_variables::AbstractEnsembleVariables,
    pot::RuNNerPotential,
)
    potential_variables.en_atom_vec = forward_pass(
        potential_variables.g_matrix, length(config), pot.nnp
    )
    en_tot = sum(potential_variables.en_atom_vec)
    return en_tot, potential_variables
end

function energy_update!(
    ensemblevariables::AbstractEnsembleVariables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    pot::RuNNerPotential,
)
    if any(
        new_dist2_vec[i] < pot.boundary for
        i in eachindex(new_dist2_vec) if i != ensemblevariables.index
    )
        new_energy = 100.0
    else
        potential_variables = get_new_state_vars!(
            ensemblevariables.trial_move,
            ensemblevariables.index,
            config,
            potential_variables,
            dist2_mat,
            new_dist2_vec,
            pot,
        )
        potential_variables, new_energy = calc_new_runner_energy!(potential_variables, pot)
    end

    return potential_variables, new_energy
end

"""
    NNPVariables{T}
Bundle of variables used for the NNP potential:
-   `en_atom_vec::Vector{T}` -- the per-atom energy vector
-   `new_en_atom::Vector{T}` -- the new per-atom energy vector
-   `g_matrix::Matrix{T}` -- the G matrix
-   `f_matrix::Matrix{T}` -- the F matrix
-   `new_g_matrix::Matrix{T}` -- the new G matrix
-   `new_f_vec::Vector{T}` -- the new F vector
Todo: someone who knows what these are should write a better description
"""
mutable struct NNPVariables{T} <: AbstractPotentialVariables
    en_atom_vec::Vector{T}
    new_en_atom::Vector{T}
    g_matrix::Matrix{T}
    f_matrix::Matrix{T}
    new_g_matrix::Matrix{T}
    new_f_vec::Vector{T}
end
"""
    get_new_state_vars!(trial_pos::PositionVector,atomindex::Int,config::Config,potential_variables::NNPVariables,dist2_mat::Matrix{Float64},new_dist2_vec::Vector{Float64},pot::RuNNerPotential{Nrad,Nang}) where {Nrad,Nang}
Function for finding the new state variables for calculating an NNP. Redefines `new_f` and `new_g` matrices based on the `trial_pos` of atom at `atomindex` and adjusts the parameters in the `potential_variables` according to the variables in `pot`.
"""
function get_new_state_vars!(
    trial_pos::PositionVector,
    atomindex::Int,
    config::Config,
    potential_variables::NNPVariables,
    dist2_mat::Matrix{Float64},
    new_dist2_vec::Vector{Float64},
    pot::RuNNerPotential{Nrad,Nang},
) where {Nrad,Nang}
    potential_variables.new_f_vec = cutoff_function.(sqrt.(new_dist2_vec), Ref(pot.r_cut))
    potential_variables.new_g_matrix = copy(potential_variables.g_matrix)
    potential_variables.new_g_matrix = total_thr_symm!(
        potential_variables.new_g_matrix,
        config,
        trial_pos,
        dist2_mat,
        new_dist2_vec,
        potential_variables.f_matrix,
        potential_variables.new_f_vec,
        atomindex,
        pot.radsymfunctions,
        pot.angsymfunctions,
        Nrad,
        Nang,
    )
    return potential_variables
end
"""
    calc_new_runner_energy!(potential_variables::NNPVariables,pot::RuNNerPotential)
Function designed to calculate the new per-atom energy according to the RuNNer forward pass with parameters defined in `pot`. utilises the `new_g_matrix` to redefine the `new_en` and `new_en_atom` variables within the `potential_variables` struct.
"""
function calc_new_runner_energy!(potential_variables::NNPVariables, pot::RuNNerPotential)
    potential_variables.new_en_atom = forward_pass(
        potential_variables.new_g_matrix, length(potential_variables.en_atom_vec), pot.nnp
    )
    new_en = sum(potential_variables.new_en_atom)
    return potential_variables, new_en
end
#----------------------------------------------------------#
#--------------------NNP with two atoms--------------------#
#----------------------------------------------------------#
"""
    RuNNerPotential2Atom <: AbstractMachineLearningPotential
Contains the important structs required for a neural network potential defined in the MachineLearningPotential package for a 2 atom system:
    Fields are:
    nnp# -- structs containing the weights, biases and neural network parameters.
    symmetryfunctions -- a vector containing the hyperparameters used to calculate symmetry function values
    r_cut -- every symmetry function has an r_cut, but saving it here saves annoying memory unpacking
"""
struct RuNNerPotential2Atom{Nrad,Nang,N1,N2} <: AbstractMachineLearningPotential
    nnp1::NeuralNetworkPotential
    nnp2::NeuralNetworkPotential
    radsymfunctions::StructVector{RadialType2a{Float64}}
    angsymfunctions::StructVector{AngularType3a{Float64}}
    r_cut::Float64
    boundary::Float64
end

function RuNNerPotential2Atom(nnp1, nnp2, radsymvec, angsymvec, n1, n2)#,g_offsets_vec)
    r_cut = radsymvec[1].r_cut
    nrad = length(radsymvec)
    nang = length(angsymvec)
    radvec = StructVector([rsymm for rsymm in radsymvec])
    angvec = StructVector([asymm for asymm in angsymvec])

    return RuNNerPotential2Atom{nrad,nang,n1,n2}(nnp1, nnp2, radvec, angvec, r_cut, 0.0)#,SVector{nrad*2+nang*3}(g_offsets),SVector{nrad*2+nang*3}(tpz) )
end
function RuNNerPotential2Atom(nnp1, nnp2, radsymvec, angsymvec, n1, n2, boundary)#,g_offsets_vec)
    r_cut = radsymvec[1].r_cut
    nrad = length(radsymvec)
    nang = length(angsymvec)
    radvec = StructVector([rsymm for rsymm in radsymvec])
    angvec = StructVector([asymm for asymm in angsymvec])

    return RuNNerPotential2Atom{nrad,nang,n1,n2}(
        nnp1, nnp2, radvec, angvec, r_cut, boundary * boundary
    )#,SVector{nrad*2+nang*3}(g_offsets),SVector{nrad*2+nang*3}(tpz) )
end

function set_variables(
    config::Config{T},
    dist2_mat::Matrix{Float64},
    pot::RuNNerPotential2Atom{nrad,nang,n1,n2},
) where {T,nrad,nang,n1,n2}
    N = length(config)
    if n1 + n2 != N
        # ??? TODO: how severe is the problem? throw an error with propper message?
        println("problem")
    end
    Ng = nrad * 2 + nang * 3

    f_matrix = MMatrix{N,N}(cutoff_function.(sqrt.(dist2_mat), Ref(pot.r_cut)))
    g_temp_matrix = total_symm_calc(
        config,
        dist2_mat,
        f_matrix,
        pot.radsymfunctions,
        pot.angsymfunctions,
        nrad,
        nang,
        n1,
        n2,
    )
    g_matrix = MMatrix{Ng,N}(g_temp_matrix)
    return NNPVariables2a{T,N,Ng}(
        zeros(N),
        zeros(N),
        g_matrix,
        f_matrix,
        MMatrix{Ng,N}(zeros(Ng, N)),
        MVector{N}(zeros(N)),
    )
end

function initialise_energy(
    config,
    dist2_mat,
    potential_variables,
    ensemble_variables,
    pot::RuNNerPotential2Atom{Nrad,Nang,N1,N2},
) where {Nrad,Nang,N1,N2}
    potential_variables.en_atom_vec[1:N1] = forward_pass(
        potential_variables.g_matrix[:, 1:N1], N1, pot.nnp1
    )

    if N2 != 0
        potential_variables.en_atom_vec[(N1 + 1):(N1 + N2)] = forward_pass(
            potential_variables.g_matrix[:, (N1 + 1):(N1 + N2)], N2, pot.nnp2
        )
    end

    en_tot = sum(potential_variables.en_atom_vec)

    return en_tot, potential_variables
end

function energy_update!(
    ensemblevariables::AbstractEnsembleVariables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    pot::RuNNerPotential2Atom,
)
    potential_variables = get_new_state_vars!(
        ensemblevariables.trial_move,
        ensemblevariables.index,
        config,
        potential_variables,
        dist2_mat,
        new_dist2_vec,
        pot,
    )
    potential_variables, new_energy = calc_new_runner_energy!(potential_variables, pot)

    return potential_variables, new_energy
end

"""
    NNPVariables2a{T,Na,Ng} <: AbstractPotentialVariables
    T  variable type, usually Float64
    Na number of atoms
    Ng number of symmetry functions
Mutable parameters relevant to a 2 atom NNP using the RuNNer Package.
fields include:
    -en_atom_vec: atomic energy corresponding to config
    -new_en_atom: after an atom move, the new atomic energy
    -g_matrix: matrix of symmetry values, length NgxNa
    -f_matrix: matrix of cutoff function values aor atom pairs i,j
    -new_g_matrix: after atom move, new symmetry values
    -new_f_vec: after atom move, new cutoff values
"""
mutable struct NNPVariables2a{T,Na,Ng} <: AbstractPotentialVariables
    en_atom_vec::Vector
    new_en_atom::Vector
    g_matrix::MMatrix{Ng,Na,T}
    f_matrix::MMatrix{Na,Na,T}
    new_g_matrix::MMatrix{Ng,Na,T}
    new_f_vec::MVector{Na,T}
end
"""
    get_new_state_vars!(trial_pos, atomindex, config::Config, potential_variables::NNPVariables2a, dist2_mat, new_dist2_vec, pot::RuNNerPotential2Atom{Nrad, Nang, N1, N2}) where {Nrad, Nang, N1, N2}
    get_new_state_vars!(indices, config, potential_variables, dist2_mat, potential::RuNNerPotential2Atom{Nrad, Nang, N1, N2}) where {Nrad, Nang, N1, N2}
Function to calculate the altered state variables after an atom move:
Takes the new trial_position, its index, the total config, the current state variables, the distance matrix and updated vector and potential values.
Calculates the new cutoff function values, the updated symmetry function matrix and passes these back to potential_variables.

Method 2 calculates the new state variables based on an atom_swap. Accepts many of the same variables, but the main difference is the `indices` vector, indicating which two atoms we are swapping.
Also returns, most imporantly `potential_variables.new_g_matrix`.

"""
function get_new_state_vars!(
    trial_pos,
    atomindex,
    config::Config,
    potential_variables::NNPVariables2a,
    dist2_mat,
    new_dist2_vec,
    pot::RuNNerPotential2Atom{Nrad,Nang,N1,N2},
) where {Nrad,Nang,N1,N2}
    potential_variables.new_f_vec = MVector{N1 + N2}(
        cutoff_function.(sqrt.(new_dist2_vec), Ref(pot.r_cut))
    )

    potential_variables.new_g_matrix = fill!(potential_variables.new_g_matrix, 0.0)

    potential_variables.new_g_matrix = calc_delta_matrix(
        potential_variables.new_g_matrix,
        config,
        trial_pos,
        atomindex,
        dist2_mat,
        new_dist2_vec,
        potential_variables.f_matrix,
        potential_variables.new_f_vec,
        pot.radsymfunctions,
        pot.angsymfunctions,
        Nrad,
        Nang,
        N1,
        N2,
    )

    potential_variables.new_g_matrix =
        potential_variables.g_matrix .+ potential_variables.new_g_matrix

    return potential_variables
end
function get_new_state_vars!(
    indices,
    config,
    potential_variables,
    dist2_mat,
    potential::RuNNerPotential2Atom{Nrad,Nang,N1,N2},
) where {Nrad,Nang,N1,N2}
    potential_variables.new_g_matrix = fill!(potential_variables.new_g_matrix, 0.0)

    potential_variables.new_g_matrix = calc_swap_matrix(
        potential_variables.new_g_matrix,
        config,
        indices[1],
        indices[2],
        dist2_mat,
        potential_variables.f_matrix,
        potential.radsymfunctions,
        potential.angsymfunctions,
        Nrad,
        Nang,
        N1,
        N2,
    )
    potential_variables.new_g_matrix =
        potential_variables.g_matrix .+ potential_variables.new_g_matrix

    return potential_variables
end
"""
    calc_new_runner_energy!(potential_variables::NNPVariables2a{T,Na,Ng},pot::RuNNerPotential2Atom{Nrad,Nang,N1,N2}) where {T,Na,Ng} where {Nrad,Nang,N1,N2}
Function to calculate the energy of a new configuration after an atom move. Accepts the potential_variables struct and runs a forward pass on the new_g_matrix. Returns the new energy.
"""
function calc_new_runner_energy!(
    potential_variables::NNPVariables2a{T,Na,Ng}, pot::RuNNerPotential2Atom{Nrad,Nang,N1,N2}
) where {T,Na,Ng} where {Nrad,Nang,N1,N2}
    potential_variables.new_en_atom[1:N1] = forward_pass(
        potential_variables.new_g_matrix[:, 1:N1], N1, pot.nnp1
    )
    if N2 != 0
        potential_variables.new_en_atom[(N1 + 1):(N1 + N2)] = forward_pass(
            potential_variables.new_g_matrix[:, (N1 + 1):(N1 + N2)], N2, pot.nnp2
        )
    end

    new_en = sum(potential_variables.new_en_atom)

    return potential_variables, new_en
end
