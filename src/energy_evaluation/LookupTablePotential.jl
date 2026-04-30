"""
    struct LookupTablePotential <: AbstractDimerPotentialB

# Fields:
- `table::Matrix{Float64}`: Table of values.
- `start_dist::Float64`: First distance in list.
- `start_angle::Float64`: First angle in list.
- `l_dist::Int32`: Number of distances in the table.
- `l_angle::Int32`: Number of angles in the table.
- `d_dist::Float64`:
- `d_angle::Float64`:
- `c6coeff::Float64`:

# Constructor

    LookupTablePotential(file)

Read lookup table potential from `file`.
"""
struct LookupTablePotential <: AbstractDimerPotentialB
    table::Matrix{Float64}
    start_dist::Float64
    start_angle::Float64
    l_dist::Int32     #how many distances
    l_angle::Int32    #how many angles
    d_dist::Float64
    d_angle::Float64
    c6coeff::Float64
end

"""
    read_lookuptable(file)

Read a lookup table from `file`. Return all information required to construct
[`LookupTablePotential`](@ref).
"""
function read_lookuptable(link::String)
    open(link) do f
        line = readline(f)
        line = first(split(line, '!')) # remove comments
        l_dist, start_dist, d_dist = parse(Int32, split(line, ", ")[1]),
        parse(Float64, split(line, ", ")[2]),
        parse(Float64, split(line, ", ")[3])

        line = readline(f)
        line = first(split(line, '!')) # remove comments
        l_angle, start_angle, d_angle = parse(Int32, split(line, ", ")[1]),
        parse(Float64, split(line, ", ")[2]),
        parse(Float64, split(line, ", ")[3])
        table = Matrix{Float64}(undef, l_angle, l_dist)
        for i in 1:l_angle
            for j in 1:l_dist
                line = readline(f)
                line = first(split(line, '!')) # remove comments
                table[i, j] = parse(Float64, line)
            end
        end

        avg_lr = 0
        for i in 1:l_angle
            avg_lr += table[i, 1000]
        end
        avg_lr /= l_angle
        c6coeff = avg_lr * (start_dist + 1000 * d_dist)^6

        return table, start_dist, start_angle, l_dist, l_angle, d_dist, d_angle, c6coeff
    end
end

LookupTablePotential(link::String) = LookupTablePotential(read_lookuptable(link)...)

#TODO: document me
mutable struct LookupTableVariables{T} <: AbstractPotentialVariables
    en_atom_vec::Array{T}
    tan_mat::Matrix{T}
    new_tan_mat::Matrix{T}
    new_tan_vec::Vector{T}
end

function dimer_energy_update!(
    index, dist2_mat, tanmat, new_dist2_vec, new_tan_vec, en_tot, pot::LookupTablePotential
)
    @views delta_en =
        dimer_energy_atom(index, new_dist2_vec, new_tan_vec, pot) -
        dimer_energy_atom(index, dist2_mat[index, :], tanmat[index, :], pot)

    return delta_en + en_tot
end
function dimer_energy_update!(
    index,
    dist2_mat,
    tanmat,
    new_dist2_vec,
    new_tan_vec,
    en_tot,
    r_cut,
    pot::LookupTablePotential,
)
    @views delta_en =
        dimer_energy_atom(index, new_dist2_vec, new_tan_vec, r_cut, pot) -
        dimer_energy_atom(index, dist2_mat[index, :], tanmat[index, :], r_cut, pot)

    return delta_en + en_tot
end

function dimer_energy(pot::LookupTablePotential, r2, tan)
    angle_index = 1
    if abs(tan) > 0.00872687153 && abs(tan) <= 114.592845357
        angle_index = round(Int32, atan(abs(tan)) / pot.d_angle * 180.0 / pi + 1.0)
    elseif abs(tan) > 114.592845357
        angle_index = 91
    end

    if r2 <= (pot.start_dist + 0.5 * pot.d_dist)^2
        e = pot.table[angle_index, 1]
    elseif r2 <= (pot.start_dist + pot.l_dist * pot.d_dist)^2
        dist_index = round(Int32, (r2^0.5 - pot.start_dist) / pot.d_dist)
        if dist_index == 0
            println(r2)
        end
        e = pot.table[angle_index, dist_index]
    else
        e = pot.c6coeff / r2^3
    end

    return e
end
function set_variables(
    config::Config{T}, dist2_matrix::Matrix{Float64}, pot::LookupTablePotential
) where {T}
    N = length(config)
    tan_matrix = get_tantheta_mat(config)

    return LookupTableVariables{T}(zeros(N), tan_matrix, tan_matrix, zeros(N))
end
function long_range_correction(pot::LookupTablePotential, num_atoms, r_cut)
    if r_cut <= 10
        e_lrc = 1.0
    else
        rc3 = r_cut ^ 1.5
        e_lrc = pot.c6coeff / rc3 / 3
        e_lrc *= π * num_atoms^2 / 4 / rc3
    end
    return e_lrc
end
