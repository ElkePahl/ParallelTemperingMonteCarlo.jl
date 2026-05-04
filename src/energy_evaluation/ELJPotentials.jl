"""
    ELJPotentialEven(coefficients) <: AbstractDimerPotential

Implements type for extended Lennard Jones potential with only even powers; subtype of [`AbstractDimerPotential`](@ref)<:[`AbstractPotential`](@ref);
as sum over `c_i r^(-i)`, starting with `i=6` up to `i=N+6` with only even integers `i`
field name: `coeff::SVector{N,T}` : contains ELJ coefficients `c_i` from `i=6` to `i=N+6` in steps of 2, coefficient for every even power needed.
"""
struct ELJPotentialEven{N,T} <: AbstractDimerPotential
    coeff::SVector{N,T}
end
function ELJPotentialEven{N}(c) where {N}
    @boundscheck length(c) == N ||
        error("number of ELJ coefficients does not match given length")
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialEven{N,T}(coeff)
end
function ELJPotentialEven(c)
    N = length(c)
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialEven{N,T}(coeff)
end
function long_range_correction(pot::ELJPotentialEven, num_atoms, r_cut)
    if r_cut <= 50 # TODO: why
        e_lrc = 0.0
    else
        r_cut_sqrt = r_cut^0.5
        rc3 = r_cut * r_cut_sqrt
        e_lrc = 0.0
        for i in eachindex(pot.coeff)
            e_lrc += pot.coeff[i] / rc3 / (2i + 1)
            rc3 *= r_cut
        end
        e_lrc *= pi * num_atoms^2 / 4 / r_cut_sqrt^3
    end
    return e_lrc
end

"""
    ELJPotentialB(coeff_a, coeff_b, coeff_c) <: AbstractDimerPotentialB

Extended Lennard-Jones Potential in a magnetic field where there is anisotropy in the coefficient vectors `coeff_a::SVector{N,T}`, `coeff_b::SVector{N,T}`, `coeff_c::SVector{N,T}`.
"""
struct ELJPotentialB{N,T} <: AbstractDimerPotentialB
    coeff_a::SVector{N,T}
    coeff_b::SVector{N,T}
    coeff_c::SVector{N,T}
end
function ELJPotentialB{N}(a, b, c) where {N}
    @boundscheck length(c) == N ||
        error("number of ELJ coefficients does not match given length")
    coeff_a = SVector{N}(a)
    coeff_b = SVector{N}(b)
    coeff_c = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialB{N,T}(coeff_a, coeff_b, coeff_c)
end
function ELJPotentialB(a, b, c)
    N = length(c)
    coeff_a = SVector{N}(a)
    coeff_b = SVector{N}(b)
    coeff_c = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialB{N,T}(coeff_a, coeff_b, coeff_c)
end
function set_variables(
    config::Config{T}, dist2_matrix::Matrix{Float64}, pot::AbstractDimerPotentialB
) where {T}
    N = length(config)
    tan_matrix = get_tantheta_mat(config)

    return ELJPotentialBVariables{T}(zeros(N), tan_matrix, tan_matrix, zeros(N))
end
function long_range_correction(pot::ELJPotentialB, num_atoms, r_cut)
    coeff = (-0.1279111890228638, -1.328138539967966, 12.260941135261255, 41.12212408251662)
    if r_cut <= 16 # TODO: why 16? doesn't this depend on anything?
        e_lrc = 0.0
    else
        r_cut_sqrt = r_cut^0.5
        rc3 = r_cut * r_cut_sqrt
        e_lrc = 0.0
        for i in 1:4
            e_lrc += coeff[i] / rc3 / (2i + 1)
            rc3 *= r_cut
        end
        e_lrc *= pi * num_atoms^2 / 4 / r_cut_sqrt^3
    end
    return e_lrc
end

"""
    ELJPotentialBVariables{T}
Contains the `en_atom_vec::Array{T}`, `tan_mat::Matrix{T}` and `new_tan_vec::Vector{T}` for the ELJPotentialB potential.
"""
mutable struct ELJPotentialBVariables{T} <: AbstractPotentialVariables
    en_atom_vec::Array{T}
    tan_mat::Matrix{T}
    new_tan_mat::Matrix{T}
    new_tan_vec::Vector{T}
end

"""
    ELJPotential{N,T}
Implements type for extended Lennard Jones potential; subtype of [`AbstractDimerPotential`](@ref)<:[`AbstractPotential`](@ref);
as sum over `c_i r^(-i)`, starting with `i=6` up to `i=N+6`
field name: `coeff::SVector{N,T}` : contains ELJ coefficients `c_i` from `i=6` to `i=N+6`, coefficient for every power needed.
Constructors:
    ELJPotential{N}(c) where N
    ELJPotential(c)
"""
struct ELJPotential{N,T} <: AbstractDimerPotential
    coeff::SVector{N,T}
end

function ELJPotential{N}(c) where {N}
    @boundscheck length(c) == N ||
        error("number of ELJ coefficients does not match given length")
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotential{N,T}(coeff)
end

function ELJPotential(c)
    N = length(c)
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotential{N,T}(coeff)
end

"""
    dimer_energy(pot::ELJPotential{N}, r2::Real) where N
    dimer_energy(pot::ELJPotentialEven{N}, r2::Real) where N
    dimer_energy(pot::ELJPotentialB{N}, r2::Real, z_angle::Real) where N
Calculates energy of dimer for given potential `pot` and squared distance `r2` between atoms
Methods implemented for:

-   [`ELJPotential`](@ref)

-   [`ELJPotentialEven`](@ref)
Dimer energy when the distance square between two atom is `r2` and the angle between the line connecting them and z-direction is `z_angle`.
When `r2 < 5.30`, returns 1.
"""
function dimer_energy(pot::ELJPotential{N}, r2::Real) where {N}
    r = sqrt(r2)
    r6inv = 1 / (r2 * r2 * r2)
    sum1 = 0.0
    for i in 1:N
        sum1 += pot.coeff[i] * r6inv
        r6inv /= r
    end
    return sum1
end

function dimer_energy(pot::ELJPotentialEven{N}, r2::Real) where {N}
    r6inv = 1 / (r2 * r2 * r2)
    sum1 = 0.0
    for i in 1:N
        sum1 += pot.coeff[i] * r6inv
        r6inv /= r2
    end
    return sum1
end
function dimer_energy(pot::ELJPotentialB{N}, r2::Real, z_angle::Real) where {N}
    if r2 >= 5.30
        r6inv = 1 / (r2 * r2 * r2)
        t2 = 2 / (z_angle^2 + 1) - 1     #cos(2*theta)
        t4 = 2 * t2^2 - 1
        sum1 = pot.coeff_c[1] * r6inv * (1 + pot.coeff_a[1] * t2 + pot.coeff_b[1] * t4)
        r6inv /= r2
        for i in 2:N
            sum1 += pot.coeff_c[i] * r6inv * (1 + pot.coeff_a[i] * t2 + pot.coeff_b[i] * t4)
            r6inv /= r2^0.5
        end
    else
        sum1 = 0.1
    end
    return sum1
end
