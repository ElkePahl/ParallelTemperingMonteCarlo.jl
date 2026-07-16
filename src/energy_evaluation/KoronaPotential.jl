"""
    KoronaPotential(coefficients) <: AbstractDimerPotential

Implements functional form proposed by Korona et al.; subtype of [`AbstractDimerPotential`](@ref)<:[`AbstractPotential`](@ref);
It is a combination of a exponential part and an extended Lennard-Jones part.
The exponential part contains coefficients A, a1, a2, alpha and beta.
The extended Lennard Jones part is a sum over f_{2i}(br)c_i r^(-i), starting with `i=6` up to `i=16` with only even integers `i`
"""
struct KoronaPotential{N,T} <: AbstractDimerPotential
    coeff_ab::SVector{N,T}
    coeff_c::SVector{N,T}
    coeff_elj::SVector{N,T}
end



function KoronaPotential(a, b, c)
    N = length(c)
    coeff_ab = SVector{N}(a)
    coeff_c = SVector{N}(b)
    coeff_elj = SVector{N}(c)
    T = eltype(c)
    return KoronaPotential{N,T}(coeff_ab,coeff_c,coeff_elj)
end

"""
    long_range_correction(pot::KoronaPotential, num_atoms, r_cut)
    The Korona potential is very close to the extended Lennard-Jones potential from the equilibrium distance to long range.
    For convenience, the long range correction uses the ELJ coefficients.
"""
function long_range_correction(pot::KoronaPotential, num_atoms, r_cut)
    if r_cut <= 50 # TODO: why
        e_lrc = 0.0
    else
        r_cut_sqrt = r_cut^0.5
        rc3 = r_cut * r_cut_sqrt
        e_lrc = 0.0
        for i in eachindex(pot.coeff_elj)
            e_lrc += pot.coeff_elj[i] / rc3 / (2i + 1)
            rc3 *= r_cut
        end
        e_lrc *= pi * num_atoms^2 / 4 / r_cut_sqrt^3
    end
    return e_lrc
end


function dimer_energy(pot::KoronaPotential{N}, r2::Real) where {N}
    r = sqrt(r2)
    r6inv = 1 / (r2 * r2 * r2)

    y = pot.coeff_ab[6] * r
    y2 = y^2
    y7 = y2 * y2 * y2 * y
    f6 = 1. - exp(-y) * (1 + y * (1 + y * (1/2 + y * (1/6 + y * (1/24 + y * (1/120 + y/720))))))
    f8 = f6 - exp(-y) * y7 * (1/5040 + y/40320)
    f10 = f8 - exp(-y) * y7 * y2 * (1/362880 + y/3628800)
    f12 = f10 - exp(-y) * y7 * y2 * y2 * (1/39916800 + y/479001600)
    f14 = f12 - exp(-y) * y7 * y7 / y * (1/6227020800 + y/87178291200)
    f16 = f14 - exp(-y) * y7 * y7 * y * (1/1307674368000 + y/20922789888000)
    
    C_b = (pot.coeff_c[1] * f6, 
    pot.coeff_c[2] * f8, 
    pot.coeff_c[3] * f10, 
    pot.coeff_c[4] * f12, 
    pot.coeff_c[5] * f14, 
    pot.coeff_c[6] * f16)

    sum = (pot.coeff_ab[1] + pot.coeff_ab[2]*r + pot.coeff_ab[3]/r) * exp( -pot.coeff_ab[4]*r + pot.coeff_ab[5]*r2)

    for i =1:N
        sum -= C_b[i] * r6inv
        r6inv /= r2
    end

    return sum
end
