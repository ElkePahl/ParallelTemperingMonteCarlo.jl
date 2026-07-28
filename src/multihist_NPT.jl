module Multihistogram_NPT

using DelimitedFiles, LinearAlgebra, StaticArrays
using ProgressMeter

using ..InputParams
using ..Ensembles
using ..EnergyEvaluation
using ..CustomTypes
export multihistogram_NPT

function temp_trajectories(temp::TempGrid)
    tempnumber = length(temp.t_grid)
    tempnumber_result = tempnumber * 10
    return tempnumber, tempnumber_result
end

function histogram_initialise(ensemble::AbstractEnsemble, temp::TempGrid, results::Output)
    p = ensemble.pressure
    k = 3.166811429 / 10^6
    temp_o = temp.t_grid
    beta = temp.beta_grid
    Emin = results.en_min
    Emax = results.en_max
    Vmin = results.v_min
    Vmax = results.v_max
    Ebins = results.n_bin
    Vbins = results.n_bin
    dEhist = (Emax - Emin) / Ebins
    dVhist = (Vmax - Vmin) / Vbins
    EVhistogram = results.ev_histogram
    return p, k, temp_o, beta, Emin, Vmin, Ebins, Vbins, dEhist, dVhist, EVhistogram
end

function Temp_grid_result(ti::Number, tf::Number, tempnumber_result::Int)
    temp_grid_result = TempGrid{tempnumber_result}(ti, tf)
    temp_result = temp_grid_result.t_grid
    beta_result = temp_grid_result.beta_grid
    return temp_result, beta_result
end
const NPTHistogram = Vector{Matrix{Float64}}
function free_energy_initialise(
    EVhistogram::NPTHistogram,
    Ebins::Int,
    Vbins::Int,
    tempnumber::Int,
    tempnumber_result::Int,
)
    free_energy = Array{Float64}(undef, tempnumber)
    new_free_energy = Array{Float64}(undef, tempnumber)
    normalconst = Array{Float64}(undef, tempnumber_result)
    ncycles = Array{Float64}(undef, tempnumber)

    for i in 1:tempnumber
        free_energy[i] = 0
        new_free_energy[i] = 0
        ncycles[i] = 0
        for m in 1:Ebins
            for n in 1:Vbins
                ncycles[i] = ncycles[i] + EVhistogram[i][m + 1, n + 1]
            end
        end
    end

    for i in 1:tempnumber_result
        normalconst[i] = 0
    end
    return free_energy, new_free_energy, normalconst, ncycles
end

function quasiprob(
    betat::Number,
    m::Int,
    n::Int,
    ncycles::VorS,
    dEhist::Number,
    dVhist::Number,
    Emin::Number,
    Vmin::Number,
    tempnumber::Number,
    EVhistogram,
    beta::VorS,
    p::Number,
    free_energy::VorS,
)
    energy_t = Emin + (m - 0.5) * dEhist
    volume = Vmin + (n - 0.5) * dVhist
    quasiprob = 0
    denom = 0
    offset = -1e6
    for i in 1:tempnumber
        offset = max(offset, -beta[i] * (energy_t + p * volume) - free_energy[i])
    end
    offset = offset + log(10^3)
    for i in 1:tempnumber
        quasiprob = quasiprob + EVhistogram[i][m + 1, n + 1]
        denom =
            denom +
            ncycles[i] * exp(-beta[i] * (energy_t + p * volume) - free_energy[i] - offset)
    end

    quasiprob = quasiprob / denom * exp(-betat * (energy_t + p * volume) - offset)
    return quasiprob
end

"""
    multihistogram_NPT(ensemble::AbstractEnsemble, temp::TempGrid, results::Output, conv_threshold::Number, readfile::Bool; show_progress=true)
Multihistogram analysis for NPT:
-   `conv_threshold` is the convergence threshold, which user can choose.
-   Now "readfile" can only be false.
-   Example: `multihistogram_NPT(ensemble, temp, results, 10^(-3), false)`
"""
function multihistogram_NPT(
    ensemble::AbstractEnsemble,
    temp::TempGrid,
    results::Output,
    conv_threshold,
    readfile::Bool;
    show_progress=true,
    maxiter=1000,
)
    if readfile == false
        tempnumber, tempnumber_result = temp_trajectories(temp)
        p, k, temp_o, beta, Emin, Vmin, Ebins, Vbins, dEhist, dVhist, EVhistogram = histogram_initialise(
            ensemble, temp, results
        )
    end
    temp_result, beta_result = Temp_grid_result(
        temp_o[1], temp_o[tempnumber], tempnumber_result
    )

    free_energy, new_free_energy, normalconst, ncycles = free_energy_initialise(
        EVhistogram, Ebins, Vbins, tempnumber, tempnumber_result
    )

    progress = ProgressThresh(conv_threshold; enabled=show_progress)
    for it in 1:maxiter
        for i in 1:tempnumber
            betat = beta[i]
            new_free_energy[i] = 0
            for m in 1:Ebins
                for n in 1:Vbins
                    new_free_energy[i] =
                        new_free_energy[i] + quasiprob(
                            betat,
                            m,
                            n,
                            ncycles,
                            dEhist,
                            dVhist,
                            Emin,
                            Vmin,
                            tempnumber,
                            EVhistogram,
                            beta,
                            p,
                            free_energy,
                        )
                end
            end
            new_free_energy[i] = log(new_free_energy[i])
        end

        delta = 0.0
        for i in 1:tempnumber
            delta = delta + abs(new_free_energy[i] - free_energy[i])^2
            free_energy[i] = new_free_energy[i]
        end

        update!(progress, delta)
        if delta < conv_threshold
            break             #if converged, exit the loop
        end
    end

    for i in 1:tempnumber_result
        betat = beta_result[i]
        for m in 1:Ebins
            for n in 1:Vbins
                normalconst[i] =
                    normalconst[i] + quasiprob(
                        betat,
                        m,
                        n,
                        ncycles,
                        dEhist,
                        dVhist,
                        Emin,
                        Vmin,
                        tempnumber,
                        EVhistogram,
                        beta,
                        p,
                        free_energy,
                    )
            end
        end
    end

    cp = zeros(tempnumber_result)
    vol = zeros(tempnumber_result)
    for i in 1:tempnumber_result
        betat = beta_result[i]
        eenergy = 0
        evolume = 0
        eenthalpy = 0
        eenthalpy2 = 0
        for m in 1:Ebins
            for n in 1:Vbins
                energy_t = Emin + (m - 0.5) * dEhist
                volume = Vmin + (n - 0.5) * dVhist

                qp =
                    quasiprob(
                        betat,
                        m,
                        n,
                        ncycles,
                        dEhist,
                        dVhist,
                        Emin,
                        Vmin,
                        tempnumber,
                        EVhistogram,
                        beta,
                        p,
                        free_energy,
                    ) / normalconst[i]

                eenergy += qp * energy_t
                evolume += qp * volume
                eenthalpy += qp * (energy_t + p * volume)
                eenthalpy2 += qp * (energy_t + p * volume)^2
            end
        end
        cp[i] = (eenthalpy2 - eenthalpy^2) / (k * temp_result[i]^2)
        vol[i] = evolume
    end

    return (; T=temp_result, C=cp, V=vol)
end

end
