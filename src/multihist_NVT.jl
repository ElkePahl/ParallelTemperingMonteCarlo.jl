module Multihistogram_NVT

using DelimitedFiles, LinearAlgebra, StaticArrays, ProgressMeter

using ..InputParams
using ..EnergyEvaluation
using ..Ensembles
using ..CustomTypes
export multihistogram_NVT

function temp_trajectories(temp::TempGrid)
    tempnumber = length(temp.t_grid)
    tempnumber_result = tempnumber * 10
    return tempnumber, tempnumber_result
end

function histogram_initialise_en(ensemble::NVT, temp::TempGrid, results::Output)
    k = 3.166811429 / 10^6
    temp_o = temp.t_grid
    beta = temp.beta_grid
    Emin = results.en_min
    Emax = results.en_max
    Ebins = results.n_bin
    dEhist = (Emax - Emin) / Ebins
    ENhistogram = results.en_histogram
    return k, temp_o, beta, Emin, Ebins, dEhist, ENhistogram
end

function Temp_grid_result(ti::Number, tf::Number, tempnumber_result::Int)
    temp_grid_result = TempGrid{tempnumber_result}(ti, tf)
    temp_result = temp_grid_result.t_grid
    beta_result = temp_grid_result.beta_grid
    return temp_result, beta_result
end
const NVTHistogram = Vector{Vector{Float64}}
function free_energy_initialise(
    ENhistogram::NVTHistogram, Ebins::Int, tempnumber::Int, tempnumber_result::Int
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
            ncycles[i] = ncycles[i] + ENhistogram[i][m + 1]
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
    ncycles::VorS,
    dEhist::Number,
    Emin::Number,
    tempnumber::Int,
    ENhistogram::Vector{Vector{N}},
    beta::VorS,
    free_energy::VorS,
) where {N<:Number}
    energy_t = Emin + (m - 0.5) * dEhist
    quasiprob = 0
    denom = 0
    offset = -10^6
    for i in 1:tempnumber
        offset = max(offset, -beta[i] * energy_t - free_energy[i])
    end
    offset = offset + log(10^3)
    for i in 1:tempnumber
        quasiprob = quasiprob + ENhistogram[i][m + 1]
        denom = denom + ncycles[i] * exp(-beta[i] * energy_t - free_energy[i] - offset)
    end

    quasiprob = quasiprob / denom * exp(-betat * energy_t - offset)
    return quasiprob
end

"""
    multihistogram_NVT(ensemble::AbstractEnsemble, temp::TempGrid, results::Output, conv_threshold::Number, readfile::Bool; show_progress=true)
Multihistogram analysis for NVT:
-   `conv_threshold` is the convergence threshold, which user can choose.
-   `readfile` can only be false.
-   Example: `multihistogram_NVT(ensemble, temp, results, 10^(-3), false)`
"""
function multihistogram_NVT(
    ensemble::AbstractEnsemble,
    temp::TempGrid,
    results::Output,
    conv_threshold::Number,
    readfile::Bool;
    max_iter=1000,
    show_progress=true,
)
    if readfile == false
        tempnumber, tempnumber_result = temp_trajectories(temp)
        k, temp_o, beta, Emin, Ebins, dEhist, ENhistogram = histogram_initialise_en(
            ensemble, temp, results
        )
    end
    temp_result, beta_result = Temp_grid_result(
        temp_o[1], temp_o[tempnumber], tempnumber_result
    )

    free_energy, new_free_energy, normalconst, ncycles = free_energy_initialise(
        ENhistogram, Ebins, tempnumber, tempnumber_result
    )

    progress = ProgressThresh(conv_threshold; enabled=show_progress)
    for it in 1:max_iter
        for i in 1:tempnumber
            betat = beta[i]
            new_free_energy[i] = 0
            for m in 1:Ebins
                new_free_energy[i] =
                    new_free_energy[i] + quasiprob(
                        betat,
                        m,
                        ncycles,
                        dEhist,
                        Emin,
                        tempnumber,
                        ENhistogram,
                        beta,
                        free_energy,
                    )
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
            break
        end
    end

    for i in 1:tempnumber_result
        betat = beta_result[i]
        for m in 1:Ebins
            normalconst[i] =
                normalconst[i] + quasiprob(
                    betat,
                    m,
                    ncycles,
                    dEhist,
                    Emin,
                    tempnumber,
                    ENhistogram,
                    beta,
                    free_energy,
                )
        end
    end

    cv = zeros(tempnumber_result)
    for i in 1:tempnumber_result
        betat = beta_result[i]
        eenergy = 0
        eenergy2 = 0
        for m in 1:Ebins
            energy_t = Emin + (m - 0.5) * dEhist

            eenergy =
                eenergy +
                quasiprob(
                    betat,
                    m,
                    ncycles,
                    dEhist,
                    Emin,
                    tempnumber,
                    ENhistogram,
                    beta,
                    free_energy,
                ) / normalconst[i] * energy_t
            eenergy2 =
                eenergy2 +
                quasiprob(
                    betat,
                    m,
                    ncycles,
                    dEhist,
                    Emin,
                    tempnumber,
                    ENhistogram,
                    beta,
                    free_energy,
                ) / normalconst[i] * energy_t^2
        end
        cv[i] = (eenergy2 - eenergy^2) / (k * temp_result[i]^2)
    end
    return (; T=temp_result, C=cv)
end

end
