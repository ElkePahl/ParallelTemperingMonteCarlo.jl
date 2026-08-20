module MultiHistogramAnalysis

using ..InputParams: TempGrid
using ProgressMeter: ProgressMeter, ProgressThresh
using LogExpFunctions: logsumexp
using DataFrames: DataFrame
using LinearAlgebra: dot
export MultiHistogram, num_trajectories, num_bins, thermodynamic_properties

const kB = 3.16681196e-6

"""
    check_trajectory_temperature_consistency(traj_id, temperature)

Return the temperature associated with each trajectory, checking that each
trajectory has exactly one consistent temperature.
"""
function check_trajectory_temperature_consistency(traj_id, temperature)
    num_traj = maximum(traj_id)
    seen = fill(false, num_traj)
    temp_of_traj = fill(NaN, num_traj)
    @inbounds for i in eachindex(traj_id, temperature)
        t = traj_id[i]
        T = temperature[i]
        if !seen[t]
            seen[t] = true
            temp_of_traj[t] = T
        elseif temp_of_traj[t] != T
            throw(ArgumentError("inconsistent temperature in row $i"))
        end
    end
    if !all(seen)
        throw(ArgumentError("traj_id values $(findall(!, seen)) never appear in the data"))
    end
    return temp_of_traj
end

"""
    MultiHistogram(df; num_bins=100, skip_ratio=1/11)

Construct a `MultiHistogram` from parallel tempering Monte Carlo data in `df` for use with
multihistogram analysis.

`df` is a `DataFrame` that contains the columns `trajectory_id`, `temperature`, `cycle`, and
`hamiltonian`. Each trajectory is assumed to correspond to a fixed temperature. The
generalized Hamiltonian is ``E`` for an [`NVT`](@ref) ensemble and ``E + pV`` for an
[`NPT`](@ref) ensemble.

# Arguments
- `df`: DataFrame containing the Monte Carlo samples.
- `num_bins=100`: Number of equally spaced Hamiltonian bins.
- `skip_ratio=1/11`: Fraction of the simulation discarded as initial
  equilibration. Samples with `cycle ≤ round(maximum(cycle) * skip_ratio)`
  are discarded.

# Fields
- `num_bins`: Number of Hamiltonian bins.
- `temperature`: Temperature associated with each trajectory.
- `beta`: Inverse thermal energy, `1 / (kB * temperature)`, for each trajectory.
- `bin_centre`: Hamiltonian value at the centre of each bin.
- `edges`: Hamiltonian bin edges, with `num_bins + 1` entries.
- `weights`: Histogram counts, indexed by `(bin, trajectory)`.
- `num_samples`: Total number of retained samples for each trajectory.
"""
struct MultiHistogram
    temperature::Vector{Float64}
    beta::Vector{Float64}
    bin_centre::Vector{Float64}
    edges::Vector{Float64}
    weights::Matrix{Int}
    num_samples::Vector{Int}
end

"""
    num_trajectories(mh)

Return the number of trajectories in `mh`.
"""
num_trajectories(mh::MultiHistogram) = size(mh.weights, 2)
"""
    num_bins(mh)

Return the number of histogram bins in `mh`.
"""
num_bins(mh::MultiHistogram) = size(mh.weights, 1)

function MultiHistogram(df; kwargs...)
    return MultiHistogram(
        df.trajectory_id, df.temperature, df.cycle, df.hamiltonian; kwargs...
    )
end

function MultiHistogram(
    traj_id, temperature, cycle, hamiltonian; num_bins=100, skip_ratio=1 / 11
)
    if !(length(traj_id) == length(temperature) == length(cycle) == length(hamiltonian))
        throw(
            DimensionMismatch(
                "lengths of `trajectory_id`, `temperature`, `cycle`, and/or `hamiltonian` do not match!",
            ),
        )
    end
    if num_bins ≤ 2
        throw(ArgumentError("`num_bins` must be at least 3"))
    elseif skip_ratio < 0 || skip_ratio ≥ 1
        throw(ArgumentError("`skip_ratio` must be in range [0, 1)"))
    end

    max_cycle = maximum(cycle)
    first_used = round(Int, max_cycle * skip_ratio)

    # Find range of data, excluding initial equilibration phase
    lo = Inf
    hi = -Inf
    for i in eachindex(hamiltonian)
        if cycle[i] > first_used
            lo = min(lo, hamiltonian[i])
            hi = max(hi, hamiltonian[i])
        end
    end
    if !isfinite(lo) || !isfinite(hi) || lo == hi
        throw(
            ArgumentError("cannot construct histogram. Consider decreasing `skip_ratio`.")
        )
    end

    num_traj = maximum(traj_id)
    unique_temps = check_trajectory_temperature_consistency(traj_id, temperature)

    ΔH = (hi - lo) / num_bins
    edges = collect(range(lo, hi; length=num_bins + 1))
    weights = zeros(Int, num_bins, num_traj)

    for j in eachindex(hamiltonian)
        if cycle[j] > first_used
            H = hamiltonian[j]
            traj = traj_id[j]

            bin = floor(Int, (H - lo) / ΔH) + 1
            bin = clamp(bin, 1, num_bins)
            weights[bin, traj] += 1
        end
    end

    num_samples = vec(sum(weights; dims=1))

    beta = inv.(unique_temps .* kB)
    bin_centre = (edges[1:(end - 1)] + edges[2:end]) ./ 2
    return MultiHistogram(unique_temps, beta, bin_centre, edges, weights, num_samples)
end

"""
    update_log_denominator!(log_denominator, mh, free_energy)

Update the logarithmic denominator of the multihistogram weights for the given free
energies.
"""
function update_log_denominator!(log_denominator, mh::MultiHistogram, free_energy)
    Threads.@threads for k in 1:num_bins(mh)
        H = mh.bin_centre[k]
        log_denominator[k] = logsumexp(
            log(mh.num_samples[j]) - mh.beta[j] * H + free_energy[j] for
            j in 1:num_trajectories(mh)
        )
    end
    return log_denominator
end

"""
    get_log_weights(mh, tol, maxiter)

Compute the logarithm of the unnormalised multihistogram density of states
using self-consistent iteration of the trajectory free energies.

Iteration stops when the sum of squared changes in the free energies is below
`tol`, or when `maxiter` iterations have been reached.
"""
function get_log_weights(mh::MultiHistogram, tol, maxiter)
    tol > 0 || throw(ArgumentError("`tol` must be positive"))
    maxiter > 0 || throw(ArgumentError("`maxiter` must be positive"))

    log_numerator = zeros(num_bins(mh))
    for k in 1:num_bins(mh)
        total = sum(mh.weights[k, j] for j in 1:num_trajectories(mh))
        log_numerator[k] = iszero(total) ? -Inf : log(total)
    end

    free_energy = zeros(num_trajectories(mh))
    new_free_energy = zeros(num_trajectories(mh))
    log_denominator = zeros(num_bins(mh))

    progress = ProgressThresh(tol)
    for iter in 1:maxiter
        update_log_denominator!(log_denominator, mh, free_energy)

        # Update the free energy.
        for i in 1:num_trajectories(mh)
            beta = mh.beta[i]
            new_free_energy[i] =
                -logsumexp(
                    log_numerator[k] - log_denominator[k] - beta * mh.bin_centre[k] for
                    k in 1:num_bins(mh)
                )
        end
        new_free_energy .-= maximum(new_free_energy)

        # Check convergence.
        delta = sum(abs2, new_free_energy .- free_energy)
        ProgressMeter.update!(progress, delta)
        isnan(delta) && error("diverged while computing free energy")

        if delta < tol
            update_log_denominator!(log_denominator, mh, new_free_energy)
            return log_numerator - log_denominator
        end
        free_energy, new_free_energy = new_free_energy, free_energy
    end
    return error("Failed to converge in $maxiter iterations.")
end

"""
    thermodynamic_properties(data; tol=1e-12, maxiter=5000, points=nothing, kwargs...)
    thermodynamic_properties(mh::MultiHistogram; points, tol=1e-12, maxiter=5000)

Compute thermodynamic properties from multihistogram analysis.

For `data`, `kwargs...` are passed to `MultiHistogram`. `points` specifies the
number of temperatures at which to evaluate the properties; by default it is
`10 * num_trajectories(mh)`. `tol` and `maxiter` control the free-energy
iteration (convergence tolerance and number of iterations).

Returns a `DataFrame` containing `temperature`, `heat_capacity`, `hamiltonian`,
`hamiltonian_squared`, and `entropy`.
"""
function thermodynamic_properties(data; tol=1e-12, maxiter=5000, points=nothing, kwargs...)
    mh = MultiHistogram(data; kwargs...)
    if isnothing(points)
        points = 10 * num_trajectories(mh)
    end
    return thermodynamic_properties(mh; points, tol, maxiter)
end
function thermodynamic_properties(
    mh::MultiHistogram; points=10 * num_trajectories(mh), tol=1e-12, maxiter=5000
)
    points > 0 || throw(ArgumentError("`points` must be positive"))

    log_weights = get_log_weights(mh, tol, maxiter)

    temp_grid_result = TempGrid{points}(extrema(mh.temperature)...)
    temperatures_result = temp_grid_result.t_grid
    betas_result = temp_grid_result.beta_grid

    heat_capacity = zeros(points)
    hamiltonian = zeros(points)
    hamiltonian_squared = zeros(points)
    entropy = zeros(points)
    bin_centre = mh.bin_centre
    bin_centre_squared = bin_centre .^ 2

    Threads.@threads for i in 1:points
        β = betas_result[i]
        log_q = log_weights .- β .* bin_centre
        log_Z = logsumexp(log_q)
        weights = exp.(log_q .- log_Z)

        H = dot(weights, bin_centre)
        H² = dot(weights, bin_centre_squared)

        hamiltonian[i] = H
        hamiltonian_squared[i] = H²
        heat_capacity[i] = (H² - H^2) / (kB * temperatures_result[i]^2)
        entropy[i] = kB * (log_Z + β * H)
    end

    return DataFrame(;
        temperature=temperatures_result,
        heat_capacity,
        hamiltonian,
        hamiltonian_squared,
        entropy,
    )
end
end
