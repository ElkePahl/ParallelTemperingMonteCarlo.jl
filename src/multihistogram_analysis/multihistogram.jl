export MultiHistogram,
    num_trajectories, num_bins, thermodynamic_properties, heat_capacity_peak

"""
    check_trajectory_temperature_consistency(traj_id, temperature)

Return the temperature associated with each trajectory, checking that each
trajectory has exactly one consistent temperature.
"""
function check_trajectory_temperature_consistency(traj_id, temperature)
    num_traj = maximum(traj_id)
    seen = fill(false, num_traj)
    temp_of_traj = fill(NaN, num_traj)
    @inbounds for i in 1:length(traj_id)
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
    MultiHistogram(df; num_bins=100, equilibration_cycles, stop_at, window_size, limits)

Construct a `MultiHistogram` from parallel tempering Monte Carlo data in `df` for use with
multihistogram analysis.

`df` is a `DataFrame` that contains the columns `trajectory_id`, `temperature`, `cycle`, and
`hamiltonian`. Each trajectory is assumed to correspond to a fixed temperature. The
generalized Hamiltonian is ``E`` for an [`NVT`](@ref) ensemble, ``E + pV`` for an
[`NPT`](@ref) ensemble, and ``E + pV + L_0 σ ⋅ L`` for the ``NσT`` (see [`NPT`](@ref))
ensemble.

# Arguments
- `df`: DataFrame containing the Monte Carlo samples.
- `num_bins=100`: Number of equally spaced Hamiltonian bins.
- `equilibration_cycles`: Number of steps of the simulation discarded as initial
  equilibration. Samples with `cycle ≤ equilibration_cycles` are discarded. Defaults to
  reading the value from `df`'s metadata.
- `stop_at`: Last cycle to include. Defaults to the last cycle in the data.
- `window_size`: Number of cycles to include, starting after equilibration. Mutually
  exclusive with `stop_at`.
- `limits`: `(lo, hi)` range of the Hamiltonian histogram. Samples outside it are
  discarded. Defaults to the range of the selected data.

`df` must be sorted by `cycle`.

# Fields
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
        df.trajectory_id, df.temperature, df.cycle, df.hamiltonian, metadata(df); kwargs...
    )
end

function MultiHistogram(
    traj_id,
    temperature,
    cycle,
    hamiltonian,
    metadata;
    num_bins=100,
    equilibration_cycles=nothing,
    stop_at=nothing,
    window_size=nothing,
    limits=nothing,
)
    if !(length(traj_id) == length(temperature) == length(cycle) == length(hamiltonian))
        throw(
            DimensionMismatch(
                "lengths of `trajectory_id`, `temperature`, `cycle`, and/or `hamiltonian` do not match!",
            ),
        )
    end
    issorted(cycle) || throw(ArgumentError("the data should be sorted by `cycle`"))
    if num_bins ≤ 2
        throw(ArgumentError("`num_bins` must be at least 3"))
    elseif !isnothing(equilibration_cycles) && equilibration_cycles < 0
        throw(ArgumentError("`equilibration_cycles` must be non-negative"))
    elseif !isnothing(stop_at) && !isnothing(window_size)
        throw(ArgumentError("specify at most one of `stop_at` and `window_size`"))
    elseif !isnothing(window_size) && window_size ≤ 0
        throw(ArgumentError("`window_size` must be positive"))
    end

    num_cycles = maximum(cycle; init=0)
    if isnothing(equilibration_cycles)
        eq_cycles = get(metadata, "eq_cycles", nothing)
        if isnothing(eq_cycles)
            @warn "metadata does not contain `eq_cycles`. Using `1/6 * num_cycles`."
            first_used = round(Int, num_cycles / 6) + 1
        else
            first_used = parse(Int, eq_cycles) + 1
        end
    else
        first_used = equilibration_cycles + 1
    end

    if !isnothing(window_size)
        last_used = first_used + window_size - 1
    elseif !isnothing(stop_at)
        last_used = stop_at
    else
        last_used = num_cycles
    end
    if last_used < first_used
        throw(
            ArgumentError(
                "no cycles selected: first used $first_used, last used $last_used"
            ),
        )
    elseif last_used > num_cycles
        throw(
            ArgumentError("requested cycles up to $last_used, but data ends at $num_cycles")
        )
    end

    rows = searchsortedfirst(cycle, first_used):searchsortedlast(cycle, last_used)
    isempty(rows) && throw(ArgumentError("no samples in cycles $first_used:$last_used"))

    if isnothing(limits)
        lo, hi = extrema(@view hamiltonian[rows])
    else
        lo, hi = limits
    end
    if !isfinite(lo) || !isfinite(hi) || lo ≥ hi
        throw(
            ArgumentError(
                "invalid Hamiltonian range ($lo, $hi); cannot construct histogram"
            ),
        )
    end

    num_traj = maximum(traj_id; init=0)
    unique_temps = check_trajectory_temperature_consistency(traj_id, temperature)

    ΔH = (hi - lo) / num_bins
    edges = collect(range(lo, hi; length=num_bins + 1))
    weights = zeros(Int, num_bins, num_traj)

    for j in rows
        H = hamiltonian[j]
        if !(lo ≤ H ≤ hi)
            continue
        end
        bin = min(floor(Int, (H - lo) / ΔH) + 1, num_bins)  # H == hi → last bin
        weights[bin, traj_id[j]] += 1
    end

    num_samples = vec(sum(weights; dims=1))
    sum(num_samples) > 0 || throw(ArgumentError("no samples within limits ($lo, $hi)"))

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
    get_log_weights!(mh, tol, maxiter, show_progress, free_energy)

Compute the logarithm of the unnormalised multihistogram density of states
using self-consistent iteration of the trajectory free energies.

Iteration stops when the sum of squared changes in the free energies is below
`tol`, or when `maxiter` iterations have been reached.
"""
function get_log_weights!(mh::MultiHistogram, tol, maxiter, show_progress, free_energy)
    tol > 0 || throw(ArgumentError("`tol` must be positive"))
    maxiter > 0 || throw(ArgumentError("`maxiter` must be positive"))

    log_numerator = zeros(num_bins(mh))
    for k in 1:num_bins(mh)
        total = sum(mh.weights[k, j] for j in 1:num_trajectories(mh))
        log_numerator[k] = iszero(total) ? -Inf : log(total)
    end

    new_free_energy = similar(free_energy)
    log_denominator = zeros(num_bins(mh))

    progress = ProgressThresh(tol; enabled=show_progress)
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
            return log_numerator - log_denominator, new_free_energy
        end
        free_energy, new_free_energy = new_free_energy, free_energy
    end
    return error("Failed to converge in $maxiter iterations.")
end

"""
    thermodynamic_properties(data; tol=1e-12, maxiter=5000, points=nothing, kwargs...)
    thermodynamic_properties(
        mh::MultiHistogram, [free_energy=nothing];
        points, tol=1e-12, maxiter=5000
    )

Compute thermodynamic properties from multihistogram analysis.

For `data`, `kwargs...` are passed to `MultiHistogram`. `points` specifies the
number of temperatures at which to evaluate the properties; by default it is
`10 * num_trajectories(mh)`. `tol` and `maxiter` control the free-energy
iteration (convergence tolerance and number of iterations).

Returns a `DataFrame` containing `temperature`, `heat_capacity`, `hamiltonian`,
`hamiltonian_squared`, and `entropy`. If free energy is passed, free energy is returned as
well.
"""
function thermodynamic_properties(data; tol=1e-12, maxiter=5000, points=nothing, kwargs...)
    mh = MultiHistogram(data; kwargs...)
    if isnothing(points)
        points = 10 * num_trajectories(mh)
    end
    return thermodynamic_properties(mh; points, tol, maxiter)
end
function thermodynamic_properties(
    mh::MultiHistogram,
    free_energy_init=nothing;
    points=10 * num_trajectories(mh),
    tol=1e-12,
    maxiter=5000,
    show_progress=isinteractive(),
)
    points > 0 || throw(ArgumentError("`points` must be positive"))
    if isnothing(free_energy_init)
        free_energy = zeros(num_trajectories(mh))
    else
        free_energy = copy(free_energy_init) # to make it thread-safe
    end

    log_weights, free_energy = get_log_weights!(
        mh, tol, maxiter, show_progress, free_energy
    )

    temp_grid_result = TempGrid{Int(points)}(extrema(mh.temperature)...)
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

    result = DataFrame(;
        temperature=temperatures_result,
        heat_capacity,
        hamiltonian,
        hamiltonian_squared,
        entropy,
    )
    if isnothing(free_energy_init)
        return result
    else
        return result, free_energy
    end
end

"""
    heat_capacity_peak(df; x_name=:temperature, y_name=:heat_capacity, warn=true)
    heat_capacity_peak(temperature, heat_capacity; warn=true)

Find the peak in the heat capacity curve by fitting a parabola to the global maximum and its
neighbouring points. Returns the temperature and the heat capacity value at that point.

If the maximum lies at either end of the curve, returns `(missing, missing)` and, if `warn`
is set, emits a warning.
"""
function heat_capacity_peak(df; x_name=:temperature, y_name=:heat_capacity, warn=true)
    return heat_capacity_peak(df[!, x_name], df[!, y_name]; warn)
end
function heat_capacity_peak(temperature, heat_capacity; warn=true)
    if length(temperature) ≠ length(heat_capacity)
        throw(ArgumentError("`temperature` and `heat_capacity` must have the same length."))
    end
    index_of_max = argmax(heat_capacity)

    if index_of_max == 1
        warn &&
            @warn "Maximum found in the first index. Repeat computation for a wider temperature range."
        return (missing, missing)
    elseif index_of_max == length(heat_capacity)
        warn &&
            @warn "Maximum found in the last index. Repeat computation for a wider temperature range."
        return (missing, missing)
    end

    # fit parabola to 3 points around the maximum
    x1, x2, x3 = temperature[(index_of_max - 1):(index_of_max + 1)]
    y1, y2, y3 = heat_capacity[(index_of_max - 1):(index_of_max + 1)]

    x_max =
        0.5 * (y1 * (x2^2 - x3^2) + y2 * (x3^2 - x1^2) + y3 * (x1^2 - x2^2)) /
        (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2))
    a =
        (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) /
        ((x1 - x2) * (x1 - x3) * (x2 - x3))
    y_max = y2 - a*(x2 - x_max)^2
    @assert y_max ≥ y2

    return (x_max, y_max)
end
