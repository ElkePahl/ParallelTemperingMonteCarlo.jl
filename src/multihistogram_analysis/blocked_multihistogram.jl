export BlockedMultiHistogram, equilibration_analysis, bootstrap_heat_capacity_peak

struct BlockedMultiHistogram
    temperature::Vector{Float64}
    beta::Vector{Float64}
    bin_centre::Vector{Float64}
    edges::Vector{Float64}
    weights::Array{Int32,3}
    num_samples::Matrix{Int32}
    block_size::Int
end

# This is done in a single pass because it's faster for large mmapped arrow files.
function num_cycles_and_trajectories(cycle, trajectory_id)
    num_cycles = num_traj = 0
    for (cycle, traj) in zip(cycle, trajectory_id)
        num_cycles = max(num_cycles, Int(cycle))
        num_traj = max(num_traj, Int(traj))
    end
    return num_cycles, num_traj
end
function bin_edges(hamiltonian, num_bins)
    lo, hi = extrema(hamiltonian)
    if lo ≥ hi
        throw(ArgumentError("all Hamiltonian values are equal; cannot construct histogram"))
    end
    return range(lo, hi; length=num_bins+1)
end

function BlockedMultiHistogram(filename::String; kwargs...)
    return BlockedMultiHistogram(Arrow.Table(filename); kwargs...)
end
function BlockedMultiHistogram(tbl; kwargs...)
    return BlockedMultiHistogram(
        tbl.trajectory_id, tbl.temperature, tbl.cycle, tbl.hamiltonian; kwargs...
    )
end
function BlockedMultiHistogram(
    trajectory_id, temperature, cycle, hamiltonian; num_bins=1000, num_blocks=1000
)
    num_cycles, num_traj = num_cycles_and_trajectories(cycle, trajectory_id)
    unique_temps = check_trajectory_temperature_consistency(trajectory_id, temperature)
    beta = inv.(unique_temps .* kB)

    edges = bin_edges(hamiltonian, num_bins)
    bin_centre = (edges[1:(end - 1)] + edges[2:end]) ./ 2

    block_size = num_cycles ÷ num_blocks
    block_size > 0 ||
        throw(ArgumentError("more blocks ($num_blocks) than cycles ($num_cycles)"))
    # trailing cycles that don't fill a block are dropped
    last_cycle = num_blocks * block_size

    weights = zeros(Int32, length(bin_centre), num_traj, num_blocks)
    num_samples = zeros(Int32, num_traj, num_blocks)

    prog = Progress(length(cycle); enabled=isinteractive(), desc="Building histograms...")
    for i in eachindex(trajectory_id, cycle, hamiltonian)
        cycle[i] ≤ last_cycle || continue
        block = (cycle[i] - 1) ÷ block_size + 1
        bin = min(searchsortedlast(edges, hamiltonian[i]), num_bins)
        weights[bin, trajectory_id[i], block] += 1
        num_samples[trajectory_id[i], block] += 1
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)

    return BlockedMultiHistogram(
        unique_temps, beta, bin_centre, edges, weights, num_samples, block_size
    )
end
Base.length(bmh::BlockedMultiHistogram) = size(bmh.weights, 3)
block_size(bmh::BlockedMultiHistogram) = bmh.block_size
num_trajectories(bmh::BlockedMultiHistogram) = size(bmh.weights, 2)
num_bins(bmh::BlockedMultiHistogram) = size(bmh.weights, 1)

function Base.show(io::IO, bmh::BlockedMultiHistogram)
    print(io, "BlockedMultiHistogram with")
    return print(io, " $(length(bmh)) blocks of $(block_size(bmh)) cycles each.")
end

"""
    collapse_target(bmh::BlockedMultiHistogram)

Prepare an empty [`MultiHistogram`](@ref) that is usable as a target for [`collapse!`](@ref).
"""
function collapse_target(bmh::BlockedMultiHistogram)
    weights = zeros(Int, size(bmh.weights, 1), size(bmh.weights, 2))
    num_samples = zeros(Int, size(bmh.num_samples, 1))

    return MultiHistogram(
        bmh.temperature, bmh.beta, bmh.bin_centre, bmh.edges, weights, num_samples
    )
end

"""
    collapse(bmh::BlockedMultiHistogram, indices)

Collapse a [`BlockedMultiHistogram`](@ref) to a [`MultiHistogram`](@ref) by summing the
blocks that correspond to `indices`.
"""
function collapse(bmh::BlockedMultiHistogram, indices)
    return collapse!(collapse_target(bmh), bmh, indices)
end
function collapse!(mh::MultiHistogram, bmh::BlockedMultiHistogram, indices)
    mh.weights .= 0
    mh.num_samples .= 0
    for index in indices
        mh.weights .+= @view bmh.weights[:, :, index]
        mh.num_samples .+= @view bmh.num_samples[:, index]
    end
    return mh
end

"""
    equilibration_analysis(bmh::BlockedMultiHistogram; window_size=10)

Analyse convergence of a Monte Carlo run by merging adjacent `window_size` blocks in a
[`BlockedMultiHistogram`](@ref) and finding the heat capacity peak (see
[`heat_capacity_peak`](@ref)) for each window of blocks.
"""
function equilibration_analysis(
    bmh::BlockedMultiHistogram; window_size=10, points=num_trajectories(bmh) * 10
)
    num_steps = (length(bmh) - window_size + 1)
    critical_temperature = Vector{Union{Missing,Float64}}(undef, num_steps)
    heat_capacity_maximum = Vector{Union{Missing,Float64}}(undef, num_steps)

    prog = Progress(num_steps; desc="Processing...", enabled=isinteractive())
    Threads.@threads for i in 1:num_steps
        mh = collapse(bmh, (i - 1) .+ (1:window_size))
        props = thermodynamic_properties(mh; show_progress=false, points)
        critical_temperature[i], heat_capacity_maximum[i] = heat_capacity_peak(
            props; warn=false
        )
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)

    return DataFrame(;
        block_start=1:num_steps,
        block_end=(1:num_steps) .+ (window_size - 1),
        cycle_start=(0:(num_steps - 1)) .* bmh.block_size .+ 1,
        critical_temperature,
        heat_capacity_maximum,
    )
end

"""
    bootstrap_heat_capacity_peak(
        blocked_multihist::BlockedMultiHistogram; equilibration_blocks, num_samples=1000
    )

Resample [`BlockedMultiHistogram`](@ref) `num_samples` times and compute heat capacity peak
for each sample. Returns a named tuple `(; value, error, samples)`, where `value` is the
critical temperature from all blocks after equilibration, `error` is the standard deviation
of the bootstrapped critical temperatures, and `samples` is a `DataFrame` with the
bootstrapped critical temperatures and heat capacity maxima.

`equilibration_blocks` sets the number of block to skip at the beginning at the run. It must
be set manualy; use [`equilibration_analysis`](@ref) to determine the correct number of
block to skip.
"""
function bootstrap_heat_capacity_peak(
    blocked_multihist::BlockedMultiHistogram; equilibration_blocks, num_samples=1000
)
    selection = (equilibration_blocks + 1):length(blocked_multihist)
    multihist = collapse(blocked_multihist, selection)
    props, free_energy = thermodynamic_properties(
        multihist, zeros(num_trajectories(multihist))
    )
    value = heat_capacity_peak(props)[1]

    critical_temperature = Vector{Union{Missing,Float64}}(undef, num_samples)
    heat_capacity_maximum = Vector{Union{Missing,Float64}}(undef, num_samples)

    prog = Progress(num_samples; desc="Bootstrapping...", enabled=isinteractive())
    Threads.@threads for i in 1:num_samples
        resampled = collapse(blocked_multihist, rand(selection, length(selection)))
        resampled_props, _ = thermodynamic_properties(
            resampled, free_energy; show_progress=false
        )
        critical_temperature[i], heat_capacity_maximum[i] = heat_capacity_peak(
            resampled_props; warn=false
        )
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)
    error = std(skipmissing(critical_temperature))
    return (;
        value, error, samples=DataFrame(; critical_temperature, heat_capacity_maximum)
    )
end
