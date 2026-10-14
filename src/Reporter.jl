using Arrow
using DataFrames: DataFrames, metadata!

"""
    Reporter(
        start_config,
        potential,
        ensemble,
        mc_params;
        flush_interval,
        filename,
        verbose,
        return_stats,
        extra_metadata,
    )

Handles reporting statistics to `DataFrame` during a Monte Carlo run. `start_config`,
`potential`, `ensemble`, and `mc_params` are used to construct the result's metadata.

## Keyword arguments
- `filename`: sets the filename where the `DataFrame` is stored. If set to `nothing`,
  writing to file is disabled.
- `flush_interval`: sets how often the `DataFrame` is flushed to disk.
- `verbose`: if set to `true`, a message is printed when flushing.
- `return_stats`: if set to `true`, the full `DataFrame` is returned at the end of a
  computation.
- `extra_metadata`: additional metadata to be recorded in the `DataFrame`. Should be an
  iterable of key-value pairs. Both the keys and the values are converted to strings.
"""
struct Reporter{F<:Union{String,Nothing},W<:Union{Arrow.Writer{IOStream},Nothing}}
    stats::Vector{NamedTuple}
    flush_interval::Int
    filename::F
    writer::W
    metadata::Vector{Pair{String,String}}
    verbose::Bool
    mc_cycles::Int
    return_stats::Bool
end

function Reporter(
    start_config,
    potential,
    ensemble,
    mc_params;
    filename,
    flush_interval,
    verbose,
    return_stats,
    extra_metadata,
)
    # Make sure filename does not already exist.
    if !isnothing(filename)
        counter = 0
        base, ext = splitext(filename)
        while isfile(filename)
            counter += 1
            new_filename = string("$base-$counter", ext)
            @warn "File $filename exists. Using $new_filename"
            filename = new_filename
        end
    end
    mc_cycles = mc_params.mc_cycles

    metadata = [
        "potential" => repr(potential),
        "ensemble" => repr(ensemble),
        "start_config" => summary(start_config),
        "eq_cycles" => string(mc_params.eq_cycles),
        "mc_cycles" => string(mc_params.mc_cycles),
        "min_acc" => string(mc_params.min_acc),
        "max_acc" => string(mc_params.max_acc),
        "n_atoms" => string(mc_params.n_atoms),
        "n_traj" => string(mc_params.n_traj),
        "mc_sample" => string(mc_params.mc_sample),
        "n_atoms" => string(mc_params.n_atoms),
        "n_bin" => string(mc_params.n_bin),
        "timestamp" => string(time()),
    ]
    if !isnothing(extra_metadata)
        for (k, v) in extra_metadata
            push!(metadata, string(k) => string(v))
        end
    end

    # Open a writer if flushing
    if !isnothing(filename) && flush_interval ≤ mc_params.mc_cycles
        writer = open(Arrow.Writer, filename; compress=:zstd, metadata)
    else
        writer = nothing
    end

    if !isnothing(writer)
        verbose && @info "Starting MC computation." mc_cycles flush_interval filename
    else
        verbose && @info "Starting MC computation." mc_cycles filename
    end

    return Reporter(
        NamedTuple[],
        flush_interval,
        filename,
        writer,
        metadata,
        verbose,
        mc_cycles,
        return_stats,
    )
end

"""
    flush!(r::Reporter, mc_cycle)

Attempt to flush the data to file. Whether the data is actually flushed is determined from
`mc_cycle`.
"""
function flush!(r::Reporter, mc_cycle)
    if !isnothing(r.writer) && iszero(mc_cycle % r.flush_interval)
        if r.verbose
            cycle_str = lpad(mc_cycle, length(string(r.mc_cycles)))
            @info "mc_cycle $(cycle_str) / $(r.mc_cycles)"
        end
        !isempty(r.stats) && Arrow.write(r.writer, r.stats)
        empty!(r.stats)
    end
end

"""
    report!(r::Reporter, row)

Add a row to the report.
"""
report!(r::Reporter, row) = push!(r.stats, row)

"""
    finalise!(r::Reporter)

Finalise the reporter: flush all remaining data to file (if flushing), convert the result
to a `DataFrame` and return it.
"""
function finalise!(r::Reporter)
    if !isnothing(r.filename) && isnothing(r.writer)
        Arrow.write(r.filename, r.stats; compress=:zstd, metadata=r.metadata)
        df = DataFrame(Arrow.Table(r.filename))
    elseif !isnothing(r.filename) && !isnothing(r.writer)
        !isempty(r.stats) && Arrow.write(r.writer, r.stats)
        close(r.writer)
        df = DataFrame(Arrow.Table(r.filename))
    else
        df = DataFrame(r.stats)
        for (key, value) in r.metadata
            metadata!(df, key, value)
        end
    end
    r.verbose && @info "Done."
    return df
end
