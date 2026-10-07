using Arrow
using DataFrames

struct Reporter{F<:Union{String,Nothing},W<:Union{Arrow.Writer{IOStream},Nothing}}
    stats::Vector{NamedTuple}
    flush_interval::Int
    filename::F
    writer::W
    metadata::Vector{Pair{String,String}}
    verbose::Bool
    return_stats::Bool
end

function Reporter(
    start_config,
    potential,
    ensemble,
    mc_params;
    flush_interval,
    filename,
    verbose,
    return_stats,
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
    ]

    # Open a writer if flushing
    if !isnothing(filename) && flush_interval ≤ mc_params.mc_cycles
        writer = open(Arrow.Writer, stats_filename; compress=:zstd, metadata)
    else
        writer = nothing
    end
    return Reporter(
        NamedTuple[], flush_interval, filename, writer, metadata, verbose, return_stats
    )
end

function flush!(r::Reporter, mc_cycle)
    if !isnothing(r.writer) && iszero(mc_cycle % r.flush_interval)
        r.verbose && @info "mc_cycle $mc_cycle: Flushing..."
        Arrow.write(r.writer, r.stats)
        empty!(r.stats)
    end
end

report!(r::Reporter, row) = push!(r.stats, row)

function finalise!(r::Reporter)
    if !isnothing(r.filename) && isnothing(r.writer)
        Arrow.write(stats_filename, stats; compress=:zstd, metadata=r.metadata)
        df = DataFrame(Arrow.Table(stats_filename))
    elseif !isnothing(r.filename) && !isnothing(r.writer)
        Arrow.write(writer, stats)
        empty!(stats)
        close(writer)
        df = DataFrame(Arrow.Table(stats_filename))
    else
        df = DataFrame(r.stats)
    end
    foreach(r.metadata) do (key, value)
        return metadata!(df, key, value)
    end
    return df
end
