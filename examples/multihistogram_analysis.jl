# # Heat capacity from a parallel tempering run
#
# This tutorial shows how to go from the output of a parallel tempering Monte Carlo
# simulation to a heat capacity curve, a check that the run is equilibrated, and an
# estimate of the heat capacity peak with an error bar.
#
# We use a simulation of the 55-atom neon cluster between 5 K and 20 K, with 51
# temperatures and 110 000 cycles. The data ships with the package's tests.
#
# Plotting is not part of the package, so any plotting library works. Both
# [Plots.jl](https://docs.juliaplots.org) and [CairoMakie.jl](https://docs.makie.org) are good
# options; this tutorial uses CairoMakie.
#
# The bootstrap and equilibration analysis are multithreaded, so start Julia with several
# threads, e.g. `julia -t auto`.

using ParallelTemperingMonteCarlo
using Arrow
using DataFrames
using CairoMakie

# ## 1. Loading the data and computing the heat capacity
#
# Simulation output is stored in Arrow files with one row per temperature per cycle. The
# analysis needs the columns `cycle`, `trajectory_id`, `temperature`, and `hamiltonian`.
#
# Load the file with `Arrow.Table`. This memory-maps the file instead of reading it into
# memory, which matters for production runs, where files can be several gigabytes. Avoid
# converting to a `DataFrame` unless you need to.

path = pkgdir(ParallelTemperingMonteCarlo, "test", "testing_data", "neon-55-100K.arrow")
data = Arrow.Table(path)

# [`thermodynamic_properties`](@ref) performs the multihistogram analysis. It combines the
# energy histograms from all temperatures into a single density of states, and uses it to
# compute thermodynamic properties on a fine temperature grid.
#
# The first part of the run is discarded as equilibration. The number of discarded cycles
# is read from the file's `eq_cycles` metadata if present; otherwise, as here, the first
# sixth of the run is discarded with a warning. You can also set it with the
# `equilibration_cycles` keyword. See [`MultiHistogram`](@ref) for all options.

properties = thermodynamic_properties(data)

# The result is a `DataFrame` with the temperature, heat capacity, mean and mean square of
# the Hamiltonian, and entropy (up to an additive constant). All quantities are in atomic
# units, and temperatures are in kelvin.

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="T (K)", ylabel="Cᵥ (Eₕ/K)")
lines!(ax, properties.temperature, properties.heat_capacity)
fig

# The peak marks the melting transition of the cluster. [`heat_capacity_peak`](@ref) locates
# it by fitting a parabola through the largest value on the temperature grid and its two
# neighbours, so the result does not depend on the grid spacing.

critical_temperature, heat_capacity_maximum = heat_capacity_peak(properties)

# If the maximum lies at either end of the temperature range, `heat_capacity_peak` returns
# `missing` values and warns: the true peak is then outside the simulated range.
#
# ## 2. Checking equilibration
#
# The analysis above assumes that the simulation has reached equilibrium. To check this, we
# split the run into blocks of consecutive cycles and see how the heat capacity peak changes
# along the run.
#
# A [`BlockedMultiHistogram`](@ref) stores one histogram per block. It reads the data once,
# and all later analysis works on the stored histograms, which is much faster than
# re-reading the file. Note that it keeps all cycles; equilibration is handled by skipping
# blocks later.
#
# We use small settings here to keep the tutorial fast. See the summary for settings
# suitable for production runs.

blocked = BlockedMultiHistogram(data; num_bins=300, num_blocks=20)

# [`equilibration_analysis`](@ref) computes the heat capacity peak for a window of
# `window_size` consecutive blocks, sliding the window along the run one block at a time.

equilibration = equilibration_analysis(blocked; window_size=5)

# Plotting the peak height and position against the first cycle of each window shows how
# the run evolves.

equilibration = dropmissing(equilibration)

fig = Figure()
ax1 = Axis(fig[1, 1]; ylabel="T* (K)")
ax2 = Axis(fig[2, 1]; xlabel="first cycle of window", ylabel="Cᵥ,max (Eₕ/K)")
scatterlines!(ax1, equilibration.cycle_start, equilibration.critical_temperature)
scatterlines!(ax2, equilibration.cycle_start, equilibration.heat_capacity_maximum)
linkxaxes!(ax1, ax2)
hidexdecorations!(ax1; grid=false)
fig

# In the first few windows, the peak moves to lower temperatures and grows: the cluster is
# still relaxing from its starting configuration. After that, both quantities fluctuate
# around a constant value. Consecutive windows share most of their blocks, so neighbouring
# points are strongly correlated, and slow wandering of this size is expected.
#
# A run is equilibrated from the point where the curves stop showing a systematic trend.
# If they keep drifting until the end of the run, or move in large steps, the simulation
# has not converged; no choice of equilibration period will fix that, and the run needs to
# be longer or sample better (e.g. with more temperatures near the transition).
#
# Here, we discard the first four blocks.

equilibration_blocks = 4

# ## 3. Error bars
#
# With a single run, the uncertainty of the peak can be estimated by a block bootstrap.
# [`bootstrap_heat_capacity_peak`](@ref) repeatedly draws blocks at random, with
# replacement, from the equilibrated part of the run, and computes the heat capacity peak
# of each resampled data set. Resampling whole blocks, rather than individual samples,
# preserves the correlations between consecutive cycles and between temperatures.

bootstrap = bootstrap_heat_capacity_peak(blocked; equilibration_blocks, num_samples=100)

# The result contains the critical temperature from all equilibrated blocks, its bootstrap
# standard error, and the individual bootstrap samples.

println("T* = $(round(bootstrap.value; digits=2)) ± $(round(bootstrap.error; digits=2)) K")

# The distribution of the samples shows the full uncertainty, including any asymmetry:

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="T* (K)", ylabel="count")
hist!(ax, collect(skipmissing(bootstrap.samples.critical_temperature)); bins=20)
vlines!(ax, bootstrap.value; color=:black)
fig

# Keep in mind what this error bar does and does not include:
#
# - It measures **statistical** error only. Systematic errors, from an equilibration period
#   that is too short or from structures the simulation never found, are not included.
# - It is only reliable if each block is much longer than the correlation time of the
#   simulation. A simple check is to rebuild the `BlockedMultiHistogram` with fewer, longer
#   blocks: if the error grows, the blocks were too short.
# - If sampling near the transition is poor, the error bar from a single run is a lower
#   bound. Independent runs are the most reliable check.
#
# ## Summary
#
# 1. Load the data with `Arrow.Table`, and get a first look with
#    [`thermodynamic_properties`](@ref) and [`heat_capacity_peak`](@ref).
# 2. Build a [`BlockedMultiHistogram`](@ref) and run [`equilibration_analysis`](@ref).
#    Choose the number of blocks to discard from where the curves stop trending.
# 3. Estimate the error with [`bootstrap_heat_capacity_peak`](@ref) and report
#    T* ± σ.
#
# For production runs, use larger settings than in this tutorial: bins narrower than about
# a tenth of ``k_B T`` at the lowest temperature, a few hundred blocks, and around 1000
# bootstrap samples.
