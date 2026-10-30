using Test
using Arrow
using DataFrames
using Statistics: mean, var
using ParallelTemperingMonteCarlo
using ParallelTemperingMonteCarlo.MultiHistogramAnalysis: kB

const MHA = ParallelTemperingMonteCarlo.MultiHistogramAnalysis

neon() = Arrow.Table(joinpath(@__DIR__, "testing_data/neon-55-100K.arrow"))
chunked() = Arrow.Table(joinpath(@__DIR__, "testing_data/chunked.arrow"))

# Build a small cycle-sorted DataFrame from a (cycles × trajectories) matrix of energies.
function synthetic_df(hamiltonian::AbstractMatrix, temperatures)
    num_cycles, num_traj = size(hamiltonian)
    return DataFrame(;
        cycle=repeat(1:num_cycles; inner=num_traj),
        trajectory_id=repeat(1:num_traj, num_cycles),
        temperature=repeat(temperatures, num_cycles),
        hamiltonian=vec(permutedims(hamiltonian)),
    )
end

# Rows: cycles, columns: trajectories at 300, 400 and 500 K.
const H_SMALL = [
    -10.0 -8.0 -5.0
    -9.0 -6.0 -3.0
    -8.0 -5.0 -2.0
    -7.0 -4.0 -1.0
]
const T_SMALL = [300.0, 400.0, 500.0]

@testset "check_trajectory_temperature_consistency" begin
    @test MHA.check_trajectory_temperature_consistency(
        [1, 2, 1, 2], [3.0, 4.0, 3.0, 4.0]
    ) == [3.0, 4.0]
    @test MHA.check_trajectory_temperature_consistency(Int16[2, 1], [4.0, 3.0]) ==
        [3.0, 4.0]
    @test_throws ArgumentError MHA.check_trajectory_temperature_consistency(
        [1, 1], [3.0, 3.5]
    )
    @test_throws ArgumentError MHA.check_trajectory_temperature_consistency(
        [1, 3], [3.0, 4.0]
    )
end

@testset "MultiHistogram" begin
    df = synthetic_df(H_SMALL, T_SMALL)

    @testset "construction" begin
        mh = MultiHistogram(df; num_bins=4, equilibration_cycles=0)

        @test mh isa MultiHistogram
        @test num_trajectories(mh) == 3
        @test num_bins(mh) == 4
        @test mh.temperature == T_SMALL
        @test mh.beta ≈ 1 ./ (kB .* mh.temperature)
        @test mh.edges ≈ range(-10.0, -1.0; length=5)
        @test mh.bin_centre ≈ (mh.edges[1:(end - 1)] .+ mh.edges[2:end]) ./ 2
        @test size(mh.weights) == (4, 3)
        @test mh.num_samples == [4, 4, 4]
        @test vec(sum(mh.weights; dims=1)) == mh.num_samples
        # Bin width 2.25: [-10, -7.75), [-7.75, -5.5), [-5.5, -3.25), [-3.25, -1]
        @test mh.weights[:, 1] == [3, 1, 0, 0]
        @test mh.weights[:, 2] == [1, 1, 2, 0]
        @test mh.weights[:, 3] == [0, 0, 1, 3]
    end

    @testset "maximum lands in last bin" begin
        mh = MultiHistogram(df; num_bins=3, equilibration_cycles=0)
        @test mh.weights[end, 3] ≥ 1
        @test sum(mh.weights) == length(H_SMALL)
    end

    @testset "equilibration is discarded" begin
        mh = MultiHistogram(df; num_bins=4, equilibration_cycles=2)

        # first_used = 3, so cycles 1 and 2 are discarded.
        @test sum(mh.weights) == 6
        @test mh.num_samples == [2, 2, 2]
        @test mh.edges[1] == -8.0
        @test mh.edges[end] == -1.0
    end

    @testset "equilibration is read from metadata" begin
        df_meta = copy(df)
        metadata!(df_meta, "eq_cycles", "2")
        mh = MultiHistogram(df_meta; num_bins=4)

        @test sum(mh.weights) == 6
        @test mh.num_samples == [2, 2, 2]
    end

    @testset "default equilibration warns" begin
        mh = @test_logs (:warn, r"eq_cycles") MultiHistogram(df; num_bins=4)
        # round(4 / 6) + 1 = 2, so cycle 1 is discarded.
        @test mh.num_samples == [3, 3, 3]
    end

    @testset "stop_at and window_size" begin
        mh = MultiHistogram(df; num_bins=4, equilibration_cycles=0, stop_at=2)
        @test mh.num_samples == [2, 2, 2]
        @test mh.edges[1] == -10.0
        @test mh.edges[end] == -3.0

        mh = MultiHistogram(df; num_bins=4, equilibration_cycles=1, window_size=2)
        @test mh.num_samples == [2, 2, 2]
        @test mh.edges[1] == -9.0
        @test mh.edges[end] == -2.0

        mh_stop = MultiHistogram(df; num_bins=4, equilibration_cycles=1, stop_at=3)
        @test mh_stop.weights == mh.weights
    end

    @testset "limits" begin
        mh = MultiHistogram(df; num_bins=5, equilibration_cycles=0, limits=(-12.0, -2.0))
        @test mh.edges[1] == -12.0
        @test mh.edges[end] == -2.0
        # The sample at H = -1 falls outside the limits.
        @test mh.num_samples == [4, 4, 3]
    end

    @testset "DataFrame and vector constructors agree" begin
        mh1 = MultiHistogram(df; num_bins=4, equilibration_cycles=0)
        mh2 = MultiHistogram(
            df.trajectory_id,
            df.temperature,
            df.cycle,
            df.hamiltonian,
            Dict();
            num_bins=4,
            equilibration_cycles=0,
        )

        for field in fieldnames(MultiHistogram)
            @test getfield(mh1, field) == getfield(mh2, field)
        end
    end

    @testset "input validation" begin
        @test_throws DimensionMismatch MultiHistogram(
            [1, 2], [300.0], [1, 2], [-1.0, -2.0], Dict()
        )
        @test_throws ArgumentError MultiHistogram(df; num_bins=2)
        @test_throws ArgumentError MultiHistogram(df; equilibration_cycles=-1)
        @test_throws ArgumentError MultiHistogram(df; equilibration_cycles=size(df, 1))
        @test_throws ArgumentError MultiHistogram(
            df; equilibration_cycles=0, stop_at=2, window_size=2
        )
        @test_throws ArgumentError MultiHistogram(df; equilibration_cycles=0, window_size=0)
        @test_throws ArgumentError MultiHistogram(df; equilibration_cycles=0, stop_at=10)
        @test_throws ArgumentError MultiHistogram(
            df; equilibration_cycles=0, window_size=10
        )
        @test_throws ArgumentError MultiHistogram(
            df; equilibration_cycles=0, limits=(1.0, 2.0)
        )
        @test_throws ArgumentError MultiHistogram(
            df; equilibration_cycles=0, limits=(-1.0, -2.0)
        )
        @test_throws ArgumentError MultiHistogram(
            df; equilibration_cycles=0, limits=(-Inf, 0.0)
        )

        unsorted = df[end:-1:1, :]
        @test_throws ArgumentError MultiHistogram(unsorted; equilibration_cycles=0)

        inconsistent_temperature = copy(df.temperature)
        inconsistent_temperature[4] = 301.0
        @test_throws ArgumentError MultiHistogram(
            df.trajectory_id,
            inconsistent_temperature,
            df.cycle,
            df.hamiltonian,
            Dict();
            num_bins=4,
            equilibration_cycles=0,
        )

        missing_trajectory = copy(df.trajectory_id)
        missing_trajectory[missing_trajectory .== 3] .= 4
        @test_throws ArgumentError MultiHistogram(
            missing_trajectory,
            df.temperature,
            df.cycle,
            df.hamiltonian,
            Dict();
            num_bins=4,
            equilibration_cycles=0,
        )
    end

    @testset "degenerate histogram validation" begin
        cycle = [1, 2, 3, 4]
        trajectory = ones(Int, 4)
        temperature = fill(300.0, 4)

        @test_throws ArgumentError MultiHistogram(
            trajectory,
            temperature,
            cycle,
            fill(1.0, 4),
            Dict();
            num_bins=4,
            equilibration_cycles=0,
        )
        @test_throws ArgumentError MultiHistogram(
            trajectory,
            temperature,
            cycle,
            fill(NaN, 4),
            Dict();
            num_bins=4,
            equilibration_cycles=0,
        )
    end

    @testset "Arrow tables" begin
        for tbl in (neon(), chunked())
            mh = @test_logs (:warn, r"eq_cycles") MultiHistogram(tbl; num_bins=500)
            num_cycles = maximum(tbl.cycle)
            first_used = round(Int, num_cycles / 6) + 1
            @test num_trajectories(mh) == maximum(tbl.trajectory_id)
            @test num_bins(mh) == 500
            @test all(==(num_cycles - first_used + 1), mh.num_samples)
            @test issorted(mh.temperature)
        end
    end
end

@testset "multihistogram iteration" begin
    df = synthetic_df(
        [
            -10.0 -8.0 -6.0
            -9.0 -7.0 -5.0
            -8.0 -6.0 -4.0
            -7.0 -5.0 -3.0
            -6.0 -4.0 -2.0
        ],
        T_SMALL,
    )
    mh = MultiHistogram(df; num_bins=8, equilibration_cycles=0)

    @testset "log denominator update" begin
        log_denominator = zeros(num_bins(mh))
        free_energy = [0.0, -0.5, 0.25]

        result = MHA.update_log_denominator!(log_denominator, mh, free_energy)

        @test result === log_denominator
        # log Σ_j N_j exp(-β_j H_k + f_j), shifted by hand to avoid overflow
        expected = map(1:num_bins(mh)) do k
            terms = [
                log(mh.num_samples[j]) - mh.beta[j] * mh.bin_centre[k] + free_energy[j] for
                j in 1:num_trajectories(mh)
            ]
            m = maximum(terms)
            m + log(sum(exp.(terms .- m)))
        end
        @test log_denominator ≈ expected
    end

    @testset "get_log_weights!" begin
        free_energy = zeros(num_trajectories(mh))
        log_weights, f = MHA.get_log_weights!(mh, 1e-12, 5000, false, free_energy)

        @test length(log_weights) == num_bins(mh)
        @test length(f) == num_trajectories(mh)
        @test maximum(f) == 0
        # Empty bins have zero weight; occupied bins have finite weight.
        occupied = vec(sum(mh.weights; dims=2)) .> 0
        @test all(isfinite, log_weights[occupied])
        @test all(==(-Inf), log_weights[.!occupied])

        # The converged free energies are a fixed point of the iteration.
        _, f2 = MHA.get_log_weights!(mh, 1e-12, 5000, false, copy(f))
        @test f2 ≈ f atol = 1e-6

        # Free energies are consistent with the density of states:
        # exp(-f_j) ∝ Σ_k Ω_k exp(-β_j H_k).
        logZ = [
            MHA.logsumexp(log_weights .- mh.beta[j] .* mh.bin_centre) for
            j in 1:num_trajectories(mh)
        ]
        @test (logZ .- logZ[1]) ≈ -(f .- f[1]) atol = 1e-5

        @test_throws ArgumentError MHA.get_log_weights!(mh, 0, 100, false, zeros(3))
        @test_throws ArgumentError MHA.get_log_weights!(mh, -1, 100, false, zeros(3))
        @test_throws ArgumentError MHA.get_log_weights!(mh, 1e-12, 0, false, zeros(3))
        @test_throws ErrorException MHA.get_log_weights!(mh, 1e-300, 1, false, zeros(3))
    end
end

@testset "thermodynamic_properties" begin
    df = synthetic_df(
        [
            -10.0 -9.0 -8.0
            -9.0 -8.0 -7.0
            -8.0 -7.0 -6.0
            -7.0 -6.0 -5.0
            -6.0 -5.0 -4.0
            -5.0 -4.0 -3.0
            -4.0 -3.0 -2.0
            -3.0 -2.0 -1.0
        ],
        T_SMALL,
    )
    mh = MultiHistogram(df; num_bins=12, equilibration_cycles=0)

    @testset "returns expected columns and sizes" begin
        result = thermodynamic_properties(
            df; num_bins=12, equilibration_cycles=0, points=7, tol=1e-10, maxiter=5000
        )

        @test result isa DataFrame
        @test names(result) == [
            "temperature", "heat_capacity", "hamiltonian", "hamiltonian_squared", "entropy"
        ]
        @test nrow(result) == 7
        for column in eachcol(result)
            @test all(isfinite, column)
        end
        @test result.temperature[1] ≈ 300.0
        @test result.temperature[end] ≈ 500.0
        @test issorted(result.temperature)
        @test all(≥(0), result.heat_capacity)
        @test all(result.hamiltonian_squared .≥ result.hamiltonian .^ 2 .- 1e-12)
    end

    @testset "non-Int number of points" begin
        @test nrow(thermodynamic_properties(mh; points=Int16(5))) == 5
    end

    @testset "default number of points" begin
        @test nrow(thermodynamic_properties(df; num_bins=12, equilibration_cycles=0)) == 30
        @test nrow(thermodynamic_properties(mh)) == 30
    end

    @testset "MultiHistogram and DataFrame interfaces agree" begin
        from_df = thermodynamic_properties(
            df; num_bins=12, equilibration_cycles=0, points=7, tol=1e-10, maxiter=5000
        )
        from_mh = thermodynamic_properties(mh; points=7, tol=1e-10, maxiter=5000)

        for name in names(from_df)
            @test from_df[!, name] ≈ from_mh[!, name]
        end
    end

    @testset "initial free energy" begin
        cold = thermodynamic_properties(mh; points=7)
        init = zeros(num_trajectories(mh))
        warm, free_energy = thermodynamic_properties(mh, init; points=7)

        @test warm isa DataFrame
        @test free_energy isa Vector{Float64}
        @test length(free_energy) == num_trajectories(mh)
        @test init == zeros(num_trajectories(mh)) # not mutated
        @test free_energy !== init

        rewarmed, free_energy2 = thermodynamic_properties(mh, free_energy; points=7)
        for name in names(cold)
            # Results agree to within the convergence tolerance of the free energies.
            @test cold[!, name] ≈ warm[!, name] rtol = 1e-6
            @test cold[!, name] ≈ rewarmed[!, name] rtol = 1e-6
        end
        @test free_energy2 ≈ free_energy atol = 1e-6
    end

    @testset "validation" begin
        @test_throws ArgumentError thermodynamic_properties(mh; points=0)
        @test_throws ArgumentError thermodynamic_properties(mh; tol=0)
        @test_throws ArgumentError thermodynamic_properties(mh; maxiter=0)
        @test_throws ErrorException thermodynamic_properties(mh; tol=1e-300, maxiter=1)
    end

    @testset "thermodynamic consistency" begin
        props = thermodynamic_properties(neon(); num_bins=1000, points=400)
        T = props.temperature
        H = props.hamiltonian
        S = props.entropy
        C = props.heat_capacity
        inner = 2:(length(T) - 1)

        # C = dH/dT and C/T = dS/dT, by central differences.
        dHdT = (H[3:end] .- H[1:(end - 2)]) ./ (T[3:end] .- T[1:(end - 2)])
        dSdT = (S[3:end] .- S[1:(end - 2)]) ./ (T[3:end] .- T[1:(end - 2)])
        @test dHdT ≈ C[inner] rtol = 1e-2
        @test dSdT ≈ C[inner] ./ T[inner] rtol = 1e-2
    end

    # chunked.arrow is not equilibrated near its transition, so the reweighted and direct
    # averages legitimately disagree there; only the equilibrated neon data is used.
    @testset "agrees with direct averages at simulated temperatures" begin
        let tbl = neon()
            num_traj = maximum(tbl.trajectory_id)
            num_cycles = maximum(tbl.cycle)
            eq_cycles = num_cycles ÷ 6
            # A geometric grid with one point per trajectory reproduces the simulated
            # temperatures.
            props = thermodynamic_properties(
                tbl; num_bins=2000, points=num_traj, equilibration_cycles=eq_cycles
            )
            df = DataFrame(tbl)
            df = df[df.cycle .> eq_cycles, :]
            direct = combine(
                groupby(df, :trajectory_id; sort=true),
                :temperature => first => :temperature,
                :hamiltonian => mean => :hamiltonian,
                :hamiltonian => (h -> var(h; corrected=false)) => :variance,
            )

            @test props.temperature ≈ direct.temperature
            @test props.hamiltonian ≈ direct.hamiltonian rtol = 1e-3
            @test props.heat_capacity ≈ direct.variance ./ (kB .* direct.temperature .^ 2) rtol =
                0.2
        end
    end

    @testset "known results" begin
        properties = @test_logs (:warn, r"eq_cycles") thermodynamic_properties(neon())
        @test 12 ≤ properties.temperature[argmax(properties.heat_capacity)] ≤ 14

        properties = @test_logs (:warn, r"eq_cycles") thermodynamic_properties(chunked())
        @test 25 < properties.temperature[argmax(properties.heat_capacity)] < 40
    end
end

@testset "heat_capacity_peak" begin
    @testset "exact for a parabola" begin
        for x in (collect(0.0:0.5:5.0), [0.0, 0.3, 1.0, 1.8, 2.0, 2.9, 3.1, 4.5])
            y = @. 5 - 2 * (x - 2.3)^2
            T, C = heat_capacity_peak(x, y)
            @test T ≈ 2.3
            @test C ≈ 5.0
        end
    end

    @testset "maximum on a grid point" begin
        T, C = heat_capacity_peak([1.0, 2.0, 3.0], [0.0, 1.0, 0.0])
        @test T ≈ 2.0
        @test C ≈ 1.0
    end

    @testset "DataFrame method" begin
        x = collect(0.0:0.25:4.0)
        df = DataFrame(; temperature=x, heat_capacity=@. 1 - (x - 1.1)^2)
        @test all(heat_capacity_peak(df) .≈ (1.1, 1.0))

        df2 = DataFrame(; T=x, C=df.heat_capacity)
        @test all(heat_capacity_peak(df2; x_name=:T, y_name=:C) .≈ (1.1, 1.0))
    end

    @testset "maximum at the edges" begin
        x = collect(1.0:5.0)
        @test_logs (:warn, r"first index") @test all(
            ismissing, heat_capacity_peak(x, reverse(x))
        )
        @test_logs (:warn, r"last index") @test all(ismissing, heat_capacity_peak(x, x))
        @test_logs @test all(ismissing, heat_capacity_peak(x, x; warn=false))
    end

    @testset "validation" begin
        @test_throws ArgumentError heat_capacity_peak([1.0, 2.0], [1.0])
    end

    @testset "real data" begin
        props = thermodynamic_properties(neon(); equilibration_cycles=10_000, points=200)
        T, C = heat_capacity_peak(props)
        i = argmax(props.heat_capacity)
        @test props.temperature[i - 1] < T < props.temperature[i + 1]
        @test C ≥ props.heat_capacity[i]
    end
end

@testset "BlockedMultiHistogram" begin
    @testset "helpers" begin
        @test MHA.num_cycles_and_trajectories([1, 1, 2, 2], Int16[1, 2, 1, 2]) == (2, 2)
        @test MHA.num_cycles_and_trajectories([1, 1, 2, 2], Int16[1, 2, 1, 2]) isa
            Tuple{Int,Int}
        edges = MHA.bin_edges([-3.0, 1.0, -1.0], 4)
        @test edges ≈ range(-3.0, 1.0; length=5)
        @test_throws ArgumentError MHA.bin_edges([1.0, 1.0], 4)
    end

    @testset "construction" begin
        tbl = chunked()
        bmh = BlockedMultiHistogram(tbl; num_bins=300, num_blocks=50)
        num_cycles = maximum(tbl.cycle)
        num_traj = maximum(tbl.trajectory_id)

        @test bmh isa BlockedMultiHistogram
        @test length(bmh) == 50
        @test num_bins(bmh) == 300
        @test num_trajectories(bmh) == num_traj
        @test MHA.block_size(bmh) == num_cycles ÷ 50
        @test size(bmh.weights) == (300, num_traj, 50)
        @test length(bmh.edges) == 301
        @test length(bmh.bin_centre) == 300
        @test bmh.edges[1] == minimum(tbl.hamiltonian)
        @test bmh.edges[end] == maximum(tbl.hamiltonian)
        @test bmh.temperature == MHA.check_trajectory_temperature_consistency(
            tbl.trajectory_id, tbl.temperature
        )
        @test bmh.beta ≈ 1 ./ (kB .* bmh.temperature)

        # Every row is binned, and every block has one sample per cycle per trajectory.
        @test sum(bmh.weights) == length(tbl.cycle)
        @test dropdims(sum(bmh.weights; dims=1); dims=1) == bmh.num_samples
        @test all(==(MHA.block_size(bmh)), bmh.num_samples)
    end

    @testset "file name and table constructors agree" begin
        path = joinpath(@__DIR__, "testing_data/chunked.arrow")
        bmh1 = BlockedMultiHistogram(path; num_bins=100, num_blocks=20)
        bmh2 = BlockedMultiHistogram(chunked(); num_bins=100, num_blocks=20)
        bmh3 = BlockedMultiHistogram(DataFrame(chunked()); num_bins=100, num_blocks=20)
        for field in fieldnames(BlockedMultiHistogram)
            @test getfield(bmh1, field) == getfield(bmh2, field)
            @test getfield(bmh1, field) == getfield(bmh3, field)
        end
    end

    @testset "trailing cycles are dropped" begin
        tbl = chunked()
        num_cycles = maximum(tbl.cycle)
        num_traj = maximum(tbl.trajectory_id)
        num_blocks = 300 # does not divide num_cycles
        @test num_cycles % num_blocks != 0

        bmh = BlockedMultiHistogram(tbl; num_bins=100, num_blocks)
        @test length(bmh) == num_blocks
        @test sum(bmh.num_samples) == num_blocks * (num_cycles ÷ num_blocks) * num_traj
        @test all(==(MHA.block_size(bmh)), bmh.num_samples)
    end

    @testset "validation" begin
        df = synthetic_df(H_SMALL, T_SMALL)
        @test_throws ArgumentError BlockedMultiHistogram(df; num_bins=4, num_blocks=5)
        @test_throws ArgumentError BlockedMultiHistogram(
            synthetic_df(fill(1.0, 4, 3), T_SMALL); num_bins=4, num_blocks=2
        )
    end

    @testset "show" begin
        bmh = BlockedMultiHistogram(chunked(); num_bins=100, num_blocks=20)
        @test sprint(show, bmh) ==
            "BlockedMultiHistogram with 20 blocks of 5500 cycles each."
    end

    @testset "collapse" begin
        bmh = BlockedMultiHistogram(neon(); num_bins=200, num_blocks=50)
        target = MHA.collapse_target(bmh)
        @test target isa MultiHistogram
        @test iszero(target.weights)
        @test iszero(target.num_samples)
        @test target.edges == bmh.edges
        @test target.temperature == bmh.temperature

        full = MHA.collapse(bmh, 1:length(bmh))
        @test full.weights == dropdims(sum(bmh.weights; dims=3); dims=3)
        @test full.num_samples == vec(sum(bmh.num_samples; dims=2))
        @test num_bins(full) == num_bins(bmh)
        @test num_trajectories(full) == num_trajectories(bmh)

        single = MHA.collapse(bmh, 7)
        @test single.weights == bmh.weights[:, :, 7]

        # Repeated indices are counted with multiplicity, as needed for bootstrapping.
        @test MHA.collapse(bmh, [3, 3, 5]).weights ==
            2 .* bmh.weights[:, :, 3] .+ bmh.weights[:, :, 5]

        # collapse! overwrites rather than accumulates.
        mh = MHA.collapse_target(bmh)
        MHA.collapse!(mh, bmh, 1:10)
        result = MHA.collapse!(mh, bmh, 11:12)
        @test result === mh
        @test mh.weights == MHA.collapse(bmh, 11:12).weights
        @test mh.num_samples == MHA.collapse(bmh, 11:12).num_samples
    end

    @testset "agrees with MultiHistogram" begin
        tbl = neon()
        bmh = BlockedMultiHistogram(tbl; num_bins=200, num_blocks=50)
        from_blocks = MHA.collapse(bmh, 1:length(bmh))
        direct = MultiHistogram(
            tbl; num_bins=200, equilibration_cycles=0, limits=(bmh.edges[1], bmh.edges[end])
        )

        @test direct.edges ≈ from_blocks.edges
        @test direct.num_samples == from_blocks.num_samples
        # Binning by division and by search can differ for samples within rounding
        # distance of an edge.
        @test sum(abs, direct.weights .- from_blocks.weights) ≤ 1e-4 * sum(direct.weights)

        p1 = thermodynamic_properties(from_blocks; points=50)
        p2 = thermodynamic_properties(direct; points=50)
        @test p1.heat_capacity ≈ p2.heat_capacity rtol = 1e-3

        # Skipping blocks is equivalent to skipping the same equilibration cycles.
        skip = 10
        eq_cycles = skip * MHA.block_size(bmh)
        from_blocks = MHA.collapse(bmh, (skip + 1):length(bmh))
        direct = MultiHistogram(
            tbl;
            num_bins=200,
            equilibration_cycles=eq_cycles,
            limits=(bmh.edges[1], bmh.edges[end]),
        )
        @test direct.num_samples == from_blocks.num_samples
    end

    @testset "equilibration_analysis" begin
        bmh = BlockedMultiHistogram(chunked(); num_bins=300, num_blocks=20)
        window_size = 10
        result = equilibration_analysis(bmh; window_size, points=60)
        num_steps = length(bmh) - window_size + 1

        @test result isa DataFrame
        @test names(result) == [
            "block_start",
            "block_end",
            "cycle_start",
            "critical_temperature",
            "heat_capacity_maximum",
        ]
        @test nrow(result) == num_steps
        @test result.block_start == 1:num_steps
        @test result.block_end == window_size:length(bmh)
        @test result.cycle_start == 1 .+ (0:(num_steps - 1)) .* MHA.block_size(bmh)

        Tmin, Tmax = extrema(bmh.temperature)
        found = collect(skipmissing(result.critical_temperature))
        @test !isempty(found)
        @test all(T -> Tmin < T < Tmax, found)
        @test all(>(0), skipmissing(result.heat_capacity_maximum))

        # Each row matches a direct computation on the same window.
        for i in (1, num_steps)
            mh = MHA.collapse(bmh, i:(i + window_size - 1))
            expected = heat_capacity_peak(
                thermodynamic_properties(mh; points=60, show_progress=false); warn=false
            )
            @test isequal(result.critical_temperature[i], expected[1]) ||
                result.critical_temperature[i] ≈ expected[1]
            @test isequal(result.heat_capacity_maximum[i], expected[2]) ||
                result.heat_capacity_maximum[i] ≈ expected[2]
        end

        @test nrow(equilibration_analysis(bmh; window_size=length(bmh), points=60)) == 1
    end

    @testset "bootstrap_heat_capacity_peak" begin
        bmh = BlockedMultiHistogram(chunked(); num_bins=300, num_blocks=20)
        equilibration_blocks = 4
        num_samples = 16
        result = bootstrap_heat_capacity_peak(bmh; equilibration_blocks, num_samples)

        @test keys(result) == (:value, :error, :samples)
        @test result.samples isa DataFrame
        @test names(result.samples) == ["critical_temperature", "heat_capacity_maximum"]
        @test nrow(result.samples) == num_samples

        full = MHA.collapse(bmh, (equilibration_blocks + 1):length(bmh))
        @test result.value ≈ heat_capacity_peak(thermodynamic_properties(full))[1]

        @test isfinite(result.error)
        @test result.error > 0
        Tmin, Tmax = extrema(bmh.temperature)
        @test all(T -> Tmin < T < Tmax, skipmissing(result.samples.critical_temperature))
        # Resampled estimates scatter around the full-data value.
        @test abs(mean(skipmissing(result.samples.critical_temperature)) - result.value) <
            5 * result.error

        # Without equilibration blocks and with a single sample.
        result = bootstrap_heat_capacity_peak(bmh; equilibration_blocks=0, num_samples=1)
        @test nrow(result.samples) == 1
        @test_throws UndefKeywordError bootstrap_heat_capacity_peak(bmh)
    end
end
