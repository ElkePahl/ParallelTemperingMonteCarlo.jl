using Test
using Arrow
using DataFrames
using ParallelTemperingMonteCarlo
using ParallelTemperingMonteCarlo.MultiHistogramAnalysis: kB

@testset "MultiHistogram" begin
    trajectory_id = repeat(1:3; inner=4)
    temperature = repeat([300.0, 400.0, 500.0]; inner=4)
    cycle = repeat(1:4, 3)
    hamiltonian = [-10.0, -9.0, -8.0, -7.0, -8.0, -6.0, -5.0, -4.0, -5.0, -3.0, -2.0, -1.0]

    df = DataFrame(;
        trajectory_id=trajectory_id,
        temperature=temperature,
        cycle=cycle,
        hamiltonian=hamiltonian,
    )

    @testset "construction" begin
        mh = MultiHistogram(df; num_bins=4, skip_ratio=0)

        @test mh isa MultiHistogram
        @test num_trajectories(mh) == 3
        @test num_bins(mh) == 4
        @test mh.temperature == [300.0, 400.0, 500.0]
        @test mh.beta ≈ 1 ./ (kB .* mh.temperature)
        @test length(mh.edges) == 5
        @test length(mh.bin_centre) == 4
        @test size(mh.weights) == (4, 3)
        @test mh.num_samples == [4, 4, 4]
        @test sum(mh.weights) == 12
    end

    @testset "equilibration is discarded" begin
        mh = MultiHistogram(df; num_bins=4, skip_ratio=0.5)

        # max cycle = 4, so first_used = 2 and cycles 1 and 2 are discarded.
        @test sum(mh.weights) == 6
        @test mh.num_samples == [2, 2, 2]
    end

    @testset "DataFrame and vector constructors agree" begin
        mh1 = MultiHistogram(df; num_bins=4, skip_ratio=0)
        mh2 = MultiHistogram(
            df.trajectory_id,
            df.temperature,
            df.cycle,
            df.hamiltonian;
            num_bins=4,
            skip_ratio=0,
        )

        @test mh1.temperature == mh2.temperature
        @test mh1.beta == mh2.beta
        @test mh1.bin_centre == mh2.bin_centre
        @test mh1.edges == mh2.edges
        @test mh1.weights == mh2.weights
        @test mh1.num_samples == mh2.num_samples
    end

    @testset "input validation" begin
        @test_throws DimensionMismatch MultiHistogram([1, 2], [300.0], [1, 2], [-1.0, -2.0])

        @test_throws ArgumentError MultiHistogram(df; num_bins=2)

        @test_throws ArgumentError MultiHistogram(df; skip_ratio=-0.01)

        @test_throws ArgumentError MultiHistogram(df; skip_ratio=1)

        inconsistent_temperature = copy(df.temperature)
        inconsistent_temperature[2] = 301.0

        @test_throws ArgumentError MultiHistogram(
            df.trajectory_id,
            inconsistent_temperature,
            df.cycle,
            df.hamiltonian;
            num_bins=4,
            skip_ratio=0,
        )

        missing_trajectory = copy(df.trajectory_id)
        missing_trajectory[missing_trajectory .== 3] .= 4

        @test_throws ArgumentError MultiHistogram(
            missing_trajectory,
            df.temperature,
            df.cycle,
            df.hamiltonian;
            num_bins=4,
            skip_ratio=0,
        )
    end

    @testset "degenerate histogram validation" begin
        cycle2 = [1, 2, 3, 4]
        trajectory2 = ones(Int, 4)
        temperature2 = fill(300.0, 4)

        @test_throws ArgumentError MultiHistogram(
            trajectory2, temperature2, cycle2, fill(1.0, 4); num_bins=4, skip_ratio=0
        )

        @test_throws ArgumentError MultiHistogram(
            trajectory2,
            temperature2,
            cycle2,
            [-1.0, -1.0, -1.0, -1.0];
            num_bins=4,
            skip_ratio=0,
        )
    end
end

@testset "multihistogram iteration" begin
    trajectory_id = repeat(1:3; inner=5)
    temperature = repeat([300.0, 400.0, 500.0]; inner=5)
    cycle = repeat(1:5, 3)
    hamiltonian = [
        -10.0,
        -9.0,
        -8.0,
        -7.0,
        -6.0,
        -8.0,
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
    ]

    mh = MultiHistogram(
        trajectory_id, temperature, cycle, hamiltonian; num_bins=8, skip_ratio=0
    )

    @testset "log denominator update" begin
        log_denominator = zeros(num_bins(mh))
        free_energy = zeros(num_trajectories(mh))

        result = MultiHistogramAnalysis.update_log_denominator!(
            log_denominator, mh, free_energy
        )

        @test result === log_denominator
        @test all(isfinite, log_denominator)
    end

    @testset "get_log_weights" begin
        log_weights = MultiHistogramAnalysis.get_log_weights(mh, 1e-12, 5000)

        @test length(log_weights) == num_bins(mh)
        @test any(isfinite, log_weights)
        @test all(isfinite, log_weights[isfinite.(log_weights)])

        @test_throws ArgumentError MultiHistogramAnalysis.get_log_weights(mh, 0, 100)
        @test_throws ArgumentError MultiHistogramAnalysis.get_log_weights(mh, -1, 100)
        @test_throws ArgumentError MultiHistogramAnalysis.get_log_weights(mh, 1e-12, 0)
    end
end

@testset "thermodynamic_properties" begin
    trajectory_id = repeat(1:3; inner=8)
    temperature = repeat([300.0, 400.0, 500.0]; inner=8)
    cycle = repeat(1:8, 3)
    hamiltonian = [
        -10.0,
        -9.0,
        -8.0,
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -9.0,
        -8.0,
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -8.0,
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
    ]

    df = DataFrame(;
        trajectory_id=trajectory_id,
        temperature=temperature,
        cycle=cycle,
        hamiltonian=hamiltonian,
    )

    @testset "returns expected columns and sizes" begin
        result = thermodynamic_properties(
            df; num_bins=12, skip_ratio=0, points=7, tol=1e-10, maxiter=5000
        )

        @test result isa DataFrame
        @test names(result) == [
            "temperature", "heat_capacity", "hamiltonian", "hamiltonian_squared", "entropy"
        ]
        @test nrow(result) == 7
        @test all(isfinite, result.temperature)
        @test all(isfinite, result.heat_capacity)
        @test all(isfinite, result.hamiltonian)
        @test all(isfinite, result.hamiltonian_squared)
        @test all(isfinite, result.entropy)
        @test result.temperature[1] ≈ 300.0
        @test result.temperature[end] ≈ 500.0
    end

    @testset "MultiHistogram and DataFrame interfaces agree" begin
        mh = MultiHistogram(df; num_bins=12, skip_ratio=0)

        from_df = thermodynamic_properties(
            df; num_bins=12, skip_ratio=0, points=7, tol=1e-10, maxiter=5000
        )

        from_mh = thermodynamic_properties(mh; points=7, tol=1e-10, maxiter=5000)

        @test from_df.temperature ≈ from_mh.temperature
        @test from_df.heat_capacity ≈ from_mh.heat_capacity
        @test from_df.hamiltonian ≈ from_mh.hamiltonian
        @test from_df.hamiltonian_squared ≈ from_mh.hamiltonian_squared
        @test from_df.entropy ≈ from_mh.entropy
    end

    @testset "validation" begin
        mh = MultiHistogram(df; num_bins=8, skip_ratio=0)

        @test_throws ArgumentError thermodynamic_properties(mh; points=0)
    end

    @testset "known results" begin
        df = Arrow.Table("testing_data/neon-55-100K.arrow")

        properties = thermodynamic_properties(df)
        max_heat_capacity = properties.temperature[argmax(properties.heat_capacity)]

        @test 12 ≤ max_heat_capacity ≤ 14
    end
end
