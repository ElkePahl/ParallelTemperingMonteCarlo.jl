using ParallelTemperingMonteCarlo, Test, Arrow, Random, DataFrames

function run_full_computation(; flush_interval, seed)
    Random.seed!(seed)

    n_atoms = 32
    pressure = 101325
    AtoBohr = 1.8897261259077824

    n_traj = 24
    temp = TempGrid{n_traj}(10, 25)

    max_displ_atom = [0.1 * √(0.05 * temp.t_grid[i]) for i in 1:n_traj]
    mc_params = MCParams(1000, n_traj, n_atoms; mc_sample=1, n_adjust=100)

    c = [
        -10.5097942564988,
        989.725135614556,
        -101383.865938807,
        3918846.12841668,
        -56234083.4334278,
        288738837.441765,
    ]
    pot = ELJPotentialEven{6}(c)

    separated_volume = false
    ensemble = NPT(n_atoms, pressure * 2.2937122783969076e-13 / AtoBohr^3, separated_volume)
    move_strat = MoveStrategy(ensemble)

    # Face centred cubic structure.
    pos_ne32 = face_centred_cubic(1; r_min=3.01)

    positions = pos_ne32 * AtoBohr
    box_length = 8.7674 * AtoBohr
    boundary_condition = CubicBC(box_length)

    start_config = Config(positions, boundary_condition)
    return ptmc_run!(
        mc_params,
        temp,
        start_config,
        pot,
        ensemble;
        stats_filename="test.arrow",
        flush_interval,
    )
end

@testset "Statistic tracking" begin
    _, _, stats1 = run_full_computation(; flush_interval=100, seed=123)
    _, _, stats2 = run_full_computation(; flush_interval=10000, seed=123)

    # 1000 production cycles + 200 equilibration cycles, for 24 trajectories.
    # All requested equilibration cycles now perform MC steps and are recorded.
    @test size(stats1) == size(stats2) == (28800, 10)

    @test stats1 == DataFrame(Arrow.Table("test.arrow"))
    @test stats2 == DataFrame(Arrow.Table("test-1.arrow"))
    # Only these can be compared since even with a fixed seed, multithreaded computations
    # are non-deterministic.
    @test stats1.cycle == stats2.cycle
    @test stats1.temperature == stats2.temperature

    # check that chunking was only performed for stats1
    @test stats1.cycle isa Arrow.SentinelArrays.ChainedVector
    @test stats2.cycle isa Arrow.Primitive

    # check that all chunks are 100 × number of trajectories long
    @test all(x -> length(x) == 2400, stats1.cycle.arrays)

    @test stats1.hamiltonian ≈ stats1.total_energy .+ stats1.volume .* 3.4439667494478555e-9

    # cleanup
    i = 1
    if isfile("test.arrow")
        rm("test.arrow")
        while isfile("test-$i.arrow")
            rm("test-$i.arrow")
            i += 1
        end
    end
    @test i == 2
end
