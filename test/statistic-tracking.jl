using ParallelTemperingMonteCarlo, Test, Arrow, Random, DataFrames

function run_full_computation(; flush_interval)
    Random.seed!(1234)

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
    pos_ne32 = [
        [-4.3837, -4.3837, -4.3837],
        [-2.1918, -2.1918, -4.3837],
        [-2.1918, -4.3837, -2.1918],
        [-4.3837, -2.1918, -2.1918],
        [-4.3837, -4.3837, 0.0000],
        [-2.1918, -2.1918, 0.0000],
        [-2.1918, -4.3837, 2.1918],
        [-4.3837, -2.1918, 2.1918],
        [-4.3837, 0.0000, -4.3837],
        [-2.1918, 2.1918, -4.3837],
        [-2.1918, 0.0000, -2.1918],
        [-4.3837, 2.1918, -2.1918],
        [-4.3837, 0.0000, 0.0000],
        [-2.1918, 2.1918, 0.0000],
        [-2.1918, 0.0000, 2.1918],
        [-4.3837, 2.1918, 2.1918],
        [0.0000, -4.3837, -4.3837],
        [2.1918, -2.1918, -4.3837],
        [2.1918, -4.3837, -2.1918],
        [0.0000, -2.1918, -2.1918],
        [0.0000, -4.3837, 0.0000],
        [2.1918, -2.1918, 0.0000],
        [2.1918, -4.3837, 2.1918],
        [0.0000, -2.1918, 2.1918],
        [0.0000, 0.0000, -4.3837],
        [2.1918, 2.1918, -4.3837],
        [2.1918, 0.0000, -2.1918],
        [0.0000, 2.1918, -2.1918],
        [0.0000, 0.0000, 0.0000],
        [2.1918, 2.1918, 0.0000],
        [2.1918, 0.0000, 2.1918],
        [0.0000, 2.1918, 2.1918],
    ]

    positions = pos_ne32 * AtoBohr
    box_length = 8.7674 * AtoBohr
    boundary_condition = CubicBC(box_length)

    start_config = Config(positions, boundary_condition)
    ptmc_run!(
        mc_params, temp, start_config, pot, ensemble;
        stats_filename="test.arrow", flush_interval,
    )
end

@testset "Statistic tracking" begin
    _, _, stats1 = run_full_computation(; flush_interval=100)
    _, _, stats2 = run_full_computation(; flush_interval=10000)

    @test size(stats1) == size(stats2) == (26400, 9)

    @test stats1 == DataFrame(Arrow.Table("test.arrow"))
    @test stats2 == DataFrame(Arrow.Table("test-1.arrow"))

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
