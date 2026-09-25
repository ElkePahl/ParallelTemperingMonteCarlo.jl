# ======================================================================================== #
# This test ensures that after performing a step, distance matrices, tangent matrices, and
# energies are updated consistently, i.e. that recalculating these properties from the
# current configuration yields the same result as the updated values.
using Test
using ParallelTemperingMonteCarlo
using StaticArrays
using LinearAlgebra

"""
    mc_move_deterministic!(accept, mc_state, move_strat, pot, ensemble)

Like `mc_move!`, but accepts the step if `accept ≡ true`. Returns move name for easier
debugging.
"""
function mc_move_deterministic!(accept, mc_state, move_strat, pot, ensemble)
    mc_state.ensemble_variables.index = index = rand(eachindex(move_strat.movestrat))
    move = move_strat.movestrat[index]

    generate_move!(mc_state, move)
    get_energy!(mc_state, move)

    if accept
        swap_config!(mc_state, move)
    end
    return move
end

"""
    generate_config(ensemble, boundary_condition)

Generate a (uniform) random configuration that fits into boundary condition.
"""
function generate_config(ensemble, boundary_condition)
    positions = SVector{3,Float64}[]

    while length(positions) < ensemble.n_atoms
        pos = 10 * @SVector(rand(3)) .- 5
        pos = check_boundary(boundary_condition, pos)
        if !isnothing(pos)
            push!(positions, pos)
        end
    end
    return Config(positions, boundary_condition)
end

# Potentials
pot_ne = ELJPotentialEven{6}([
    -10.5097942564988,
    989.725135614556,
    -101383.865938807,
    3918846.12841668,
    -56234083.4334278,
    288738837.441765,
])
potB_ne = ELJPotentialB{6}(
    [0.0005742, -0.4032, -0.2101, -0.0595, 0.0606, 0.1608],
    [-0.01336, -0.02005, -0.1051, -0.1268, -0.1405, -0.1751],
    [-0.1132, -1.5012, 35.6955, -268.7494, 729.7605, -583.4203],
)
pot_lut = LookupTablePotential(
    joinpath(@__DIR__, "../scripts/lookup-tables/LookupTable_Neon_B0.3_MP2.txt")
)

pot_embedded = EmbeddedAtomPotential(8.482, 4.692, 0.0013597241, 4.724325, 27.561)

"""
    generate_test_cases(n_atoms)

Generated test cases, combinations of boundary conditions, ensembles and potentials.
"""
function generate_test_cases(n_atoms)
    test_cases = []
    for ensemble_type in (NPT, NVT)
        for bc in (
            SphericalBC(; radius=5.0),
            CubicBC(5.0),
            RectangularBC(5.0, 5.0),
            RhombicBC(5.0, 5.0),
        )
            if bc isa SphericalBC && ensemble_type === NPT
                continue
            elseif bc isa CubicBC && ensemble_type === NPT
                ensemble = NPT(n_atoms, 0.01, false)
            elseif ensemble_type == NPT
                ensemble = NPT(n_atoms, 0.01, true)
            else
                ensemble = ensemble_type(n_atoms)
            end
            for potential in (
                # TODO: RuNNer
                pot_ne,
                potB_ne,
                pot_lut,
                pot_embedded,
            )
                if !(potential isa AbstractDimerPotential) && ensemble_type === NPT
                    continue
                end
                id = "$ensemble_type, $(nameof(typeof(potential))), $bc"
                push!(test_cases, id => (bc, ensemble, potential))
            end
        end
    end
    return test_cases
end

# These tests check that performing (or rejecting) a MC step correctly updates the energy,
# the distance matrix and (when applicable) the tangent matrix. The test is performed on
# a random starting configuration in a way where every other step is rejected.
@testset "Move consistency" begin
    ti = 9.0
    tf = 16.0
    n_traj = 16
    temp = TempGrid{n_traj}(ti, tf)

    for (id, (boundary_condition, ensemble, potential)) in generate_test_cases(10)
        config = generate_config(ensemble, boundary_condition)

        @testset "$id" begin
            move_strategy = MoveStrategy(ensemble)
            mc_state = MCState(temp.t_grid[5], config, ensemble, potential)

            true_dist2 = get_distance2_mat(config)
            @test mc_state.dist2_mat == true_dist2

            if potential isa AbstractDimerPotentialB
                true_tan = get_tantheta_mat(config)
                @test mc_state.potential_variables.tan_mat == true_tan
            end

            for i in 1:10_000
                accept = iseven(i)
                move = mc_move_deterministic!(
                    accept, mc_state, move_strategy, potential, ensemble
                )

                updated_dist2 = mc_state.dist2_mat
                true_dist2 = get_distance2_mat(mc_state.config)
                @test updated_dist2 ≈ true_dist2
                @test issymmetric(updated_dist2)

                if potential isa AbstractDimerPotentialB
                    updated_tan = mc_state.potential_variables.tan_mat
                    true_tan = get_tantheta_mat(mc_state.config)

                    @test updated_tan ≈ true_tan
                    @test issymmetric(updated_tan)
                end

                true_energy = initialise_energy(
                    mc_state.config,
                    true_dist2,
                    mc_state.potential_variables,
                    mc_state.ensemble_variables,
                    potential,
                )[1]

                updated_energy = mc_state.en_tot

                @test updated_energy ≈ true_energy

                # With these high energy configurations, the energy has a tendency to drift
                # a bit. Resetting it to the correct energy fixes the issue.
                # TODO: this should probably also be done by the MC algorithm
                mc_state.en_tot = true_energy
            end
        end
    end
end
