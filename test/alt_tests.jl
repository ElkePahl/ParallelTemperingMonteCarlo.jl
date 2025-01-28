using Test
using SafeTestsets
using ParallelTemperingMonteCarlo
using StaticArrays, LinearAlgebra, DelimitedFiles
using Random

include(joinpath(@__DIR__, "setup_params.jl"))

test_any(x) = @test x isa Any
get_number_set() = [1, 0.5, -3]
@testset "Ensemble" begin
    @testset "get_r_cut" begin
        test_any(get_r_cut(get_cubic_bc()))
        
        test_any(get_r_cut(get_rhombic_bc()))

        @test_throws MethodError get_r_cut(get_spherical_bc())
    end
    
    
    @testset "NVT" begin
        for trial in get_number_set()
            test_any(NVT(rand(5:20)))
        end
    end
    
    @testset "MoveStrategy" begin
        test_any(MoveStrategy(get_nvt()))
        test_any(MoveStrategy(get_npt()))
    end
    
    @testset "NVTVariables" begin
        set = [get_spherical_bc(), get_cubic_bc(), get_rhombic_bc()]
        for trial in set
            test_any(set_ensemble_variables(get_config(trial), get_nvt()))
        end
    end
    
    @testset "NPT" begin
        for trial in get_number_set()
            test_any(NPT(rand(5:20), trial))
        end
    end
    
    @testset "NPTVariables" begin
        set = [get_cubic_bc(), get_rhombic_bc()]
        for trial in set
            test_any(set_ensemble_variables(get_config(trial), get_npt()))
        end
    end
end

@testset "Configurations" begin
    @testset "Config constructor" begin
        test_any(Config(get_posvec(), get_spherical_bc()))

        test_any(Config(get_posvec(), get_cubic_bc()))

        test_any(Config(get_posvec(), get_rhombic_bc()))
    end

    @testset "distance2" begin
        test_any(distance2(get_pos(), get_pos()))

        test_any(distance2(get_pos(), get_pos(), get_spherical_bc()))

        test_any(distance2(get_pos(), get_pos(), get_cubic_bc()))

        test_any(distance2(get_pos(), get_pos(), get_rhombic_bc()))
    end

    @testset "get_distance2_mat" begin
        test_any(get_distance2_mat(get_config(get_spherical_bc())))

        test_any(get_distance2_mat(get_config(get_cubic_bc())))

        test_any(get_distance2_mat(get_config(get_rhombic_bc())))
    end

    @testset "get_volume" begin
        test_any(get_volume(get_cubic_bc()))

        test_any(get_volume(get_rhombic_bc()))

        @test_throws MethodError get_volume(get_spherical_bc())
    end

    @testset "get_tantheta_mat" begin
        bc = get_spherical_bc()
        test_any(get_tantheta_mat(get_config(bc), bc))

        bc = get_cubic_bc()
        test_any(get_tantheta_mat(get_config(bc), bc))

        bc = get_rhombic_bc()
        test_any(get_tantheta_mat(get_config(bc), bc))
    end

    @testset "get_tan" begin
        test_any(get_tan(get_pos(), get_pos()))

        test_any(get_tan(get_pos(), get_pos(), get_spherical_bc()))

        test_any(get_tan(get_pos(), get_pos(), get_cubic_bc()))

        test_any(get_tan(get_pos(), get_pos(), get_rhombic_bc()))
    end

    @testset "get_centre" begin
        posvec = get_posvec()
        test_any(get_centre(posvec, length(posvec)))
    end

    @testset "recentre!" begin
        test_any(recentre!(get_config(get_spherical_bc())))
    end
end

@testset "BoundaryConditions" begin
    @testset "SphericalBC" begin
        for trial in get_number_set()
            test_any(SphericalBC(trial))
        end
    end

    @testset "CubicBC" begin
        for trial in get_number_set()
            test_any(CubicBC(trial))
        end
    end

    @testset "RhombicBC" begin
        for trial in get_number_set()
            test_any(RhombicBC(trial, trial))
        end
    end

    @testset "check_boundary" begin
        bc = SphericalBC(radius=1.0)
        @test check_boundary(bc,SVector(0,0.5,1.))
        @test !check_boundary(bc,SVector(0,0.5,0.5))
    end
end

@testset "InputParams" begin
    @testset "MCParams" begin
        test_any(MCParams(1, 1, 1))
    end
    @testset "TempGrid" begin
        for trial in get_number_set()
            test_any(TempGrid(trial, trial + 1, rand(1:10)))
        end
    end 
    #Output functions are just constructors, with very little logic to test
end

@testset "EnergyEvaluation" begin
    @testset "dimer_energy_atom" begin
        test_any(dimer_energy_atom(get_index(),get_n_vec(),get_eljpot_even()))
        test_any(dimer_energy_atom(get_index(),get_n_vec(),0.5,get_eljpot_even()))
        test_any(dimer_energy_atom(get_index(),get_n_vec(),get_n_vec(),get_eljpot_b()))
        test_any(dimer_energy_atom(get_index(),get_n_vec(),get_n_vec(),0.5,get_eljpot_b()))
    end

    @testset "dimer_energy_config" begin
        bc = get_spherical_bc()
        config = get_config(bc)
        test_any(dimer_energy_config(get_distance2_mat(config), n_atom, get_dimer_vars(config), get_eljpot_even()))

        bc = get_cubic_bc()
        config = get_config(bc)
        test_any(dimer_energy_config(get_distance2_mat(config), n_atom, get_dimer_vars(config), 3.5, bc, get_eljpot_even()))

        bc = get_rhombic_bc()
        config = get_config(bc)
        test_any(dimer_energy_config(get_distance2_mat(config), n_atom, get_dimer_vars(config), 3.5, bc, get_eljpot_even()))

        bc = get_spherical_bc()
        config = get_config(bc)
        test_any(dimer_energy_config(get_distance2_mat(config), n_atom, get_eljb_vars(config), get_eljpot_b()))

        bc = get_cubic_bc()
        config = get_config(bc)
        test_any(dimer_energy_config(get_distance2_mat(config),n_atom,get_eljb_vars(config),3.5,bc,get_eljpot_b()))
    end

    @testset "dimer_energy_update" begin
        test_any(dimer_energy_update!(get_index(), get_n_by_n(), get_n_vec(), 0.0, get_eljpot_even()))
        test_any(dimer_energy_update!(get_index(), get_n_by_n(), get_n_vec(), 0.0, 3.5, get_eljpot_even()))
        test_any(dimer_energy_update!(get_index(), get_n_by_n(), get_n_by_n(), get_n_vec(), get_n_vec(), 0.0, get_eljpot_b()))
        test_any(dimer_energy_update!(get_index(), get_n_by_n(), get_n_by_n(), get_n_vec(), get_n_vec(), 0.0, 3.5, get_eljpot_b()))
    end
    @testset "set_variables" begin
        config = get_config(get_spherical_bc())
        test_any(set_variables(config, get_distance2_mat(config), get_eljpot_b()))
        test_any(set_variables(config, get_distance2_mat(config), get_eljpot_even()))
        test_any(set_variables(config, get_distance2_mat(config), get_eam()))
        test_any(set_variables(config, get_distance2_mat(config), get_RuNNerPotential()))
    end
    @testset "energy_update!" begin
        config = get_config(get_spherical_bc())
        test_any(energy_update!(get_pos(), get_index(), config, get_dimer_vars(config), get_n_by_n(), get_n_vec(), 0.0, get_eljpot_even()))
        test_any(energy_update!(get_pos(), get_index(), config, get_dimer_vars(config), get_n_by_n(), get_n_vec(), 0.0, 3.5, get_eljpot_even()))
        test_any(energy_update!(get_pos(), get_index(), config, get_eljb_vars(config), get_n_by_n(), get_n_vec(), 0.0, get_eljpot_b()))
        test_any(energy_update!(get_pos(), get_index(), config, get_eljb_vars(config), get_n_by_n(), get_n_vec(), 0.0, 3.5, get_eljpot_b()))
        #test_any(energy_update!(get_pos(), get_index(), config, get_eam_vars(config), get_n_by_n(), get_n_vec(), 0.0, get_eam()))
        #test_any(energy_update!(get_pos(), get_index(), config, get_RuNNer_vars(config), get_n_by_n(), get_n_vec(), 0.0, get_RuNNerPotential()))
    end
    @testset "dimer_energy" begin
        test_any(dimer_energy(get_eljpot_even(), 3.5))
        test_any(dimer_energy(get_eljpot_b(), 3.5, rand()))
    end
    @testset "lrc" begin
        test_any(lrc(n_atom, 3.5, get_eljpot_even()))
        test_any(lrc(n_atom, 3.5, get_eljpot_b()))
    end
    @testset "invrexp" begin
        test_any(invrexp(4.0, 5.0, 6.0))
        test_any(invrexp(rand(), rand(), rand()))
    end
    @testset "calc_components" begin
        test_any(calc_components(rand(2), get_n_vec(), rand(), rand()))
        test_any(calc_components(rand(n_atom, 2), get_index(), get_n_vec(), get_n_vec(), 3.0, 4.0))
        test_any(calc_components(rand(n_atom, 2), rand(n_atom, 2), get_index(), get_n_vec(), get_n_vec(), 3.0, 4.0))
    end
    @testset "calc_energies_from_components" begin
        test_any(calc_energies_from_components(rand(n_atom, 2), rand(), rand()))
    end
    @testset "initialise_energy" begin
        config = get_config(get_spherical_bc())
        test_any(initialise_energy(config, get_distance2_mat(config), get_dimer_vars(config), set_ensemble_variables(config, get_nvt()), get_eljpot_even()))
        config = get_config(get_cubic_bc())
        test_any(initialise_energy(config, get_distance2_mat(config), get_dimer_vars(config), set_ensemble_variables(config, get_npt()), get_eljpot_even()))
        config = get_config(get_spherical_bc())
        test_any(initialise_energy(config, get_distance2_mat(config), get_eljb_vars(config), set_ensemble_variables(config, get_nvt()), get_eljpot_b()))
        config = get_config(get_cubic_bc())
        test_any(initialise_energy(config, get_distance2_mat(config), get_eljb_vars(config), set_ensemble_variables(config, get_npt()), get_eljpot_b()))
        config = get_config(get_spherical_bc())
        test_any(initialise_energy(config, get_distance2_mat(config), get_eam_vars(config), set_ensemble_variables(config, get_nvt()), get_eam()))
        #test_any(initialise_energy(config, get_distance2_mat(config), get_RuNNer_vars(config), set_ensemble_variables(config, get_nvt()), get_RuNNerPotential()))
    end
end

@testset "Exchange" begin
    @testset "metropolis_condition" begin
        test_any(metropolis_condition(rand(), rand()))
        test_any(metropolis_condition(get_npt(), rand(), rand(), rand(), rand()))
        ensemble = get_nvt()
        test_any(metropolis_condition("atommove", get_mcstate(ensemble = ensemble), ensemble))
        ensemble = get_npt()
        test_any(metropolis_condition("volumemove", get_mcstate(ensemble = ensemble), ensemble))
    end
    @testset "exc_acceptance" begin
        test_any(exc_acceptance(rand(), rand(), rand(), rand()))
    end
    @testset "exc_trajectories" begin
        mcstatevec = get_mcstatevec()
        test_any(exc_trajectories!(mcstatevec[1], mcstatevec[2]))
    end
    @testset "parallel_tempering_exchange!" begin
        mc_params = get_mc_params()
        ensemble = get_nvt()
        mcstatevec = get_mcstatevec(mc_params=mc_params, ensemble=ensemble)
        test_any(parallel_tempering_exchange!(mcstatevec, mc_params, ensemble))
        ensemble = get_npt()
        mcstatevec = get_mcstatevec(mc_params=mc_params, ensemble=ensemble)
        test_any(parallel_tempering_exchange!(mcstatevec, mc_params, ensemble))
    end
    @testset "update_max_stepsize!" begin
        ensemble = get_nvt()
        mcstate = get_mcstate(ensemble=ensemble)
        test_any(update_max_stepsize!(mcstate, rand(10:20), ensemble, rand()/2, rand()/2+0.5))
        ensemble = get_npt()
        mcstate = get_mcstate(ensemble=ensemble)
        test_any(update_max_stepsize!(mcstate, rand(10:20), ensemble, rand()/2, rand()/2+0.5))
    end
end

@testset "Initialization" begin
    @testset "initialisation" begin
        mc_params = get_mc_params()
        tempgrid = get_tempgrid(; n_traj = mc_params.n_traj)
        test_any(initialisation(mc_params, tempgrid, get_config(get_spherical_bc()), get_eljpot_even(), get_nvt()))
        cd(joinpath(@__DIR__, "testing_data/"))
        test_any(initialisation(true, 0.3))
        test_any(initialisation(false, 0.3))
        cd(joinpath(@__DIR__, "..", ".."))
    end
end

@testset "MCMoves" begin
    test_any(atom_displacement(get_pos(), 0.5, get_spherical_bc()))
    test_any(atom_displacement(get_pos(), 0.5, get_cubic_bc()))
    test_any(atom_displacement(get_pos(), 0.5, get_rhombic_bc()))
    test_any(atom_displacement(get_mcstate(ensemble = get_nvt())))
    test_any(atom_displacement(get_mcstate(ensemble = get_npt())))

    bc = get_cubic_bc()
    test_any(volume_change(get_config(bc), bc, 0.5, 10))
    bc = get_rhombic_bc()
    test_any(volume_change(get_config(bc), bc, 0.5, 10))
    test_any(volume_change(get_mcstate(ensemble = get_npt())))

    test_any(generate_move!(get_mcstate(ensemble = get_nvt()), "atommove"))
    test_any(generate_move!(get_mcstate(ensemble = get_npt()), "volumemove"))
    @test_throws MethodError generate_move!(get_mcstate(ensemble = get_nvt()), "volumemove")
end

@testset "MCRun" begin
    @testset "get_energy!" begin
        pot = get_eljpot_even()
        test_any(get_energy!(get_mcstate(ensemble = get_nvt(), pot = pot), pot, "atommove"))
        test_any(get_energy!(get_mcstate(ensemble = get_npt(), pot = pot), pot, "volumemove"))
    end
    @testset "acc_test!" begin
        ensemble = get_nvt()
        test_any(acc_test!(get_mcstate(ensemble = ensemble), ensemble, "atommove"))
        ensemble = get_npt()
        test_any(acc_test!(get_mcstate(ensemble = ensemble), ensemble, "volumemove"))
    end

    @testset "swap_config!" begin
        movetype = "atommove"
        mcstate = get_mcstate(ensemble = get_nvt())
        test_any(swap_config!(mcstate, movetype))
        movetype = "volumemove"
        mcstate = get_mcstate(ensemble = get_npt())
        test_any(swap_config!(mcstate, movetype))
    end

    @testset "swap_atom_config!" begin
        test_any(swap_atom_config!(get_mcstate(), get_index(), get_pos()))
    end

    @testset "swap_config_v!" begin
        bc = get_cubic_bc()
        config = get_config(bc)
        dist2mat = get_distance2_mat(config)
        test_any(swap_config_v!(get_mcstate(ensemble = get_npt(), config = config), bc, config, dist2mat, get_n_vec(), 0.0))
        bc = get_rhombic_bc()
        config = get_config(bc)
        dist2mat = get_distance2_mat(config)
        test_any(swap_config_v!(get_mcstate(ensemble = get_npt(), config = config), bc, config, dist2mat, get_n_vec(), 0.0))
    end

    @testset "swap_vars!" begin
        test_any(swap_vars!(get_index(), get_dimer_vars(get_config(get_spherical_bc()))))
        test_any(swap_vars!(get_index(), get_eljb_vars(get_config(get_spherical_bc()))))
        test_any(swap_vars!(get_index(), get_eam_vars(get_config(get_spherical_bc()))))
        test_any(swap_vars!(get_index(), get_RuNNer_vars(get_config(get_spherical_bc()))))
    end

    @testset "mc_move!" begin
        ensemble = get_nvt()
        pot = get_eljpot_even()
        test_any(mc_move!(get_mcstate(ensemble = ensemble, pot = pot), MoveStrategy(ensemble), pot, ensemble))
    end

    @testset "mc_step!" begin
        ensemble = get_nvt()
        pot = get_eljpot_even()
        test_any(mc_step!(get_mcstatevec(ensemble = ensemble, pot = pot), MoveStrategy(ensemble), pot, ensemble, 10))
    end

    @testset "mc_cycle!" begin
        ensemble = get_nvt()
        pot = get_eljpot_even()
        mc_params = get_mc_params()
        test_any(mc_cycle!(get_mcstatevec(ensemble = ensemble, pot = pot, mc_params = mc_params), MoveStrategy(ensemble), mc_params, pot, ensemble, 10, get_index()))

        init = initialise(ensemble = ensemble, pot = pot, mc_params = mc_params)
        test_any(mc_cycle!(init[1], MoveStrategy(ensemble), mc_params, pot, ensemble, 10, init[3], get_index(), true))
    end

    @testset "check_e_bounds" begin
        test_any(check_e_bounds(rand(), [rand()/2, rand()/2+0.5]))
    end

    @testset "reset_counters" begin
        test_any(reset_counters(get_mcstate()))
    end

    @testset "equilibration_cycle!" begin
        mc_params = get_mc_params()
        ensemble = get_nvt()
        pot = get_eljpot_even()
        init = initialise(ensemble = ensemble, pot = pot, mc_params = mc_params)
        test_any(equilibration_cycle!(init[1], MoveStrategy(ensemble), mc_params, pot, ensemble, 10, init[3]))
    end

    @testset "equilibration" begin
        mc_params = get_mc_params()
        ensemble = get_nvt()
        pot = get_eljpot_even()
        init = initialise(ensemble = ensemble, pot = pot, mc_params = mc_params)
        test_any(equilibration(init[1], MoveStrategy(ensemble), mc_params, pot, ensemble, 10, init[3], false))
    end

    @testset "ptmc_run!" begin
        mc_params = get_mc_params()
        tempgrid = get_tempgrid(; n_traj = mc_params.n_traj)
        config = get_config(get_spherical_bc())
        test_any(ptmc_run!(mc_params, tempgrid, config, get_eljpot_even(), get_nvt()))
    end
end

@testset "Sampling.jl" begin
    @testset "update_energy_tot" begin
        ensemble = get_nvt()
        test_any(update_energy_tot(get_mcstatevec(ensemble = ensemble), ensemble))
        ensemble = get_npt()
        test_any(update_energy_tot(get_mcstatevec(ensemble = ensemble), ensemble))
    end

    @testset "find_hist_index" begin
        test_any(find_hist_index(30, 50))

        init = initialise()
        test_any(find_hist_index(rand(init[1]), init[3], init[3].delta_en_hist))
        init = initialise(ensemble = get_npt())
        test_any(find_hist_index(rand(init[1]), init[3], init[3].delta_en_hist, init[3].delta_v_hist))
    end

    @testset "initialise_histograms!" begin
        bc = get_spherical_bc()
        mcparams = get_mc_params()
        results = initialise(mc_params = mcparams, ensemble = get_nvt())[3]
        ebounds = [rand()/2, rand()/2 + 0.5]
        test_any(initialise_histograms!(mcparams, results, ebounds, bc))
        bc = get_cubic_bc()
        results = initialise(mc_params = mcparams, ensemble = get_npt(), config = get_config(bc))[3]
        test_any(initialise_histograms!(mcparams, results, ebounds, bc))
        bc = get_rhombic_bc()
        results = initialise(mc_params = mcparams, ensemble = get_npt(), config = get_config(bc))[3]
        test_any(initialise_histograms!(mcparams, results, ebounds, bc))
    end

    @testset "update_histograms!" begin
        init = initialise(ensemble = get_nvt())
        mcstatevec = init[1]
        results = init[3]
        test_any(update_histograms!(mcstatevec, results, results.delta_en_hist))

        init = initialise(ensemble = get_npt())
        mcstatevec = init[1]
        results = init[3]
        test_any(update_histograms!(mcstatevec, results, results.delta_en_hist, results.delta_v_hist))
    end

    @testset "rdf_index" begin
        test_any(rdf_index(rand(), rand()))
    end

    @testset "update_rdf!" begin
        init = initialise(ensemble = get_nvt())
        mcstatevec = init[1]
        results = init[3]
        test_any(update_rdf!(mcstatevec, results, results.delta_r2))
    end
    @testset "sampling_step!" begin
        mcparams = get_mc_params()
        ensemble = get_nvt()
        init = initialise(ensemble = ensemble, mc_params = mcparams)
        mcstatevec = init[1]
        results = init[3]
        test_any(sampling_step!(mcparams, mcstatevec, ensemble, 5, results, true))
        test_any(sampling_step!(mcparams, mcstatevec, ensemble, 5, results, false))

        ensemble = get_npt()
        init = initialise(ensemble = ensemble, mc_params = mcparams)
        mcstatevec = init[1]
        results = init[3]
        test_any(sampling_step!(mcparams, mcstatevec, ensemble, 5, results, true))
        test_any(sampling_step!(mcparams, mcstatevec, ensemble, 5, results, false))
    end

    @testset "finalise_results" begin
        mcparams = get_mc_params()
        init = initialise(mc_params = mcparams)
        mcstatevec = init[1]
        results = init[3]
        test_any(finalise_results(mcstatevec, mcparams, results))
    end
end

@testset "MCStates" begin
end