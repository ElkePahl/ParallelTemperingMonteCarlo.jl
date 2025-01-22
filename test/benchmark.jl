using BenchmarkTools
using ParallelTemperingMonteCarlo
using StaticArrays, LinearAlgebra, DelimitedFiles
using Random

include(joinpath(@__DIR__, "setup_params.jl"))

suite = BenchmarkGroup()

suite["Ensembles"] = BenchmarkGroup()
begin
    global suite["Ensembles"]["get_r_cut"] = BenchmarkGroup()
    begin
        global suite["Ensembles"]["get_r_cut"]["CubicBC"] = @benchmarkable get_r_cut(cubic_bc) setup=(cubic_bc = get_cubic_bc())
        global suite["Ensembles"]["get_r_cut"]["RhombicBC"] = @benchmarkable get_r_cut(rhombic_bc) setup=(rhombic_bc = get_rhombic_bc())
    end
    
    
    global suite["Ensembles"]["NVT"] = BenchmarkGroup()
    begin
        global suite["Ensembles"]["NVT"]["NVT"] = @benchmarkable NVT(n_atoms) setup=(n_atoms = rand(10:500))
    end
    
    global suite["Ensembles"]["MoveStrategy"] = BenchmarkGroup()
    begin
        global suite["Ensembles"]["MoveStrategy"]["NPT"] = @benchmarkable MoveStrategy(npt) setup = (npt = get_npt())
        global suite["Ensembles"]["MoveStrategy"]["NVT"] = @benchmarkable MoveStrategy(nvt) setup = (nvt = get_nvt())
    end
    
    global suite["Ensembles"]["NVTVariables"] = BenchmarkGroup()
    begin
        global suite["Ensembles"]["NVTVariables"]["NVTVariables"] = @benchmarkable NVTVariables{Float64}(index, trial_move) setup=(index = rand(1:10); trial_move = get_pos())
    end
    
    global suite["Ensembles"]["NPT"] = BenchmarkGroup()
    begin
        global suite["Ensembles"]["NPT"]["NPT"] = @benchmarkable NPT(n_atoms, pressure) setup=(n_atoms = rand(10:500); pressure = rand(10:500))
    end
    
    global suite["Ensembles"]["set_ensemble_variables"] = BenchmarkGroup()
    begin
        global suite["Ensembles"]["set_ensemble_variables"]["NPT"] = @benchmarkable set_ensemble_variables(config, npt) setup=(config = get_config(get_npt_bc()); npt = get_npt())
        global suite["Ensembles"]["set_ensemble_variables"]["NVT"] = @benchmarkable set_ensemble_variables(config, nvt) setup=(config = get_config(get_nvt_bc()); nvt = get_nvt())
    end

    global suite["Ensembles"]["NPTVariables"] = BenchmarkGroup()
    begin
        global suite["Ensembles"]["NPTVariables"]["NPTVariables"] = @benchmarkable NPTVariables{Float64}(index, trial_move, trial_config, d2_mat, r_cut, new_r_cut) setup=(index = rand(1:10); trial_move = get_pos(); trial_config = get_config(get_npt_bc()); d2_mat = get_n_by_n(); r_cut = rand(10:500); new_r_cut = rand(10:500))
    end
end

suite["Configurations"] = BenchmarkGroup()
begin
    global suite["Configurations"]["Config"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["Config"]["Config"] = @benchmarkable Config(posvec, bc) setup=(posvec = get_posvec(); bc = get_npt_bc())
    end

    global suite["Configurations"]["distance2"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["distance2"]["SphericalBC"] = @benchmarkable distance2(pos1, pos2, bc) setup = (pos1 = get_pos(); pos2 = get_pos(); bc = get_spherical_bc())
        global suite["Configurations"]["distance2"]["CubicBC"] = @benchmarkable distance2(pos1, pos2, bc) setup=(pos1 = get_pos(); pos2 = get_pos(); bc = get_cubic_bc())
        global suite["Configurations"]["distance2"]["RhombicBC"] = @benchmarkable distance2(pos1, pos2, bc) setup=(pos1 = get_pos(); pos2 = get_pos(); bc = get_rhombic_bc())
    end

    global suite["Configurations"]["get_distance2_mat"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["get_distance2_mat"]["SphericalBC"] = @benchmarkable get_distance2_mat(config) setup=(config = get_config(get_spherical_bc()))
        global suite["Configurations"]["get_distance2_mat"]["CubicBC"] = @benchmarkable get_distance2_mat(config) setup=(config = get_config(get_cubic_bc()))
        global suite["Configurations"]["get_distance2_mat"]["RhombicBC"] = @benchmarkable get_distance2_mat(config) setup=(config = get_config(get_rhombic_bc()))
    end

    global suite["Configurations"]["get_volume"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["get_volume"]["CubicBC"] = @benchmarkable get_volume(cubic_bc) setup=(cubic_bc = get_cubic_bc())
        global suite["Configurations"]["get_volume"]["RhombicBC"] = @benchmarkable get_volume(rhombic_bc) setup=(rhombic_bc = get_rhombic_bc())
    end

    global suite["Configurations"]["get_tantheta_mat"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["get_tantheta_mat"]["SphericalBC"] = @benchmarkable get_tantheta_mat(config, bc) setup=(bc = get_spherical_bc(); config = get_config(bc))
        global suite["Configurations"]["get_tantheta_mat"]["CubicBC"] = @benchmarkable get_tantheta_mat(config, bc) setup=(bc = get_cubic_bc(); config = get_config(bc))
        global suite["Configurations"]["get_tantheta_mat"]["RhombicBC"] = @benchmarkable get_tantheta_mat(config, bc) setup=(bc = get_rhombic_bc(); config = get_config(bc))
    end

    global suite["Configurations"]["get_tan"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["get_tan"]["SphericalBC"] = @benchmarkable get_tan(pos1, pos2, spherical_bc) setup=(pos1 = get_pos(); pos2 = get_pos(); spherical_bc = get_spherical_bc())
        global suite["Configurations"]["get_tan"]["CubicBC"] = @benchmarkable get_tan(pos1, pos2, cubic_bc) setup=(pos1 = get_pos(); pos2 = get_pos(); cubic_bc = get_cubic_bc())
        global suite["Configurations"]["get_tan"]["RhombicBC"] = @benchmarkable get_tan(pos1, pos2, rhombic_bc) setup=(pos1 = get_pos(); pos2 = get_pos(); rhombic_bc = get_rhombic_bc())
    end

    global suite["Configurations"]["get_centre"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["get_centre"]["get_centre"] = @benchmarkable get_centre(posvec, n_atom) setup=(posvec = get_posvec())
    end

    global suite["Configurations"]["recentre!"] = BenchmarkGroup()
    begin
        global suite["Configurations"]["recentre!"]["SphericalBC"] = @benchmarkable recentre!(config) setup=(config = get_config(get_spherical_bc()))
        global suite["Configurations"]["recentre!"]["CubicBC"] = @benchmarkable recentre!(config) setup=(config = get_config(get_cubic_bc()))
        global suite["Configurations"]["recentre!"]["RhombicBC"] = @benchmarkable recentre!(config) setup=(config = get_config(get_rhombic_bc()))
    end
end

suite["BoundaryConditions"] = BenchmarkGroup()
begin
    global suite["BoundaryConditions"]["BC"] = BenchmarkGroup()
    begin
        global suite["BoundaryConditions"]["BC"]["SphericalBC"] = @benchmarkable get_spherical_bc()
        global suite["BoundaryConditions"]["BC"]["CubicBC"] = @benchmarkable get_cubic_bc()
        global suite["BoundaryConditions"]["BC"]["RhombicBC"] = @benchmarkable get_rhombic_bc()
    end

    global suite["BoundaryConditions"]["check_boundary"] = BenchmarkGroup()
    begin
        global suite["BoundaryConditions"]["check_boundary"]["SphericalBC"] = @benchmarkable check_boundary(bc, pos) setup=(bc = get_spherical_bc(); pos = get_pos())
    end
end

suite["InputParams"] = BenchmarkGroup()
begin
    global suite["InputParams"]["MCParams"] = BenchmarkGroup()
    begin
        global suite["InputParams"]["MCParams"]["MCParams"] = @benchmarkable get_mc_params()
    end
    
    global suite["InputParams"]["TempGrid"] = BenchmarkGroup()
    begin
        global suite["InputParams"]["TempGrid"]["TempGrid"] = @benchmarkable get_tempgrid()
    end

    global suite["InputParams"]["Output"] = BenchmarkGroup()
    begin
        global suite["InputParams"]["Output"]["Output"] = @benchmarkable Output{Float64}(rand(10:500))
    end
end

suite["EnergyEvaluation"] = BenchmarkGroup()
begin
    global suite["EnergyEvaluation"]["dimer_energy_atom"] = BenchmarkGroup()
    begin
        r_cut = 50
        global suite["EnergyEvaluation"]["dimer_energy_atom"]["dimer_energy_atom"] = @benchmarkable dimer_energy_atom(index,d2_mat,$(get_eljpot_even())) setup=(index = get_index(); d2_mat = (get_n_by_n()*7)[index,:])
        global suite["EnergyEvaluation"]["dimer_energy_atom"]["dimer_energy_atom_r_cut"] = @benchmarkable dimer_energy_atom(index,d2_mat,r_cut,$(get_eljpot_even())) setup=(index = get_index(); d2_mat = (get_n_by_n()*7)[index,:]; r_cut = 7*rand())
        global suite["EnergyEvaluation"]["dimer_energy_atom"]["dimer_energy_atom_b"] = @benchmarkable dimer_energy_atom(index,d2_mat,thetavec,$(get_eljpot_b())) setup=(index = get_index(); d2_mat = (get_n_by_n()*7)[index,:]; thetavec = (get_n_by_n()*7)[index,:])
        global suite["EnergyEvaluation"]["dimer_energy_atom"]["dimer_energy_atom_b_r_cut"] = @benchmarkable dimer_energy_atom(index,d2_mat,thetavec,r_cut,$(get_eljpot_b())) setup=(index = get_index(); d2_mat = (get_n_by_n()*7)[index,:]; thetavec = (get_n_by_n()*7)[index,:]; r_cut = 7*rand())
    end

    global suite["EnergyEvaluation"]["dimer_energy_config"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["dimer_energy_config"]["ELJEven"] = @benchmarkable dimer_energy_config(d2mat,n_atom,dimer_potential_variables,$(get_eljpot_even())) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config); dimer_potential_variables = set_variables(config, d2mat, get_eljpot_even()))
        global suite["EnergyEvaluation"]["dimer_energy_config"]["ELJEven/CubicBC"] = @benchmarkable dimer_energy_config(d2mat,n_atom,dimer_potential_variables, r_cut, bc, $(get_eljpot_even())) setup=(bc = get_cubic_bc(); config = get_config(bc); d2mat = get_distance2_mat(config); dimer_potential_variables = set_variables(config, d2mat, get_eljpot_even()); r_cut = rand()*7)
        global suite["EnergyEvaluation"]["dimer_energy_config"]["ELJEven/RhombicBC"] = @benchmarkable dimer_energy_config(d2mat,n_atom,dimer_potential_variables, r_cut, bc, $(get_eljpot_even())) setup=(bc = get_rhombic_bc(); config = get_config(bc); d2mat = get_distance2_mat(config); dimer_potential_variables = set_variables(config, d2mat, get_eljpot_even()); r_cut = rand()*7)
        global suite["EnergyEvaluation"]["dimer_energy_config"]["ELJB"] = @benchmarkable dimer_energy_config(d2mat,n_atom,dimer_b_potential_variables,$(get_eljpot_b())) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config); dimer_b_potential_variables = set_variables(config, d2mat, get_eljpot_b()))
        global suite["EnergyEvaluation"]["dimer_energy_config"]["ELJB/CubicBC"] = @benchmarkable dimer_energy_config(d2mat,n_atom,dimer_b_potential_variables, r_cut, bc, $(get_eljpot_b())) setup=(bc = get_cubic_bc(); config = get_config(bc); d2mat = get_distance2_mat(config); dimer_b_potential_variables = set_variables(config, d2mat, get_eljpot_b()); r_cut = rand()*7)
    end

    global suite["EnergyEvaluation"]["dimer_energy_update"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["dimer_energy_update"]["ELJEven"] = @benchmarkable dimer_energy_update!(index,d2mat_spherical,new_d2_spherical_vec,0.0,$(get_eljpot_even())) setup=(index = get_index(); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = (get_n_by_n()*7)[index,:])
        global suite["EnergyEvaluation"]["dimer_energy_update"]["ELJEven/r_cut"] = @benchmarkable dimer_energy_update!(index,d2mat_spherical,new_d2_spherical_vec,0.0, r_cut,$(get_eljpot_even())) setup=(index = get_index(); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = (get_n_by_n()*7)[index,:]; r_cut = 7*rand())
        global suite["EnergyEvaluation"]["dimer_energy_update"]["ELJB"] = @benchmarkable dimer_energy_update!(index,d2mat_spherical,thetamat_spherical,new_d2_spherical_vec,new_tanvec_spherical,0.0,$(get_eljpot_b())) setup=(index = get_index(); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = (get_n_by_n()*7)[index,:]; thetamat_spherical = get_n_by_n()*7; new_tanvec_spherical = thetamat_spherical[index,:])
        global suite["EnergyEvaluation"]["dimer_energy_update"]["ELJB/r_cut"] = @benchmarkable dimer_energy_update!(index,d2mat_spherical,thetamat_spherical,new_d2_spherical_vec,new_tanvec_spherical,0.0,r_cut,$(get_eljpot_b())) setup=(index = get_index(); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = (get_n_by_n()*7)[index,:]; thetamat_spherical = get_n_by_n()*7; new_tanvec_spherical = thetamat_spherical[index,:]; r_cut = 7*rand())
    end

    global suite["EnergyEvaluation"]["set_variables"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["set_variables"]["ELJEven"] = @benchmarkable set_variables(config, d2mat, $(get_eljpot_even())) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config))
        global suite["EnergyEvaluation"]["set_variables"]["ELJB"] = @benchmarkable set_variables(config, d2mat, $(get_eljpot_b())) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config))
        global suite["EnergyEvaluation"]["set_variables"]["EmbeddedAtomPotential"] = @benchmarkable set_variables(config, d2mat, $(get_eam())) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config))
        #global suite["EnergyEvaluation"]["set_variables"]["RuNNerPotential"] = @benchmarkable set_variables(config, d2mat, $(get_RuNNerPotential())) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config))
    end

    global suite["EnergyEvaluation"]["energy_update!"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["energy_update!"]["ELJEven"] = @benchmarkable energy_update!(trial_pos, index, spherical_config, potvars, d2mat_spherical, new_d2_spherical_vec, 1000., eljpot_even) setup=(eljpot_even = get_eljpot_even(); index = get_index(); spherical_config = get_config(get_spherical_bc()); trial_pos = get_trial_pos(spherical_config, index); potvars = set_variables(spherical_config, get_distance2_mat(spherical_config), eljpot_even); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = get_new_d2_spherical_vec(d2mat_spherical, index))
        global suite["EnergyEvaluation"]["energy_update!"]["ELJEven/r_cut"] = @benchmarkable energy_update!(trial_pos, index, spherical_config, potvars, d2mat_spherical, new_d2_spherical_vec, 1000., r_cut, eljpot_even) setup=(eljpot_even = get_eljpot_even(); index = get_index(); spherical_config = get_config(get_spherical_bc()); trial_pos = get_trial_pos(spherical_config, index); potvars = set_variables(spherical_config, get_distance2_mat(spherical_config), eljpot_even); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = get_new_d2_spherical_vec(d2mat_spherical, index); r_cut = 7*rand())
        global suite["EnergyEvaluation"]["energy_update!"]["ELJB"] = @benchmarkable energy_update!(trial_pos, index, spherical_config, potvars, d2mat_spherical, new_d2_spherical_vec, 1000., elj_b) setup=(elj_b = get_eljpot_b(); index = get_index(); spherical_config = get_config(get_spherical_bc()); trial_pos = get_trial_pos(spherical_config, index); potvars = set_variables(spherical_config, get_distance2_mat(spherical_config), elj_b); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = get_new_d2_spherical_vec(d2mat_spherical, index))
        global suite["EnergyEvaluation"]["energy_update!"]["ELJB/r_cut"] = @benchmarkable energy_update!(trial_pos, index, spherical_config, potvars, d2mat_spherical, new_d2_spherical_vec, 1000., elj_b) setup=(elj_b = get_eljpot_b(); index = get_index(); spherical_config = get_config(get_spherical_bc()); trial_pos = get_trial_pos(spherical_config, index); potvars = set_variables(spherical_config, get_distance2_mat(spherical_config), elj_b); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = get_new_d2_spherical_vec(d2mat_spherical, index); r_cut = 7*rand())
        #global suite["EnergyEvaluation"]["energy_update!"]["EmbeddedAtomPotential"] = @benchmarkable energy_update!(trial_pos, index, spherical_config, potvars, d2mat_spherical, new_d2_spherical_vec, 1000., eam) setup=(eam = get_eam(); index = get_index(); spherical_config = get_config(get_spherical_bc()); trial_pos = get_trial_pos(spherical_config, index); potvars = set_variables(spherical_config, get_distance2_mat(spherical_config), eam); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = get_new_d2_spherical_vec(d2mat_spherical, index))
        #global suite["EnergyEvaluation"]["energy_update!"]["RuNNerPotential"] = @benchmarkable energy_update!(trial_pos, index, spherical_config, potvars, d2mat_spherical, new_d2_spherical_vec, 0.0, runner_pot) setup=(runner_pot = get_RuNNerPotential(); index = get_index(); spherical_config = get_config(get_spherical_bc()); trial_pos = get_trial_pos(spherical_config, index); dimer_potential_variables = set_variables(spherical_config, get_distance2_mat(spherical_config), runner_pot); d2mat_spherical = get_n_by_n()*7; new_d2_spherical_vec = (get_n_by_n()*7)[index,:])
    end

    global suite["EnergyEvaluation"]["dimer_energy"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["dimer_energy"]["ELJ"] = @benchmarkable dimer_energy(pot, r2) setup=(pot = get_elj(); r2 = rand()*2)
        global suite["EnergyEvaluation"]["dimer_energy"]["ELJEven"] = @benchmarkable dimer_energy(pot, r2) setup=(pot = get_eljpot_even(); r2 = rand()*2)
        global suite["EnergyEvaluation"]["dimer_energy"]["ELJB"] = @benchmarkable dimer_energy(pot, r2, z_angle) setup=(pot = get_eljpot_b(); r2 = rand()*2; z_angle = rand()*2)
    end
    
    global suite["EnergyEvaluation"]["lrc"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["lrc"]["ELJEven"] = @benchmarkable lrc(n_atoms, r_cut, pot) setup=(n_atoms = n_atom; r_cut = rand()*7; pot = get_eljpot_even())
        global suite["EnergyEvaluation"]["lrc"]["ELJB"] = @benchmarkable lrc(n_atoms, r_cut, pot) setup=(n_atoms = n_atom; r_cut = rand()*7; pot = get_eljpot_b())
    end

    global suite["EnergyEvaluation"]["invrexp"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["invrexp"]["invrexp"] = @benchmarkable invrexp(r2, n, m) setup=(r2 = rand()*2; n = rand()*2; m = rand(1:10))
    end
    
    global suite["EnergyEvaluation"]["calc_components"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["calc_components"]["singular_component"] = @benchmarkable calc_components(component, distancevec, n, m) setup=(component = rand(2); distancevec = rand(10); n = rand(); m = rand())
        global suite["EnergyEvaluation"]["calc_components"]["component_vec"] = @benchmarkable calc_components(componentvec, atomindex, old_dist_vec, new_dist_vec, n, m) setup=(componentvec = rand(10,2); old_dist_vec = rand(10); n = rand(); m = rand(); new_dist_vec = rand(10); atomindex = rand(1:10))
        global suite["EnergyEvaluation"]["calc_components"]["new_component_vec"] = @benchmarkable calc_components(componentvec, new_component_vec,atomindex, old_dist_vec, new_dist_vec, n, m) setup=(componentvec = rand(10,2); old_dist_vec = rand(10); n = rand(); m = rand(); new_dist_vec = rand(10); atomindex = rand(1:10); new_component_vec = rand(10,2))
    end

    global suite["EnergyEvaluation"]["calc_energies_from_components"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["calc_energies_from_components"]["calc_energies_from_components"] = @benchmarkable calc_energies_from_components(componentvec, ean, ecan) setup=(componentvec = rand(2,n_atom); eam = get_eam(); ean = eam.ean; ecan = eam.eCam)
    end

    global suite["EnergyEvaluation"]["initialise_energy"] = BenchmarkGroup()
    begin
        global suite["EnergyEvaluation"]["initialise_energy"]["ELJEven/NVT"] = @benchmarkable initialise_energy(config, d2mat, potvars, ensemble_vars, pot) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config); potvars = set_variables(config, d2mat, get_eljpot_even()); ensemble_vars = set_ensemble_variables(config, get_nvt()); pot = get_eljpot_even())
        global suite["EnergyEvaluation"]["initialise_energy"]["ELJEven/NPT"] = @benchmarkable initialise_energy(config, d2mat, potvars, ensemble_vars, pot) setup=(config = get_config(get_npt_bc()); d2mat = get_distance2_mat(config); potvars = set_variables(config, d2mat, get_eljpot_even()); ensemble_vars = set_ensemble_variables(config, get_npt()); pot = get_eljpot_even())
        global suite["EnergyEvaluation"]["initialise_energy"]["ELJB/NVT"] = @benchmarkable initialise_energy(config, d2mat, potvars, ensemble_vars, pot) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config); potvars = set_variables(config, d2mat, get_eljpot_b()); ensemble_vars = set_ensemble_variables(config, get_nvt()); pot = get_eljpot_b())
        global suite["EnergyEvaluation"]["initialise_energy"]["ELJB/NPT"] = @benchmarkable initialise_energy(config, d2mat, potvars, ensemble_vars, pot) setup=(config = get_config(get_npt_bc()); d2mat = get_distance2_mat(config); potvars = set_variables(config, d2mat, get_eljpot_b()); ensemble_vars = set_ensemble_variables(config, get_npt()); pot = get_eljpot_b())
        global suite["EnergyEvaluation"]["initialise_energy"]["EmbeddedAtomPotential"] = @benchmarkable initialise_energy(config, d2mat, potvars, ensemble_vars, pot) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config); potvars = set_variables(config, d2mat, get_eam()); ensemble_vars = set_ensemble_variables(config, get_nvt()); pot = get_eam())
        #global suite["EnergyEvaluation"]["initialise_energy"]["RuNNerPotential"] = @benchmarkable initialise_energy(config, d2mat, potvars, ensemble_vars, pot) setup=(config = get_config(get_spherical_bc()); d2mat = get_distance2_mat(config); potvars = set_variables(config, d2mat, get_RuNNerPotential()); ensemble_vars = set_ensemble_variables(config, get_nvt()); pot = get_RuNNerPotential())
    end
end

suite["Exchange"] = BenchmarkGroup()
begin
    global suite["Exchange"]["metropolis_condition"] = BenchmarkGroup()
    begin
        global suite["Exchange"]["metropolis_condition"]["delta_energy/beta"] = @benchmarkable metropolis_condition(delta, beta) setup=(delta = rand()*2; beta = rand()*2)
        global suite["Exchange"]["metropolis_condition"]["ensemble/etc"] = @benchmarkable metropolis_condition(ensemble, delta, vi, vf, beta) setup=(ensemble = get_npt(); delta = rand()*2; beta = rand()*2; vi = rand()*2; vf = rand()*2)
        global suite["Exchange"]["metropolis_condition"]["movetype/etc"] = @benchmarkable metropolis_condition(movetype, mcstate, ensemble) setup=(movetype = rand(["atommove", "volumemove"]);
            ensemble = if movetype == "volumemove"
                get_npt()
            else
                get_ensemble()
            end; mcstate = get_mcstate(ensemble = ensemble))
    end

    global suite["Exchange"]["exc_acceptance"] = BenchmarkGroup()
    begin
        global suite["Exchange"]["exc_acceptance"]["exc_acceptance"] = @benchmarkable exc_acceptance(b1, b2, en1, en2) setup=(b1 = rand()*2; b2 = rand()*2; en1 = rand()*2; en2 = rand()*2)
    end
    
    global suite["Exchange"]["exc_trajectories!"] = BenchmarkGroup()
    begin
        global suite["Exchange"]["exc_trajectories!"]["exc_trajectories!"] = @benchmarkable exc_trajectories!(mcstate1, mcstate2) setup=(init = initialise(); mcstate1 = rand(init[1]); mcstate2 = rand(init[1]))
    end

    global suite["Exchange"]["parallel_tempering_exchange!"] = BenchmarkGroup()
    begin
        global suite["Exchange"]["parallel_tempering_exchange!"]["NVT"] = @benchmarkable parallel_tempering_exchange!(mcstates, mcparams, ensemble) setup=(ensemble = get_nvt(); mcparams = get_mc_params(); mcstates = get_mcstatevec(ensemble = ensemble, mc_params = mcparams))
        global suite["Exchange"]["parallel_tempering_exchange!"]["NPT"] = @benchmarkable parallel_tempering_exchange!(mcstates, mcparams, ensemble) setup=(ensemble = get_npt(); mcparams = get_mc_params(); mcstates = get_mcstatevec(ensemble = ensemble, mc_params = mcparams))
    end
    
    global suite["Exchange"]["update_max_stepsize!"] = BenchmarkGroup()
    begin
        global suite["Exchange"]["update_max_stepsize!"]["NVT"] = @benchmarkable update_max_stepsize!(mcstate, n_update, ensemble, min_acc, max_acc) setup=(ensemble = get_nvt(); mcstate = get_mcstate(ensemble = ensemble); n_update = rand(1:10); min_acc = rand(); max_acc = rand()*(1-min_acc)+min_acc)
        global suite["Exchange"]["update_max_stepsize!"]["NPT"] = @benchmarkable update_max_stepsize!(mcstate, n_update, ensemble, min_acc, max_acc) setup=(ensemble = get_npt(); mcstate = get_mcstate(ensemble = ensemble); n_update = rand(1:10); min_acc = rand(); max_acc = rand()*(1-min_acc)+min_acc)
    end
end

suite["Initialization"] = BenchmarkGroup()
begin
    global suite["Initialization"]["initialisation"] = BenchmarkGroup()
    begin
        global suite["Initialization"]["initialisation"]["NoRestart"] = @benchmarkable initialise(ham = false)
        global suite["Initialization"]["initialisation"]["Restart"] = @benchmarkable initialisation(true, eq_cycles) setup=(eq_cycles = rand()) 
    end
end

suite["MCMoves"] = BenchmarkGroup()
begin
    global suite["MCMoves"]["atom_displacement"] = BenchmarkGroup()
    begin
        global suite["MCMoves"]["atom_displacement"]["SphericalBC"] = @benchmarkable atom_displacement(trial_pos, max_displacement, bc) setup=(trial_pos = get_pos(); max_displacement = rand()*2; bc = get_spherical_bc())
        global suite["MCMoves"]["atom_displacement"]["CubicBC"] = @benchmarkable atom_displacement(trial_pos, max_displacement, bc) setup=(trial_pos = get_pos(); max_displacement = rand()*2; bc = get_cubic_bc())
        global suite["MCMoves"]["atom_displacement"]["RhombicBC"] = @benchmarkable atom_displacement(trial_pos, max_displacement, bc) setup=(trial_pos = get_pos(); max_displacement = rand()*2; bc = get_rhombic_bc())
        global suite["MCMoves"]["atom_displacement"]["NVT"] = @benchmarkable atom_displacement(mcstate) setup=(mcstate = get_mcstate(ensemble = get_nvt()))
        global suite["MCMoves"]["atom_displacement"]["NPT"] = @benchmarkable atom_displacement(mcstate) setup=(mcstate = get_mcstate(ensemble = get_npt()))
    end

    global suite["MCMoves"]["volume_change"] = BenchmarkGroup()
    begin
        global suite["MCMoves"]["volume_change"]["CubicBC"] = @benchmarkable volume_change(config, bc, max_vchange, maxlength) setup=(bc = get_cubic_bc(); config = get_config(bc); max_vchange = rand()*2; maxlength = rand()*2)
        global suite["MCMoves"]["volume_change"]["RhombicBC"] = @benchmarkable volume_change(config, bc, max_vchange, maxlength) setup=(bc = get_rhombic_bc(); config = get_config(bc); max_vchange = rand()*2; maxlength = rand()*2)
        global suite["MCMoves"]["volume_change"]["NPTState"] = @benchmarkable volume_change(mcstate) setup=(mcstate = get_mcstate(ensemble = get_npt()))
    end

    global suite["MCMoves"]["generate_move"] = BenchmarkGroup()
    begin
        global suite["MCMoves"]["generate_move"]["NVT"] = @benchmarkable generate_move!(mcstate, movetype) setup=(mcstate = get_mcstate(ensemble = get_nvt()); movetype = "atommove")
        global suite["MCMoves"]["generate_move"]["NPT"] = @benchmarkable generate_move!(mcstate, movetype) setup=(mcstate = get_mcstate(ensemble = get_npt()); movetype = rand(["atommove", "volumemove"]))
    end
end

suite["MCRun"] = BenchmarkGroup()
begin
    global suite["MCRun"]["get_energy!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["get_energy!"]["NVT"] = @benchmarkable get_energy!(mcstate, pot, movetype) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); mcstate = get_mcstate(ensemble = ensemble, pot = pot); movetype = "atommove")
        global suite["MCRun"]["get_energy!"]["NPT"] = @benchmarkable get_energy!(mcstate, pot, movetype) setup=(ensemble = get_npt(); pot = get_pot(ensemble = ensemble); mcstate = get_mcstate(ensemble = ensemble, pot = pot); movetype = rand(["atommove", "volumemove"]))
    end

    global suite["MCRun"]["acc_test!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["acc_test!"]["NVT"] = @benchmarkable acc_test!(mcstate, ensemble, movetype) setup=(ensemble = get_nvt(); mcstate = get_mcstate(ensemble = ensemble); movetype = "atommove")
        global suite["MCRun"]["acc_test!"]["NPT"] = @benchmarkable acc_test!(mcstate,ensemble, movetype) setup=(ensemble = get_npt(); mcstate = get_mcstate(ensemble = ensemble); movetype = rand(["atommove", "volumemove"]))
    end

    global suite["MCRun"]["swap_config!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["swap_config!"]["atommove"] = @benchmarkable swap_config!(mcstate, movetype) setup=(mcstate = get_mcstate(); movetype = "atommove")
        global suite["MCRun"]["swap_config!"]["volumemove"] = @benchmarkable swap_config!(mcstate, movetype) setup=(mcstate = get_mcstate(ensemble = get_npt()); movetype = "volumemove")
    end

    global suite["MCRun"]["swap_atom_config!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["swap_atom_config!"]["swap_atom_config!"] = @benchmarkable swap_atom_config!(mcstate, atom_index, trial_pos) setup=(mcstate = get_mcstate(); atom_index = get_index(); trial_pos = get_pos())
    end

    global suite["MCRun"]["swap_config_v!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["swap_config_v!"]["CubicBC"] = @benchmarkable swap_config_v!(mc_state, bc, trial_config, new_dist2_mat, en_vec_new, en_tot) setup=(ensemble = get_npt(); bc = get_cubic_bc(); mc_state = get_mcstate(ensemble = ensemble, config = get_config(bc)); trial_config = mc_state.ensemble_variables.trial_config; new_dist2_mat = mc_state.ensemble_variables.new_dist2_mat; en_vec_new = mc_state.potential_variables.en_atom_vec; en_tot = mc_state.new_en)
        global suite["MCRun"]["swap_config_v!"]["RhombicBC"] = @benchmarkable swap_config_v!(mc_state, bc, trial_config, new_dist2_mat, en_vec_new, en_tot) setup=(ensemble = get_npt(); bc = get_rhombic_bc(); mc_state = get_mcstate(ensemble = ensemble, config = get_config(bc)); trial_config = mc_state.ensemble_variables.trial_config; new_dist2_mat = mc_state.ensemble_variables.new_dist2_mat; en_vec_new = mc_state.potential_variables.en_atom_vec; en_tot = mc_state.new_en)
    end

    global suite["MCRun"]["swap_vars!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["swap_vars!"]["DimerPotentialVariables"] = @benchmarkable swap_vars!(atom_index, pot_vars) setup=(atom_index = get_index(); config = get_config(get_nvt_bc()); pot_vars = set_variables(config, get_distance2_mat(config), get_eljpot_even()))
        global suite["MCRun"]["swap_vars!"]["ELJBVariables"] = @benchmarkable swap_vars!(atom_index, pot_vars) setup=(atom_index = get_index(); config = get_config(get_nvt_bc()); pot_vars = set_variables(config, get_distance2_mat(config), get_eljpot_b()))
        global suite["MCRun"]["swap_vars!"]["EAMVariables"] = @benchmarkable swap_vars!(atom_index, pot_vars) setup=(atom_index = get_index(); config = get_config(get_nvt_bc()); pot_vars = set_variables(config, get_distance2_mat(config), get_eam()))
        #global suite["MCRun"]["swap_vars!"]["RuNNerVariables"] = @benchmarkable swap_vars!(atom_index, pot_vars) setup=(atom_index = get_index(); config = get_config(get_nvt_bc()); pot_vars = set_variables(config, get_distance2_mat(config), get_RuNNerPotential()))
    end

    global suite["MCRun"]["mc_move!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["mc_move!"]["NVT"] = @benchmarkable mc_move!(mcstate, movestrat, pot, ensemble) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); init = initialise(ensemble = ensemble, pot = pot); mcstate = rand(init[1]); movestrat = init[2])
        global suite["MCRun"]["mc_move!"]["NPT"] = @benchmarkable mc_move!(mcstate, movestrat, pot, ensemble) setup=(ensemble = get_npt(); pot = get_pot(ensemble = ensemble); init = initialise(ensemble = ensemble, pot = pot); mcstate = rand(init[1]); movestrat = init[2])
    end

    global suite["MCRun"]["mc_step!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["mc_step!"]["NVT"] = @benchmarkable mc_step!(mcstatevec, movestrat, pot, ensemble, n_steps) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); init = initialise(ensemble = ensemble, pot = pot); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4])
        global suite["MCRun"]["mc_step!"]["NPT"] = @benchmarkable mc_step!(mcstatevec, movestrat, pot, ensemble, n_steps) setup=(ensemble = get_npt(); pot = get_pot(ensemble = ensemble); init = initialise(ensemble = ensemble, pot = pot); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4])
    end

    global suite["MCRun"]["mc_cycle!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["mc_cycle!"]["SphericalBC"] = @benchmarkable mc_cycle!(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, index) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams, pot = pot); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; index = rand(1:n_steps))
        global suite["MCRun"]["mc_cycle!"]["CubicBC"] = @benchmarkable mc_cycle!(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, index) setup=(ensemble = get_npt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams, config = get_config(get_cubic_bc()), pot = pot); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; index = rand(1:n_steps))
        global suite["MCRun"]["mc_cycle!"]["RhombicBC"] = @benchmarkable mc_cycle!(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, index) setup=(ensemble = get_npt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams, config = get_config(get_rhombic_bc()), pot = pot); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; index = rand(1:n_steps))
        global suite["MCRun"]["mc_cycle!"]["Restart/no_save"] = @benchmarkable mc_cycle!(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, results, index, rdfsave) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams, pot = pot); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; index = rand(1:n_steps); results = init[3]; rdfsave = false)
        global suite["MCRun"]["mc_cycle!"]["Restart/save"] = @benchmarkable mc_cycle!(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, results, index, rdfsave) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams, pot = pot); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; index = rand(1:n_steps); results = init[3]; rdfsave = true)
    end

    global suite["MCRun"]["check_e_bounds"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["check_e_bounds"]["check_e_bounds"] = @benchmarkable check_e_bounds(energy, ebounds) setup=(energy = rand(); ebounds = [rand()/2, rand()/2 + 0.5])
    end

    global suite["MCRun"]["reset_counters"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["reset_counters"]["reset_counters"] = @benchmarkable reset_counters(mcstate) setup=(mcstate = get_mcstate())
    end

    global suite["MCRun"]["equilibration_cycle!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["equilibration_cycle!"]["NVT"] = @benchmarkable equilibration_cycle!(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, results) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(pot = pot, ensemble = ensemble, mc_params = mcparams); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; results = init[3])
        global suite["MCRun"]["equilibration_cycle!"]["NPT"] = @benchmarkable equilibration_cycle!(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, results) setup=(ensemble = get_npt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(pot = pot, ensemble = ensemble, mc_params = mcparams); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; results = init[3])
    end

    global suite["MCRun"]["equilibration"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["equilibration"]["NVT"] = @benchmarkable equilibration(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, results, false) setup=(ensemble = get_nvt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(pot = pot, ensemble = ensemble, mc_params = mcparams, ham = false); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; results = init[3])
        global suite["MCRun"]["equilibration"]["NPT"] = @benchmarkable equilibration(mcstatevec, movestrat, mcparams, pot, ensemble, n_steps, results, false) setup=(ensemble = get_npt(); pot = get_pot(ensemble = ensemble); mcparams = MCParams(100, 20, n_atom); init = initialise(pot = pot, ensemble = ensemble, mc_params = mcparams, ham = false); mcstatevec = init[1]; movestrat = init[2]; n_steps = init[4]; results = init[3])
    end

    global suite["MCRun"]["ptmc_run!"] = BenchmarkGroup()
    begin
        global suite["MCRun"]["ptmc_run!"]["NVT"] = @benchmarkable ptmc_run!(mcparams, tempgrid, config, pot, ensemble) setup=(mcparams = MCParams(100, 20, n_atom); tempgrid = get_tempgrid(n_traj = mcparams.n_traj); config = get_config(get_spherical_bc()); ensemble = get_nvt(); pot = get_pot(ensemble = ensemble))
        global suite["MCRun"]["ptmc_run!"]["NPT"] = @benchmarkable ptmc_run!(mcparams, tempgrid, config, pot, ensemble) setup=(mcparams = MCParams(100, 20, n_atom); tempgrid = get_tempgrid(n_traj = mcparams.n_traj); config = get_config(get_npt_bc()); ensemble = get_npt(); pot = get_pot(ensemble = ensemble))
    end
end

suite["Sampling"] = BenchmarkGroup()
begin
    global suite["Sampling"]["update_energy_tot"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["update_energy_tot"]["NVT"] = @benchmarkable update_energy_tot(mcstatevec, ensemble) setup=(ensemble = get_nvt(); mcstatevec = get_mcstatevec(ensemble = ensemble))
        global suite["Sampling"]["update_energy_tot"]["NPT"] = @benchmarkable update_energy_tot(mcstatevec, ensemble) setup=(ensemble = get_npt(); mcstatevec = get_mcstatevec(ensemble = ensemble))
    end

    global suite["Sampling"]["find_hist_index"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["find_hist_index"]["no_v"] = @benchmarkable find_hist_index(mcstate, results, delta_en_hist) setup=(init = initialise(); mcstate = rand(init[1]); results = init[3]; delta_en_hist = rand()*5)
        global suite["Sampling"]["find_hist_index"]["has_v"] = @benchmarkable find_hist_index(mcstate, results, delta_en_hist, delta_v_hist) setup=(init = initialise(ensemble = get_npt()); mcstate = rand(init[1]); results = init[3]; delta_en_hist = rand()*5; delta_v_hist = rand()*5)
    end

    global suite["Sampling"]["initialise_histograms"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["initialise_histograms"]["SphericalBC"] = @benchmarkable initialise_histograms!(mcparams, results, ebounds, bc) setup=(bc = get_spherical_bc(); mcparams = get_mc_params(); results = initialise(mc_params = mcparams, ensemble = get_nvt())[3]; ebounds = [rand()/2, rand()/2 + 0.5])
        global suite["Sampling"]["initialise_histograms"]["CubicBC"] = @benchmarkable initialise_histograms!(mcparams, results, ebounds, bc) setup=(bc = get_cubic_bc(); mcparams = get_mc_params(); results = initialise(mc_params = mcparams, ensemble = get_npt(), config = get_config(bc))[3]; ebounds = [rand()/2, rand()/2 + 0.5])
        global suite["Sampling"]["initialise_histograms"]["RhombicBC"] = @benchmarkable initialise_histograms!(mcparams, results, ebounds, bc) setup=(bc = get_rhombic_bc(); mcparams = get_mc_params(); results = initialise(mc_params = mcparams, ensemble = get_npt(), config = get_config(bc))[3]; ebounds = [rand()/2, rand()/2 + 0.5])
    end

    global suite["Sampling"]["update_histograms!"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["update_histograms"]["no_v"] = @benchmarkable update_histograms!(mcstatevec, results, delta_en_hist) setup=(init = initialise(); mcstatevec = init[1]; results = init[3]; delta_en_hist = rand()*5)
        global suite["Sampling"]["update_histograms"]["has_v"] = @benchmarkable update_histograms!(mcstatevec, results, delta_en_hist, delta_v_hist) setup=(init = initialise(ensemble = get_npt()); mcstatevec = init[1]; results = init[3]; delta_en_hist = rand()*5; delta_v_hist = rand()*5)
    end

    global suite["Sampling"]["rdf_index"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["rdf_index"]["rdf_index"] = @benchmarkable rdf_index(r2val, delta_r2) setup=(r2val = rand()*5; delta_r2 = rand()*5)
    end

    global suite["Sampling"]["update_rdf!"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["update_rdf!"]["update_rdf!"] = @benchmarkable update_rdf!(mcstatevec, results, delta_en_hist) setup=(init = initialise(); mcstatevec = init[1]; results = init[3]; delta_en_hist = rand()*5)
    end

    global suite["Sampling"]["sampling_step!"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["sampling_step!"]["NVT/no_rdf"] = @benchmarkable sampling_step!(mcparams, mcstatevec, ensemble, save_index, results, rdfsave) setup=(ensemble = get_nvt(); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams); mcstatevec = init[1]; results = init[3]; save_index = rand(10:15); rdfsave = false)
        global suite["Sampling"]["sampling_step!"]["NVT/rdf"] = @benchmarkable sampling_step!(mcparams, mcstatevec, ensemble, save_index, results, rdfsave) setup=(ensemble = get_nvt(); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams); mcstatevec = init[1]; results = init[3]; save_index = rand(10:15); rdfsave = true)
        global suite["Sampling"]["sampling_step!"]["NPT/no_rdf"] = @benchmarkable sampling_step!(mcparams, mcstatevec, ensemble, save_index, results, rdfsave) setup=(ensemble = get_npt(); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams); mcstatevec = init[1]; results = init[3]; save_index = rand(10:15); rdfsave = false)
        global suite["Sampling"]["sampling_step!"]["NPT/rdf"] = @benchmarkable sampling_step!(mcparams, mcstatevec, ensemble, save_index, results, rdfsave) setup=(ensemble = get_npt(); mcparams = MCParams(100, 20, n_atom); init = initialise(ensemble = ensemble, mc_params = mcparams); mcstatevec = init[1]; results = init[3]; save_index = rand(10:15); rdfsave = true)
    end

    global suite["Sampling"]["finalise_results"] = BenchmarkGroup()
    begin
        global suite["Sampling"]["finalise_results"]["finalise_results"] = @benchmarkable finalise_results(mcstatevec, mcparams, results) setup=(mcparams = get_mc_params(); init = initialise(mc_params = mcparams); mcstatevec = init[1]; results = init[3])
    end
end

suite["MCStates"] = BenchmarkGroup()
begin
    global suite["MCStates"]["MCState"] = BenchmarkGroup()
    begin
        global suite["MCStates"]["MCState"]["MCState"] = @benchmarkable MCState(temp, beta, config, ensemble, pot) setup=(temp = rand()*2; beta = 1/temp; config = get_config(get_spherical_bc()); ensemble = get_nvt(); pot = get_pot())
    end
end
cd(joinpath(@__DIR__, "testing_data/"))
#It is highly recommended to not benchmark the whole test suite at once, as it can take a long time to run,
#and is probably unnecessary. Instead, here index into only the functions you wish to benchmark.
to_run = suite
tune!(to_run)
cd(joinpath(@__DIR__, ".."))
trials = run(to_run)
# I hope you ran this in the Julia REPL, because to view the results you need to run something simiilar to the following:
# using BenchmarkPlots, StatsPlots
# plot(trials["function_name"]["benchmark_descriptor"])

# Only individual trials, not BenchmarkGroups, can be plot, so how to visualise the results
#will depend on what exactly you decided to tune and run.