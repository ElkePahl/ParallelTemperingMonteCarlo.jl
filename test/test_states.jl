using Random 

@testset "States"  begin
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = SphericalBC(radius=5.0)
    conf1 = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf1)
    c1 = [-2.,1.]
    pot1 = ELJPotentialEven{2}(c1)

    ensemble = NVT(3)

    temp = TempGrid{2}(10,15)

    state = MCState(temp.t_grid[1],temp.beta_grid[1],conf1,ensemble,pot1)
    state2 = MCState(temp.t_grid[2],temp.beta_grid[2],conf1,ensemble,pot1)
    @test typeof(state.ensemble_variables) == NVTVariables{Float64}
    @test typeof(state.potential_variables) == DimerPotentialVariables{Float64}

    @test state.en_tot == dimer_energy_config(state.dist2_mat,3,state.potential_variables,pot1)[2]

    state.ensemble_variables.index = 1

    Random.seed!(1234)

    trialpos = atom_displacement(state.config.pos[1] , state.max_displ[1] , state.config.bc)

    Random.seed!(1234)
    generate_move!(state,"atommove")

    @test trialpos == state.ensemble_variables.trial_move
    state = get_energy!(state,pot1,"atommove")
    @test state.new_en - state.en_tot ≈ -4.99895252855e-5
    @test metropolis_condition("atommove",state,ensemble) == 1.0

    MCRun.swap_config!(state,"atommove")
    @test state.ensemble_variables.trial_move == state.config.pos[1]
    @test state.new_dist2_vec == state.dist2_mat[1,:]

    @test state2.en_tot - state.en_tot  ≈ 4.99895252855e-5

    update_max_stepsize!(state,3,ensemble)
    @test state.max_displ[1] < 0.1
    exc_trajectories!(state,state2)
    @test state2.en_tot - state.en_tot  ≈ -4.99895252855e-5
    @test state.config.pos == conf1.pos


end 