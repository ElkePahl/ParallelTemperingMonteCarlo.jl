using Random 


@testset "swap"  begin
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = CubicBC(10.0)
    conf = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf)

    a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608]
    b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751]
    c1=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203]
    potB = ELJPotentialB{6}(a,b,c1)

    ensemble = NPT(3,101325*3.398928944382626e-14,false)

    temp = TempGrid{2}(10,15)

    state = MCState(temp.t_grid[1],temp.beta_grid[1],conf,ensemble,potB)
    
    state.ensemble_variables.index = rand(1:3)
    state = atom_displacement(state)
    state.potential_variables,state.new_en = energy_update!(state.ensemble_variables.trial_move,state.ensemble_variables.index,state.config,state.potential_variables,state.dist2_mat,state.new_dist2_vec,state.en_tot,state.ensemble_variables.r_cut,potB)

    
    
end