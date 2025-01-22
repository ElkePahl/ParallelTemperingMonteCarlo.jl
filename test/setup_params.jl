n_atom = 500
get_nvt() = NVT(n_atom)
get_npt() = NPT(n_atom, 101325)
get_spherical_bc() = SphericalBC(radius=rand()*20 + 10)
get_cubic_bc() = CubicBC(Float64(rand(5:20)))
get_rhombic_bc() = RhombicBC(Float64(rand(5:20)), Float64(rand(5:20)))
get_nvt_bc() = get_spherical_bc()
get_npt_bc() = rand([get_cubic_bc(), get_rhombic_bc()])
get_pos() = rand(SVector{3,Float64}) * 10
get_posvec() = [get_pos() for i in 1:n_atom]
get_config(bc::AbstractBC) = Config(get_posvec(), bc)
get_n_by_n() = rand(n_atom, n_atom)
get_mc_params() = MCParams(rand(10:500), rand(10:500), n_atom; eq_percentage = rand())
get_tempgrid(; n_traj = rand(10:500)) = TempGrid{n_traj}(rand(10:500), rand(10:500))
get_index() = rand(1:n_atom)
get_pot(; ensemble = get_nvt()) = if ensemble isa NVT
    rand([get_eljpot_even(), get_eljpot_b()]) #rand([get_eljpot_even(), get_eam(), get_eljpot_b()]) - EAM is currently problematic.
    else
        rand([get_eljpot_even(), get_eljpot_b()])
    end
get_ensemble() = rand([get_nvt(), get_npt()])
get_eljpot_even() = (c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]; return ELJPotentialEven{6}(c))
initialise(;
    ensemble = get_ensemble(),
    pot = get_pot(ensemble = ensemble),
    config = (
        if ensemble isa NVT
            get_config(get_nvt_bc())
        else
            get_config(get_npt_bc())
        end
    ),
    mc_params = get_mc_params(),
    tempgrid = get_tempgrid(; n_traj = mc_params.n_traj),
    ham = true,
    ebounds = [-100., 100.]
) = (
    output = initialisation(mc_params, tempgrid, config, pot, ensemble);
    if ham
        for state in output[1]
            push!(state.ham, 0.0)
            push!(state.ham, 0.0)
        end
    end;
    results = initialise_histograms!(mc_params, output[3], ebounds, config.bc);
    output = (output[1], output[2], results, output[4], output[5]);
    return output
)
get_trial_pos(config::Config, index::Int) = config.pos[index] + (rand(3)*2 .-1)*0.01
get_new_d2_spherical_vec(d2mat_spherical::Matrix{Float64}, index::Int) = d2mat_spherical[index,:]*rand(0.95:0.01:1.05)
get_mcstatevec(; kargs...) = initialise(; kargs...)[1]
get_mcstate(; kargs...) = rand(get_mcstatevec(; kargs...))
get_eljpot_b() = (
    a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608];
    b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751];
    c1=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203];
    return ELJPotentialB{6}(a,b,c1)
)
get_eam() = (
    nmtobohr = 18.8973;
    evtohartree = 0.0367493;
    n = 8.482;
    m = 4.692;
    ϵ = evtohartree*0.0370;
    a = 0.25*nmtobohr;
    C = 27.561;
    return EmbeddedAtomPotential(n,m,ϵ,C,a)
)
get_elj() = ELJPotential{11}([-10.5097942564988, 0., 989.725135614556, 0., -101383.865938807, 0., 3918846.12841668, 0., -56234083.4334278, 0., 288738837.441765])
function get_RuNNerPotential()
    data_path = joinpath(@__DIR__,"..","scripts","data")
    X = [ 1    1              0.001   0.000  11.338
    1    0              0.001   0.000  11.338
    1    1              0.020   0.000  11.338
    1    0              0.020   0.000  11.338
    1    1              0.035   0.000  11.338
    1    0              0.035   0.000  11.338
    1    1              0.100   0.000  11.338
    1    0              0.100   0.000  11.338
    1    1              0.400   0.000  11.338
    1    0              0.400   0.000  11.338]
    radsymmvec = RadialType2{Float64}[]
    V = [[0.0001,1,1,11.338],[0.0001,-1,2,11.338],[0.003,-1,1,11.338],[0.003,-1,2,11.338],[0.008,-1,1,11.338],[0.008,-1,2,11.338],[0.008,1,2,11.338],[0.015,1,1,11.338],[0.015,-1,2,11.338],[0.015,-1,4,11.338],[0.015,-1,16,11.338],[0.025,-1,1,11.338],[0.025,1,1,11.338],[0.025,1,2,11.338],[0.025,-1,4,11.338],[0.025,-1,16,11.338],[0.025,1,16,11.338],[0.045,1,1,11.338],[0.045,-1,2,11.338],[0.045,-1,4,11.338],[0.045,1,4,11.338],[0.045,1,16,11.338],[0.08,1,1,11.338],[0.08,-1,2,11.338],[0.08,-1,4,11.338],[0.08,1,4,11.338]]
    T = [[1.,1.,1.],[1.,1.,0.],[1.,0.,0.]]
    angularsymmvec = AngularType3{Float64}[]
    file = open(joinpath(data_path,"scaling.data")) # full path "./data/scaling.data"
    scalingvalues = readdlm(file)
    close(file)
    G_value_vec = []
    for row in eachrow(scalingvalues[1:88,:])
        max_min = [row[4],row[3]]
        push!(G_value_vec,max_min)
    end
    for symmindex in eachindex(eachrow(X))
        row = X[symmindex,:]
        radsymm = RadialType2{Float64}(row[3],row[5],[row[1],row[2]],G_value_vec[symmindex])
        push!(radsymmvec,radsymm)
    end
    let n_index = 10
    for element in V
        for types in T
            n_index += 1
            symmfunc = AngularType3{Float64}(element[1],element[2],element[3],11.338,types,G_value_vec[n_index])
            push!(angularsymmvec,symmfunc)
        end
    end
    end
    num_nodes::Vector{Int32} = [88, 20, 20, 1]
    activation_functions::Vector{Int32} = [1, 2, 2, 1]
    file = open(joinpath(data_path, "weights.029.data"), "r+") # "./data/weights.029.data"
    weights=readdlm(file)
    close(file)
    weights = vec(weights)
    nnp = NeuralNetworkPotential(num_nodes,activation_functions,weights)
    return RuNNerPotential(nnp,radsymmvec, angularsymmvec)
end