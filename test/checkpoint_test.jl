@testset begin
    using Random 

    test_params = MCParams(1000,5,13,mc_sample=1,n_adjust=100)
    test_temps = TempGrid{5}(12.,16.)


    c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
    test_pot = ELJPotentialEven{6}(c)


    testensemble=NVT(13)
    teststrat=MoveStrategy(testensemble)

    test_pos = [[2.825384495892464, 0.928562467914040, 0.505520149314310],
    [2.023342172678102,	-2.136126268595355, 0.666071287554958],
    [2.033761811732818,	-0.643989413759464, -2.133000349161121],
    [0.979777205108572,	2.312002562803556, -1.671909307631893],
    [0.962914279874254,	-0.102326586625353, 2.857083360096907],
    [0.317957619634043,	2.646768968413408, 1.412132053672896],
    [-2.825388342924982, -0.928563755928189, -0.505520471387560],
    [-0.317955944853142, -2.646769840660271, -1.412131825293682],
    [-0.979776174195320, -2.312003751825495, 1.671909138648006],
    [-0.962916072888105, 0.102326392265998,	-2.857083272537599],
    [-2.023340541398004, 2.136128558801072,	-0.666071089291685],
    [-2.033762834001679, 0.643989905095452, 2.132999911364582],
    [0.000002325340981,	0.000000762100600, 0.000000414930733]]

    test_bc = SphericalBC(radius=6)
    testconfig = Config(test_pos,test_bc)

    save_init(test_pot,testensemble,test_params,test_temps)
    @test ispath("./checkpoint/params.data")

    teststates,teststrat,testresults,nstep,startcount = initialisation(test_params,test_temps,testconfig,test_pot,testensemble)

    for state in teststates
        push!(state.ham, 0)
        push!(state.ham, 0)
    end

    testresults=initialise_histograms!(test_params,testresults,[-0.006 , -0.002],test_bc)
    save_histparams(testresults)

    @test isa(teststates[1],MCState)
    @test isa(testresults,Output)

    Random.seed!(0)
    teststates=mc_cycle!(teststates,teststrat,test_params,test_pot,testensemble,nstep,testresults,1,false)

    @test teststates[1].en_tot != teststates[1].new_en
    @test teststates[1].en_tot != teststates[2].en_tot

    checkpoint(1,teststates,testresults,testensemble,false)
    @test ispath("checkpoint/config.1")


    testingparams,testingensemble,testingpotential,testingstates,movstrat,testingresults,nsteps,startcounter=initialisation(true)
    recentre!(teststates[1].config)
    @test testingensemble == testensemble
    @test testingstates[1].config.pos == teststates[1].config.pos
    @test test_pot == testingpotential

    rm("./checkpoint" , recursive = true)

end