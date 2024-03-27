@testset begin
    test_params = MCParams(1000,5,13,mc_sample=1,n_adjust=100)
    test_temps = TempGrid{5}(12.,16.)
    c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
    test_pot = ELJPotentialEven{6}(c)
    testensemble=NVT(13)
    teststrat=MoveStrategy(testensemble)

    save_init(test_pot,testensemble,test_params,test_temps)
    

end