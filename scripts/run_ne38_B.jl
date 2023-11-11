using ParallelTemperingMonteCarlo

using Random,Plots


#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 38

# temperature grid
ti = 3.
tf = 20.
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 500000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

#moves - allowed at present: atom, volume and rotation moves (volume,rotation not yet implemented)
move_strat = MoveStrategy(atom_moves = n_atoms)  

#ensemble
ensemble = NVT(n_atoms)

#c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
#pot = ELJPotentialEven{6}(c)

a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608]
b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751]
c=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203]
pot = ELJPotentialB{6}(a,b,c)

#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne38 = [[1.513743821277026, 4.235379177911825, -0.6355315165625465],
[-0.5012881691741179, 0.7528849613643275, 2.3775376905637957],
[1.1196951023843205, -1.0508942048544472, 0.8660341562894069],
[-1.715955627024763, -1.3462549560966572, 0.8659994912928657],
[-2.2025906300333755, 2.644865941651363, 3.5565473420271463],
[1.142759631318712, -1.076695317138804, -2.2308402858290477],
[-2.457938307506491, 5.137675574643694, 2.056994143641897],
[1.9871856895344233, 0.22232358102331032, 3.5562277643968203],
[-5.275952135260584, 0.24645760818949214, 2.0569893741696674],
[-0.7564681188037087, 3.1995898883713205, -2.2309698522385117],
[-4.653300187926833, 3.165745438361175, 2.0565966590994056],
[-2.9718595635237737, -3.5173576747508504, -0.6456200921459827],
[-4.178623201628314, -0.8801691088986695, -0.6353224429084595],
[-0.5014389901477225, 0.7529430432333368, -0.7113376623954193],
[-1.7334159674934662, -1.3763502801383936, -2.2309508212197993],
[1.7151865492133898, 1.7371399333538846, 0.8661043317545649],
[-1.4952746841273463, -3.944525732327148, 2.057200157689502],
[4.007337051226065, 2.755219692298004, -0.6455824440332554],
[-5.326139640740529, 1.7831376263155394, -0.6451587896663148],
[-2.8731091345720974, 1.2594452007377288, 0.8662148417116422],
[-0.7524890314308538, 3.165094403902495, 0.866062423538938],
[-0.2368117893675837, -1.77751632184258, 3.5564740316886483],
[-3.1908254290740734, 3.7456397400265886, -0.6350476207264452],
[0.7724244336721948, 2.95536484328686, 3.5566816973240654],
[3.0642941964332966, 3.9686454427759736, 2.0570897420439094],
[3.43333442244326, -0.08741530188291055, -0.6356071876631604],
[-2.8261989175966056, -0.2809413800601538, 3.5563941078285244],
[4.27608915150984, 1.2410945537834388, 2.056063587810796],
[0.51075470231327, 5.446970761167683, 2.0563405011387292],
[-4.079067902510893, -2.4506312349313957, 2.0559806529386035],
[1.7465570080151958, 1.7513437396392146, -2.2309338458603754],
[1.4397686251214314, -3.6394795622476894, 2.05666821018291],
[-0.0844774569670562, -3.249027220426714, -0.6351473619869143],
[-0.5016021968214994, 0.752855468386831, -3.8337371271277747],
[-2.9072500712252225, 1.26672944659864, -2.230664911808922],
[3.659084569234754, -1.6442849576883758, 2.0567210402186875],
[2.796475363857865, -2.9164616899800198, -0.6455828039137896],
[-1.012257058148007, 5.659794183120995, -0.6454894810060773]]


#convert to Bohr
#AtoBohr = 1.8897259886
#pos_ne13 = pos_ne13 * AtoBohr

length(pos_ne38) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
bc_ne38 = init_AdjacencyBC(pos_ne38, 3.96)   

#starting configuration
start_config = Config(pos_ne38, bc_ne38)

#histogram information
n_bin = 100
#en_min = -0.006    #might want to update after equilibration run if generated on the fly
#en_max = -0.001    #otherwise will be determined after run as min/max of sampled energies (ham vector)

#construct array of MCState (for each temperature)
mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot) for i in 1:n_traj]

println("initial total energy= ",mc_states[1].en_tot)

#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

@time ptmc_run!((mc_states, move_strat, mc_params, pot, ensemble, results); save=true)

#boundary conditions 
bc_ne38 = init_AdjacencyBC(pos_ne38, 3.96)   

#starting configuration
start_config = Config(pos_ne38, bc_ne38)

#histogram information
n_bin = 100
#en_min = -0.006    #might want to update after equilibration run if generated on the fly
#en_max = -0.001    #otherwise will be determined after run as min/max of sampled energies (ham vector)

#construct array of MCState (for each temperature)
mc_states_1 = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot) for i in 1:n_traj]

println("initial total energy= ",mc_states_1[1].en_tot)

#results = Output(n_bin, max_displ_vec)
results_1 = Output{Float64}(n_bin; en_min = mc_states_1[1].en_tot)

@time ptmc_run!((mc_states_1, move_strat, mc_params, pot, ensemble, results_1); save=true)

plot([multihistogram(results,temp), (multihistogram(results_1,temp))], label = ["Spherical boundary condition" "Adjacency boundary condition"], legend =:bottomright, xlabel="Temperature", ylabel="Heat Capacity")
png("magfields-38-4")



# plot(temp.t_grid,results.heat_cap)

data = [results.en_histogram[i] for i in 1:n_traj]
plot(data)