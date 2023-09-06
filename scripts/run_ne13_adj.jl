using ParallelTemperingMonteCarlo

using Random
using Plots

#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 13

# temperature grid
ti = 5.
tf = 30.
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 1000000 #default 20% equilibration cycles on top


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

#ELJpotential for neon
#c1=[-10.5097942564988, 0., 989.725135614556, 0., -101383.865938807, 0., 3918846.12841668, 0., -56234083.4334278, 0., 288738837.441765]
#elj_ne1 = ELJPotential{11}(c1)

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)

#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne13 = [[2.825384495892464, 0.928562467914040, 0.505520149314310],
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

#convert to Bohr
AtoBohr = 1.8897259886
pos_ne13 = pos_ne13 * AtoBohr

length(pos_ne13) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
bc_ne13 = init_AdjacencyBC(pos_ne13, 4.39*AtoBohr)  #4.39 Angstrom

#starting configuration
start_config = Config(pos_ne13, bc_ne13)

#histogram information
n_bin = 100
#en_min = -0.006    #might want to update after equilibration run if generated on the fly
#en_max = -0.001    #otherwise will be determined after run as min/max of sampled energies (ham vector)

#construct array of MCState (for each temperature)
mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot) for i in 1:n_traj]

#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

@time ptmc_run!((mc_states, move_strat, mc_params, pot, ensemble, results); save=true)

plot(temp.t_grid,results.heat_cap)

plot(multihistogram(results,temp), legend = false, xlabel="Temperature", ylabel="Heat Capacity")
png("ne13-multihistogram-graphs.jl")

data = [results.en_histogram[i] for i in 1:n_traj]
plot(data)

rdf = [results.rdf[i] for i in 1:n_traj]
plot([rdf]; minorticks=10, color=(:thermal), line_z = (1:32)', legend = false, colorbar=true, xlabel="Bins", ylabel="Frequency of occurrence")
png("rdf-ne13-adj-1M")
# png("ne-13-adj-rdf-something-is-wrong")
#png("adjacency1M")
#png("atomloss")

# plot(results.rdf[1])
# png("1")
# plot(results.rdf[2])
# png("2")
# plot(results.rdf[3])
# png("3")
# plot(results.rdf[4])
# png("4")
# plot(results.rdf[5])
# png("5")
# plot(results.rdf[6])
# png("6")
# plot(results.rdf[7])
# png("7")
# plot(results.rdf[8])
# png("8")
# plot(results.rdf[9])
# png("9")
# plot(results.rdf[10])
# png("10")
# plot(results.rdf[11])
# png("11")
# plot(results.rdf[12])
# png("12")
# plot(results.rdf[13])
# png("13")
# plot(results.rdf[14])
# png("14")
# plot(results.rdf[15])
# png("15")
# plot(results.rdf[16])
# png("16")
# plot(results.rdf[17])
# png("17")
# plot(results.rdf[18])
# png("18")
# plot(results.rdf[19])
# png("19")
# plot(results.rdf[20])
# png("20")
# plot(results.rdf[21])
# png("21")
# plot(results.rdf[22])
# png("22")
# plot(results.rdf[23])
# png("23")
# plot(results.rdf[24])
# png("24")
# plot(results.rdf[25])
# png("25")
# plot(results.rdf[26])
# png("26")
# plot(results.rdf[27])
# png("27")
# plot(results.rdf[28])
# png("28")
# plot(results.rdf[29])
# png("29")
# plot(results.rdf[30])
# png("30")
# plot(results.rdf[31])
# png("31")
# plot(results.rdf[32])
# png("32")
