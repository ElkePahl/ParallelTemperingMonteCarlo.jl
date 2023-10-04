using ParallelTemperingMonteCarlo

using Random
using Plots


#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 38

# temperature grid
ti = 5.
tf = 25.
n_traj = 32

temp = TempGrid{n_traj}(ti,tf)

# MC simulation details

mc_cycles = 100 #default 20% equilibration cycles on top


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
#icosahedral ground state of Ne38 (from Cambridge cluster database) in Angstrom
pos_ne38 = [[ 0.5395671776, 0.9164724369, 4.7236393927],
[ 3.2125126585, -3.1858562647, -1.7338261986],
[ 4.1114809260, -0.1870902511, 2.5541423705],
[-0.4149284140, 3.9969968769, -2.7113312452],
[ 3.9534009878, 0.9763870919, -2.6236897159],
[-1.9067278997, -4.3661728391, -0.9220199918],
[-3.9548750415, -0.9763666744, 2.6233580186],
[ 1.9016230366, 4.3605135852, 0.9225266815],
[-3.2099827159, 3.1854663613, 1.7343613152],
[ 0.4145501100, -3.9971588086, 2.7119943025],
[-4.1110115139, 0.1870690189, -2.5527476195],
[-1.9530741228, 1.7189422635, -4.0851757473],
[-2.4205640455, 1.2888220899, 4.0000957602],
[-2.6979046843, -2.4483284073, -3.1890041039],
[-0.5396642768, -0.9164685233, -4.7251252604],
[ 1.0432682889, -4.7248008396, -0.1921669096],
[-4.7372927720, 0.9240120979, 0.3621586429],
[ 2.5317590070, 3.6276047065, -1.9823346752],
[-1.0430968832, 4.7247330713, 0.1924293164],
[ 4.7371056882, -0.9240091002, -0.3614015542],
[-2.5319498665, -3.6276079920, 1.9819952207],
[ 2.6979478969, 2.4482697342, 3.1892747241],
[ 1.9545013634, -1.7214869486, 4.0860058100],
[ 2.4068001545, -1.2889447480, -4.0001642450],
[-3.3113178927, -1.7100584001, -0.2835136174],
[ 0.2539023156, -2.8113502581, -2.4510248643],
[ 2.5709337917, -2.4488006035, 1.1721721694],
[ 3.3094381821, 1.7100011163, 0.2829646823],
[-2.5708516181, 2.4488006646, -1.1732903691],
[-0.9953106186, -1.3451391892, 3.3408830613],
[ 0.9952792196, 1.3451865237, -3.3409880508],
[-0.2538356801, 2.8063198009, 2.4496304029],
[ 1.7740773240, -0.5486462680, -1.0808440158],
[-1.1532342966, -0.1804032990, -1.8014620122],
[ 0.3699702073, 2.0721784406, -0.4424089080],
[-1.7741664861, 0.5486301224, 1.0805295325],
[ 1.1532228746, 0.1803935721, 1.8014422635],
[-0.3699324143, -2.0722190163, 0.4423972381]]

#convert to Bohr
AtoBohr = 1.8897259886
pos_ne38 = pos_ne38 * AtoBohr

length(pos_ne38) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
bc_ne38 = init_AdjacencyBC(pos_ne38, 4*AtoBohr)

#starting configuration
start_config = Config(pos_ne38, bc_ne38)

#histogram information
n_bin = 100
#en_min = -0.006    #might want to update after equilibration run if generated on the fly
#en_max = -0.001    #otherwise will be determined after run as min/max of sampled energies (ham vector)

#construct array of MCState (for each temperature)
mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot) for i in 1:n_traj]

#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

@time ptmc_run!((mc_states, move_strat, mc_params, pot, ensemble, results); save=true)

plot(temp.t_grid,results.heat_cap, xlabel="Temperature", ylabel="Heat Capacity")
png("Ne38-adj")

plot(multihistogram(results,temp), xlabel="Temperature", ylabel="Heat Capacity")
png("ne38-multi-adj-2")

rdf = [results.rdf[i] for i in 1:n_traj]
#println(results.rdf[1])

plot([rdf]; minorticks=10, color=(:thermal), line_z = (1:32)', legend = false, colorbar=true, xlabel="Bins", ylabel="Frequency of occurrence",)
png("Ne38RDF-adj")

data = [results.en_histogram[i] for i in 1:n_traj]
plot(data)
png("ne38hist")