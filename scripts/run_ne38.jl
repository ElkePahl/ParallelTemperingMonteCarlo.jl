using .ParallelTemperingMonteCarlo

using Random
using Plots


#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 38

# temperature grid
ti = 5.
tf = 32.
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

#ELJpotential for neon
#c1=[-10.5097942564988, 0., 989.725135614556, 0., -101383.865938807, 0., 3918846.12841668, 0., -56234083.4334278, 0., 288738837.441765]
#elj_ne1 = ELJPotential{11}(c1)

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)

#starting configurations
#icosahedral ground state of Ne38 (from Cambridge cluster database) in Angstrom
pos_ne38 = [[ 0.1947679907,        0.3306365642,        1.7069272101],
[1.1592174250,       -1.1514615100,       -0.6254746298],
[1.4851406793,       -0.0676273830,        0.9223060046],
[-0.1498046416,        1.4425168343,       -0.9785553065],
[1.4277261305,        0.3530265376,       -0.9475378022],
[-0.6881246261,       -1.5737014419,       -0.3328844168],
[-1.4277352637,       -0.3530034531,        0.9475270683],
[0.6881257085,        1.5736904826,        0.3329032458],
[-1.1592204530,        1.1514535263,        0.6254777879],
[0.1498035273,       -1.4424985165,        0.9785685322],
[-1.4851196066,        0.0676193562,       -0.9223231092],
[-0.7057028384,        0.6207073550,       -1.4756523155],
[-0.8745359533,        0.4648140463,        1.4422103492],
[-0.9742077067,       -0.8837261792,       -1.1536019836],
[-0.1947765396,       -0.3306358487,       -1.7069179299],
[0.3759933035,       -1.7072373106,       -0.0694439840],
[-1.7124296000,        0.3336352522,        0.1307959669],
[0.9143159284,        1.3089975397,       -0.7151210582],
[-0.3759920260,        1.7072300336,        0.0694634263],
[1.7124281219,       -0.3336312342,       -0.1308207313],
[-0.9143187026,       -1.3089785474,        0.7151290509],
[0.9742085109,        0.8837023041,        1.1536069633],
[0.7057104439,       -0.6206907639,        1.4756502961],
[0.8745319670,       -0.4648127187,       -1.4422106957],
[-1.1954804901,       -0.6171923123,       -0.1021449363],
[0.0917363053,       -1.0144887859,       -0.8848410405],
[0.9276243144,       -0.8836123311,        0.4234140820],
[1.1954744473,        0.6171883800,       0.1021399054],
[-0.9276176774,        0.8836123556,       -0.4234173533],
[-0.3595942315,       -0.4863167551,        1.2061133825],
[0.3595891589,        0.4863295901,       -1.2061152849],
[-0.0917352078,        1.0144694592,        0.8848400639],
[0.6410702480,       -0.1978633363,       -0.3898095439],
[-0.4162942817,       -0.0651798741,       -0.6515502084],
[0.1334019604,        0.7474406294,       -0.1600033264],
[-0.6410732823,        0.1978593218,        0.3898012337],
[0.4162968444,        0.0651733322,        0.6515490914],
[-0.1333998872,       -0.7474445984,        0.1600019961]]

#convert to Bohr
AtoBohr = 1.8897259886
pos_ne38 = pos_ne38 * AtoBohr

length(pos_ne38) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
bc_ne38 = SphericalBC(radius=6.6*AtoBohr)   #6.6 Angstrom

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

plot(temp.t_grid,results.heat_cap)

plot(multihistogram(results,temp))

rdf = [results.rdf[i] for i in 1:n_traj]

plot([rdf]; minorticks=10)

# data = [results.en_histogram[i] for i in 1:n_traj]
# plot(data)
