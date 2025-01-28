# # Example 1: Melting a 13-Atoms Neon Cluster 
# This is an example calculation for finding the melting temperature of a 13-atoms neon cluster using a parallel-tempering Monte Carlo simulation.
# First, we load ParallelTemperingMonteCarlo and Plots:
using Plots
using ParallelTemperingMonteCarlo
# ## Setting up the Model
# Firstly, we set the number of atoms:
n_atoms = 13;
# Next, we define the potential, here a Lennard Jones potential with coefficients for even powers of r, starting from -6 and decreasing:
c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)
# We further have to define the starting configuration of the simulation. For Ne13 we choose the icosahedral ground state of Ne13 (from the Cambridge cluster database). 
# The atomic positions are given in Angstrom, which are then converted to Bohr radii (as the program uses atomic units).
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
[0.000002325340981,	0.000000762100600, 0.000000414930733]];
AtoBohr = 1.8897259886;
pos_ne13 = pos_ne13 * AtoBohr
# Finally, we have to choose appropriate boundary conditions, here spherical boundary conditions (solid boundary around the cluster), to suppress atom loss processes.  
# Finding this radius is a non-trivial task, and has to be chosen and tested carefully. A radius chosen too small wil exert artificial pressure on the cluster while a too large value leads to atoms being ejected.
bc_ne13 = SphericalBC(radius=5.32*AtoBohr) 
# We package the initial configuration and boundary conditions into a Config struct:
start_config = Config(pos_ne13, bc_ne13)
# ## Setting up the simulation parameters
# We  first set the temperature grid, which defines the range of temperatures we consider.
# This is done by defining the upper and lower temperature limits, along with the number of temperatures (also called trajectories) we want to sample.
# Note, that a geometrical distribution of temperatures is chosen to maximise overlaps in the energy histograms.
ti = 4.;
tf = 16.;
n_traj = 25;
temp = TempGrid{n_traj}(ti,tf)
# Now we set the hyperparameters for this simulation:
# - `mc_cycles` is the number of Monte Carlo cycles we want to run, the longer the more accurate but also more expensive.
# - `mc_sample` is the number of MC cycles after which the energy of the state is recorded.
# - `max_displ_atom` determines the maximum displacement of an atom over a cycle. The maximum displacement is automatically adjusted in the program guaranteeing a 40-60% acceptance rate.
# - `n_adjust` is the number of moves after which the step size of atom moves is adjusted.
mc_cycles = 1000;
mc_sample = 1;
displ_atom = 0.1;
max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj];
n_adjust = 100;
# For neatness, all parameters are collected in a `MCParams` struct:
mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)
# We then define the ensemble, here we are using the NVT ensemble (keeping N, the number of atoms, V, the volume, and T, the temperature constant).
# This allows us to derive a MoveStrategy to feed into the PTMC simulation. Here, we do `n_atoms` atom displacements of randomy chosen atoms per Monte Carlo cycle.
ensemble = NVT(n_atoms);
move_strat = MoveStrategy(ensemble)
# ## Running the Simulation
# Finally, we run the simulation. This method returns the current state and results of the simulation.
# The data is stored in various local files created in the current working directory.
mc_states, results = ptmc_run!(mc_params,temp,start_config,pot,ensemble;save=1000);
# ## Post-processing and analyzing of results
# The raw heat capacity plot is obtained from:
plot(temp.t_grid,results.heat_cap)
# and the energy histograms by:
data = [results.en_histogram[i] for i in 1:n_traj]
plot(data)
# For post-processing of the data we use the multihistogram method. This method accesses the stored data created from the ptmc_run! method 
# and returns values for the energies, histogram data, temperature, partition function, heat capacity, heat capacity gradient, and entropy, which can be plotted as shown:
energies,histogramdata,T,Z,Cv,dCv,S = postprocess(; xdir=joinpath("../..", "examples"));
# Plot of heat capacity against temperature:
plot(T,Cv,label="Cv")
# Plot of the heat capacity gradient against temperature:
plot(T,dCv,label="dCv")
# Plot of the partition function against temperature:
plot(T,Z,label="Z")
# Plot of the entropy of the system against temperature:
plot(T,S,label="S")