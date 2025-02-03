using ParallelTemperingMonteCarlo
using Random
using Plots
using JLD2

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

Random.seed!(1234)

#convert to Bohr
AtoBohr = 1.8897259886

# number of atoms
n_atoms = 32
#pressure = 101325
scale = 500
pressure = scale * 101325

# temperature grid
ti = 15
tf = 60.
n_traj = 24

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 100000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles


#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 * AtoBohr # Angstrom
n_adjust = 100 # every n_adjust MC cycles, adjust the maximum displacement such that the acceptabce rate is between 40% and 60%

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)


#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

c=[0.51935577676634, −313.906518089779, 11045.2101976432, −88571.2026084401, 88684.2761441023, 946830.190858721]
#c=[-123.63510161951,21262.8963716972,-3239750.64086661,189367623.844691,-4304257347.72069,35314085074.72069]
pot = ELJPotentialEven{6}(c)


# link="/Users/tiantianyu/Downloads/look-up_table.txt"
# potlut=LookuptablePotential(link)
#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#y (Hartrees) and powers of length (Bohrs) based on the potential's exponent 


#-------------------------------------------------------------#
separated_volume=false
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14,separated_volume)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne32 =  [[ -4.3837,       -4.3837,       -4.3837],
  [-2.1918,       -2.1918,       -4.3837],
  [-2.1918,       -4.3837,       -2.1918],
  [-4.3837,       -2.1918,       -2.1918],
  [-4.3837,       -4.3837,        0.0000],
  [-2.1918,       -2.1918,        0.0000],
  [-2.1918,       -4.3837,        2.1918],
  [-4.3837,       -2.1918,        2.1918],
  [-4.3837,        0.0000,       -4.3837],
  [-2.1918,        2.1918,       -4.3837],
  [-2.1918,        0.0000,       -2.1918],
  [-4.3837,        2.1918,       -2.1918],
  [-4.3837,        0.0000,        0.0000],
  [-2.1918,        2.1918,        0.0000],
  [-2.1918,        0.0000,        2.1918],
  [-4.3837,        2.1918,        2.1918],
 [0.0000,       -4.3837,       -4.3837],
 [2.1918,       -2.1918,       -4.3837],
 [2.1918,       -4.3837,       -2.1918],
 [0.0000,       -2.1918,       -2.1918],
 [0.0000,       -4.3837,        0.0000],
 [2.1918,       -2.1918,        0.0000],
 [2.1918,       -4.3837,        2.1918],
 [0.0000,       -2.1918,        2.1918],
 [0.0000,        0.0000,       -4.3837],
 [2.1918,        2.1918,       -4.3837],
 [2.1918,        0.0000,       -2.1918],
 [0.0000,        2.1918,       -2.1918],
 [0.0000,        0.0000,        0.0000],
 [2.1918,        2.1918,       0.0000],
 [2.1918,        0.0000,        2.1918],
 [0.0000,        2.1918,        2.1918]]


#When the unit of distance is still Angstrom:
#AtoBohr = 1.0
pos_ne32 = pos_ne32 * AtoBohr

#binding sphere
box_length = 8.7674 * AtoBohr
bc_ne32 = CubicBC(box_length)   

length(pos_ne32) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config_1 = Config(pos_ne32, bc_ne32)
start_config_2 = Config(pos_ne32, bc_ne32)
start_config = [start_config_1,start_config_2]

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#

function do_experiment(mc_params,temp,start_config,pot,ensemble,scale)
  println("Start PTMC simulation with scale $scale")
  mc_states, results = ptmc_run!(mc_params,temp,start_config,pot,ensemble)

  println(results)
  #to check code in REPL
  #@profview ptmc_run!(mc_params,temp,start_config,pot,ensemble)
  #@benchmark ptmc_run!(mc_params,temp,start_config,pot,ensemble)
  plot(temp.t_grid,results.heat_cap)
  savefig("ne32_raw_heat_cap_5a.png")
  data = [results.ev_histogram[i] for i in 1:n_traj]

  plot(data)
  savefig("ne32_raw_ev_histogram_ $scale a.png")
  cp = multihistogram_NPT(ensemble, temp, results, 10^(-3), false)

  @save "cp_results_he32_ $scale a.jld2" cp

  
end

scale_val = 1
ti = 5
tf = 25
n_traj = 24


temp = TempGrid{n_traj}(ti,tf) 
println("Scale: $scale_val")
pressure = scale_val * 101325
separated_volume=false
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14,separated_volume)
move_strat = MoveStrategy(ensemble)



println("Start PTMC simulation")
mc_states, results = ptmc_run!(mc_params, temp, start_config, pot, ensemble)
for i in 1:n_traj
  println("Temperature $(temp.t_grid[i]) => Box length: $(mc_states[i].config.bc.box_length)")
end
plot(temp.t_grid, results.heat_cap)
println(mc_states[1].config.pos)
# savefig("ne32_raw_heat_cap_$scale_val a.png")
data = [results.ev_histogram[i] for i in 1:n_traj]
plot(data)
# savefig("ne32_raw_ev_histogram_$scale_val a.png")
# cp, temp_result = multihistogram_NPT(ensemble, temp, results, 10^(-3), false)
using JLD2
# @save "cp_results_he32_$scale_val a.jld2" cp
# @save "temp_results_he32_$scale_val a.jld2" temp_result
# plot([temp_result], [cp], xlabel="Temperature (K)", ylabel="Heat Capacity (Cₚ)", title="Heat Capacity vs Temperature", legend=false)

# savefig("heat_capacity_vs_temperature_$scale_val a.png")

println(results.rdf[1])
plot(results.rdf[1])
savefig("he32_rdf_$scale_val a.png")
# @load "cp_results_he32_atm.jld2" cp
index = argmax(results.rdf[1])
println("The first peak of the RDF is at $(results.rdf[1][index]) at $index")

equilibrium_distance = sqrt(index * results.delta_r2)
open("equilibrium_data.txt", "a") do file
  println(file, "Scale: ", scale_val, " Equilibrium distance: ", equilibrium_distance)
end

# @load "cp_results_he32_512000 a.jld2" cp
# for scale in 10:5:45
#   do_experiment(mc_params, temp, start_config, pot, ensemble, scale)
# end
