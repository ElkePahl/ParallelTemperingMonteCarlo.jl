using ParallelTemperingMonteCarlo
using Random

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

Random.seed!(1234)

# number of atoms
n_atoms = 32
pressure = 101325 * 10000

# temperature grid
ti = 300.
tf = 500.
n_traj = 100

temp = TempGrid{n_traj}(ti,tf) 
mc_cycles = 10000 #default 20% equilibration cycles on top



mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 1.0 # Angstrom
max_vchange = 0.02
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

# MC simulation details


#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

c=[-123.63510161951,21262.8963716972,-3239750.64086661,189367623.844691,-4304257347.72069,35314085074.72069]

pot = ELJPotentialEven{6}(c)

#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14)
move_strat = MoveStrategy(atom_moves = n_atoms, vol_moves = 1)  

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

#convert to Bohr
AtoBohr = 1.8897259886 * 1.21
pos_ne32 = pos_ne32 * AtoBohr

#binding sphere
box_length = 8.7674 * AtoBohr
bc_ne32 = CubicBC(box_length)   

length(pos_ne32) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne32, bc_ne32)
potential=pot


mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot; max_displ=[max_displ_atom[i],0.1,0.1]) for i in 1:n_traj]
mc_states_scaled = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot; max_displ=[max_displ_atom[i],0.1,0.1]) for i in 1:n_traj]


for i=1:25
    scale = 1.1-0.02*i
    pos_scaled = pos_ne32 * scale
    box_length_scaled = box_length * scale
    config_scaled = Config(pos_scaled, CubicBC(box_length_scaled))
    mc_states_scaled[i] = MCState(temp.t_grid[i], temp.beta_grid[i],config_scaled, pot; max_displ=[max_displ_atom[i],0.1,0.1])
end



function metropolis_condition(ensemble::Etype, delta_energy::Float64,volume_changed::Float64,volume_unchanged::Float64,beta::Float64) where Etype <: NPT
    delta_h = delta_energy + ensemble.pressure*(volume_changed-volume_unchanged)
    println("de = ",delta_energy)
    println("dh = ",delta_h)
    prob_val = exp(-delta_h*beta + ensemble.n_atoms*log(volume_changed/volume_unchanged))
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end


for i=1:25
    println("scale = ",1.1-0.02*i)
    println("box_length = ",mc_states_scaled[i].config.bc.box_length)
    de=mc_states_scaled[i].en_tot-mc_states[i].en_tot
    v_unc= mc_states[i].config.bc.box_length^3
    v_c=mc_states_scaled[i].config.bc.box_length^3
    println(metropolis_condition(ensemble, de, v_c, v_unc, temp.beta_grid[i]))
    println()
end

trial_configs=[]

for i=1:100
    trial_config = volume_change(mc_states[1].config, mc_states[1].config.bc, mc_states[1].max_displ[2])
    println(trial_config.bc.box_length/mc_states[1].config.bc.box_length)
    #println(trial_config.pos[1][1]/mc_states[1].config.pos[1][1])
    #println(trial_config)
    push!(trial_configs, trial_config)
end

en_tot=get_energy(trial_configs,mc_states,pot)[3]

for i=1:100
    println(en_tot[i])
end