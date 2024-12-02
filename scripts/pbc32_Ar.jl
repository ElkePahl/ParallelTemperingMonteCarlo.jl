using ParallelTemperingMonteCarlo
using Random
using Plots

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

# Random.seed!(1234)

# number of atoms
n_atoms = 32
pressure = 101325

# temperature grid
ti = 80
tf = 130
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 10000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)


save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Hmin"

#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

#c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
c=[-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]
pot = ELJPotentialEven{6}(c)


# link="/Users/tiantianyu/Downloads/look-up_table_he.txt"
# potlut=LookuptablePotential(link)

#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
separated_volume=false
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14, separated_volume)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom

#starting configurations
r_start = 3.7782 #r_start is the desired min. radius between atoms in the starting config.
L_start = 2*(r_start^2/2)^.5  #L_start refers to the distance between adjacent atoms which are parallel to the x or y axis
Cell_Repeats = cbrt(n_atoms/4) #Find how many times the unit cell has been repeated in one direction.
#isinteger(3.0)
if isinteger(Cell_Repeats) == false error("number of atoms not correct for fcc") end

#Generate fcc starting arrangement
#1st generate a simple cubic atoms
pos = Vector{Float64}[]
for i in 0:(Cell_Repeats-1)
  x = i * L_start
  for j in 0:(Cell_Repeats-1)
    y = j * L_start
    for k in 0:(Cell_Repeats-1)
      z = k * L_start
      pos_entry = [x,y,z]
      push!(pos, pos_entry)
    end
  end
end 

#2nd generate atoms that sit in the face of the unit cell, that sit in the same plane as simple cubic atoms
for i in 0:(Cell_Repeats-1)
  x = i * L_start + L_start/2
  for j in 0:(Cell_Repeats-1)
    y = j * L_start + L_start/2
    for k in 0:(Cell_Repeats-1)
      z = k * L_start
      pos_entry = [x,y,z]
      push!(pos, pos_entry)
    end
  end
end 

#3rd generate the rest of the atoms that sit in the face of the unit cell
for i in 0:(Cell_Repeats-1)
  x = i * L_start
  for j in 0:(Cell_Repeats-1)
    y = j * L_start + L_start/2
    for k in 0:(Cell_Repeats-1)
      z = k * L_start + L_start/2
      pos_entry = [x,y,z]
      push!(pos, pos_entry)
    end
  end
end 

for i in 0:(Cell_Repeats-1)
  x = i * L_start + L_start/2
  for j in 0:(Cell_Repeats-1)
    y = j * L_start
    for k in 0:(Cell_Repeats-1)
      z = k * L_start  + L_start/2
      pos_entry = [x,y,z]
      push!(pos, pos_entry)
    end
  end
end 

#center starting configation at the orgin
center = Vector{Float64}([Cell_Repeats * L_start/2, Cell_Repeats * L_start/2, Cell_Repeats * L_start/2])
for l in 1:n_atoms
  pos[l] = pos[l] - center
end

pos_ne32 = pos

#convert to Bohr
AtoBohr = 1.8897259886
pos_ne32 = pos_ne32 * AtoBohr

length(pos_ne32) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
box_length = Cell_Repeats * L_start * AtoBohr
bc_ne32 = CubicBC(box_length)   

#starting configuration
start_config_1 = Config(pos_ne32, bc_ne32)
start_config_2 = Config(pos_ne32, bc_ne32)
start_config = [start_config_1, start_config_2]

# #convert to Bohr
# AtoBohr = 1.8897259886
# #When the unit of distance is still Angstrom:
# #AtoBohr = 1.0
# pos_ne32 = pos_ne32 * AtoBohr

# #binding sphere
# box_length = 8.7674 * AtoBohr
# bc_ne32 = CubicBC(box_length)   

# length(pos_ne32) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne32, bc_ne32)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
mc_states, results = ptmc_run!(save_directory, mc_params,temp,start_config,pot,ensemble)

# temp_result, cp = multihistogram_NPT(ensemble, temp, results, 10^(-9), false)

filename = "all_rdfs.csv"
save_rdfs_concatenated(results.rdf, save_directory, filename)

println(temp.t_grid)
println(results.heat_cap)

data = [results.ev_histogram[i] for i in 1:n_traj]
filename = "all_histograms.csv"
save_multihistograms(data, save_directory, filename)

# max_value, index = findmax(cp)
# t_max = temp_result[index]
# println(t_max) 

# plot(temp.t_grid,results.heat_cap)

# data = [results.en_histogram[i] for i in 1:n_traj]
# plot(data)

#to check code in REPL
#@profview ptmc_run!(mc_params,temp,start_config,pot,ensemble)
#@benchmark ptmc_run!(mc_params,temp,start_config,pot,ensemble)



## 