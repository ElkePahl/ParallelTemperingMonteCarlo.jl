using ParallelTemperingMonteCarlo
using BenchmarkTools

##

n_atoms = 13

# temperature grid
ti = 5.
tf = 16.
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details
mc_cycles = 10000 #default 20% equilibration cycles on top
mc_sample = 1  #sample every mc_sample MC cycles
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

#moves - allowed at present: atom, volume and rotation moves (volume,rotation not yet implemented)
move_strat = MoveStrategy(atom_moves = n_atoms)  

#ensemble
ensemble = NVT(n_atoms)


c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)

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
bc_ne13 = SphericalBC(radius=5.32*AtoBohr)   #5.32 
start_config = Config(pos_ne13, bc_ne13)

n_bin = 100

mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot; max_displ=[max_displ_atom[i],0.01,1.]) for i in 1:n_traj]
##
function booltest(variable::Bool)
    if variable  == true
        return 4
    else
        return 3
    end
end
##
function sampling_step(mc_params,mc_states,i, saveham::Bool)
    if saveham == true
        if rem(i, mc_params.mc_sample) == 0
            for i_traj=1:mc_params.n_traj
            push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
            end
        end 
    end
    #step adjustment
    if rem(i, mc_params.n_adjust) == 0
        for i_traj = 1:mc_params.n_traj
            update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
        end 
    end
end
function sampling_step(mc_params,mc_states,i)
    # if saveham == true
        if rem(i, mc_params.mc_sample) == 0
            for i_traj=1:mc_params.n_traj
            push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
            end
        end 
    #end
    #step adjustment
    if rem(i, mc_params.n_adjust) == 0
        for i_traj = 1:mc_params.n_traj
            update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
        end 
    end
end
##
time1 = @benchmark sampling_step(mc_params,mc_states,1,false)

time2 = @benchmark sampling_step(mc_params,mc_states,1,true)

time3 = @benchmark sampling_step(mc_params,mc_states,1)
println(time1,time2,time3)
