using ParallelTemperingMonteCarlo

n_atoms = 13;

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 
3918846.12841668, -56234083.4334278, 288738837.441765]

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
[0.000002325340981,	0.000000762100600, 0.000000414930733]];

AtoBohr = 1.8897259886;

pos_ne13 = pos_ne13 * AtoBohr

bc_ne13 = SphericalBC(radius=5.32*AtoBohr) 

start_config = Config(pos_ne13, bc_ne13)

ti = 4.;
tf = 16.;
n_traj = 25;
temp = TempGrid{n_traj}(ti,tf)

mc_cycles = 100_000;
mc_sample = 1;
displ_atom = 0.1;
max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj];
n_adjust = 100;

save_configuration = true
save_frequency = 20_000
file_name = "Configurations"

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample=mc_sample, n_adjust=n_adjust)

ensemble = NVT(n_atoms);
move_strat = MoveStrategy(ensemble)

mc_states, results = ptmc_run!(mc_params,temp,start_config,pot,ensemble;save=1000,
saveconfigs=save_frequency, configsname=file_name);

energies,histogramdata,T,Z,Cv,dCv,S = postprocess();
for temperature in 1:25
    open("./checkpoint/Configurations100000T$temperature.xyz", "r") do file
        readline(file) # skip first line
        readline(file) # skip second line
        atom_count = 1 # tracks which atom index we are up to
        for line in eachline(file)
            line = line[4:end] # ignore atom label at start of line
            test = string(join(mc_states[temperature].config.pos[atom_count], " "), " ")
            # This puts the data from the simulation into the same format as that in storage
            if test ≠ line
                error("Test fail, saved data is not equal to stored data")
            end
            atom_count+=1
        end
    end
end