module MCRun

using StaticArrays

export metropolis_condition, mc_step_atom!

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..EnergyEvaluation

ti = 2.
tf = 20.
n_traj = 30

mc_cycles = 1

max_displ = 0.2 # Angstrom


temp = TempGrid{n_traj}(ti,tf) # move to input file at a later stage ...

mc_params = MCParams(mc_cycles)
#mc_params = MCParams(mc_cycles;eq_percentage=0.2)

count_acc = zeros(n_traj)
count_acc_adj = zeros(n_traj)

displ_param = DisplacementParamsAtomMove(max_displ, temp.t_grid; update_stepsize=100)

println("displacement=",displ_param.max_displacement)

#histograms
Ebins=50
Emin=-0.02
Emax=-0.005
dE=(Emax-Emin)/Ebins
Ehistogram=Array{Array}(undef,n_traj) 
for i=1:n_traj
    Ehistogram[i]=zeros(Ebins)
end




"""
    metropolis_condition(energy_unmoved, energy_moved, beta)

Determines probability to accept a MC move at inverse temperature beta, takes energies of new and old configurations
"""
function metropolis_condition(energy_unmoved, energy_moved, beta)
    prob_val = exp(-(energy_moved-energy_unmoved)*beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

function mc_step_atom!(config, beta, dist2_mat, en_atom_mat, en_tot, i_atom, max_displacement, count_acc, count_acc_adj,Ehistogram,pot1)
    #displace atom, until it fulfills boundary conditions
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = move_atom!(config.pos[i_atom], delta_move)
    while check_boundary(config.bc,trial_pos)
        delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
        trial_pos = move_atom!(config.pos[i_atom], delta_move)
        println("O")
    end
    #find energy difference
    #println(i_atom)
    #println(delta_move)
    #println("trial position is ",trial_pos)
    #println("position is ",config.pos)
    dist2_new = [distance2(trial_pos,b) for b in config.pos]
    en_moved = dimer_energy_atom(i_atom, dist2_new, pot1) 
    #one might want to store dimer energies per atom in vector?
    #en_unmoved = dimer_energy_atom(i_atom, config.pos[i_atom], pot1)
    en_unmoved = dimer_energy_atom(i_atom, dist2_mat[i_atom, :], pot1)
    #decide acceptance
    #println(beta)
    #println(metropolis_condition(en_unmoved, en_moved, beta))
    #println("old total energy= ",en_tot)
    #println(dimer_energy_config(dist2_mat, 13, elj_ne)[2])
    if metropolis_condition(en_unmoved, en_moved, beta) >= rand()
        println("accepted")
        config.pos[i_atom] = trial_pos
        en_tot = en_tot - en_unmoved + en_moved
        dist2_mat[i_atom, :] = dist2_new
        dist2_mat[:, i_atom] = dist2_new
        count_acc += 1
        count_acc_adj += 1
    else
        println("rejected")
    end
    #println("new total energy= ",en_tot)
    #println(dimer_energy_config(dist2_mat, 13, elj_ne)[2])
    if en_tot>Emin && en_tot<Emax
        Ehistogram[Int(floor((en_tot-Emin)/dE))+1]+=1
    end
    #println("new position is ",config.pos)
    #println("count=",count_acc)
    #restore or accept
    return config, dist2_mat, en_tot, count_acc, count_acc_adj
end

println("{{{{{{{{{{{{{{{{{{{{{")
println(conf_ne13.bc,conf_ne13.pos)
println(check_boundary(conf_ne13.bc,conf_ne13.pos[1]))
println("}}}}}}}}}}}}}}}}}}}}}")




dist2_mat_0=get_distance2_mat(conf_ne13)
println(dist2_mat_0)

en_atom_mat_0=dimer_energy_config(dist2_mat_0, 13, elj_ne)[1]

en_tot_0=dimer_energy_config(dist2_mat_0, 13, elj_ne)[2]

en_1=dimer_energy_atom(2, dist2_mat_0[2, :], elj_ne)

println("^^^^^^^^^^^^^^^^^^^^^^^^^")
println("en_atom_mat_0=", en_atom_mat_0)
println("^^^^^^^^^^^^^^^^^^^^^^^^^")
println(en_tot_0)
println("^^^^^^^^^^^^^^^^^^^^^^^^^")
println(en_1)
println("en_1^^^^^^^^^^^^^^^^^^^^^^^^^")
dist2_new = [distance2(conf_ne13.pos[1],b) for b in conf_ne13.pos]
println(dist2_new)
println("^^^^^^^^^^^^^^^^^^^^^^^^^")

config=Array{Config}(undef,n_traj)
dist2_mat = Array{Matrix}(undef,n_traj) 
en_atom_mat = Array{Array}(undef,n_traj) 
en_tot = zeros(n_traj)
#max_displacement= zeros(n_traj)
max_displacement=displ_param.max_displacement
for i=1:n_traj
    config[i]=conf_ne13
    dist2_mat[i]=dist2_mat_0
    en_atom_mat[i]=en_atom_mat_0
    en_tot[i]=en_tot_0
    #max_displacement[i]=max_displ
end
println(max_displacement)




for i=1:mc_cycles
    for j=1:13
        for k=1:n_traj
            println(k)
            i_atom=rand(1:13)
            println(dimer_energy_config(dist2_mat[k], 13, elj_ne)[2])
            config[k], dist2_mat[k], en_tot[k], count_acc[k], count_acc_adj[k]=mc_step_atom!(config[k], temp.beta_grid[k], dist2_mat[k], en_atom_mat[k], en_tot[k], i_atom, displ_param.max_displacement[k], count_acc[k],count_acc_adj[k], Ehistogram[k], elj_ne)
            println(dimer_energy_config(dist2_mat[k], 13, elj_ne)[2])
            println()
        end
        if rem(i*13+j,100)==0
            update_max_stepsize!(displ_param, count_acc_adj, 1)
            for k=1:n_traj
                count_acc_adj[k]=0
            end
        end
    end
    println(i)
end

println(count_acc)
println(count_acc_adj)
println(displ_param.max_displacement)
for i=1:n_traj
    println(i)
    println(Ehistogram[i])
end



end