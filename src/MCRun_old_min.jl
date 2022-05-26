module MCRun

using StaticArrays
using DataFrames
using Plots
using Dates
using BenchmarkTools
using Optim

export metropolis_condition, mc_step_atom!

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..EnergyEvaluation

println(now())

NAtoms=13


ti = 8.
tf = 20.
n_traj = 30

equilibrium = 1000
mc_cycles = 5000

max_displ = 0.5 # Angstrom


temp = TempGrid{n_traj}(ti,tf) # move to input file at a later stage ...

#println("temperatures= ")
println(temp)
#println()

mc_params = MCParams(mc_cycles)
#mc_params = MCParams(mc_cycles;eq_percentage=0.2)

count_acc = zeros(n_traj)        #total count of acceptance
count_acc_adj = zeros(n_traj)    #acceptance used for stepsize adjustment, will be reset to 0 after each adjustment
count_exc = zeros(n_traj)        #number of proposed exchanges
count_exc_acc = zeros(n_traj)    #number of accepted exchanges

displ_param = DisplacementParamsAtomMove(max_displ, temp.t_grid; update_stepsize=100)

println("displacement=")
println(displ_param.max_displacement)
#println()


block=5         #blocking

#histograms
Ebins=50
Emin=-0.02
Emax=-0.00
dE=(Emax-Emin)/Ebins
Ehistogram=Array{Array}(undef,n_traj)      #initialization
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

function mc_step_atom!(config, beta, dist2_mat, en_atom_mat, en_tot, i_atom, max_displacement, count_acc, count_acc_adj,pot1)
    #displace atom, until it fulfills boundary conditions
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = move_atom!(config.pos[i_atom], delta_move)
    while check_boundary(config.bc,trial_pos)         #displace the atom until it's inside the sphere
        delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
        trial_pos = move_atom!(config.pos[i_atom], delta_move)
    end
    #find energy difference
    dist2_new = [distance2(trial_pos,b) for b in config.pos]
    en_moved = dimer_energy_atom(i_atom, dist2_new, pot1) 
    #one might want to store dimer energies per atom in vector?
    en_unmoved = dimer_energy_atom(i_atom, dist2_mat[i_atom, :], pot1)
    #decide acceptance
    if metropolis_condition(en_unmoved, en_moved, beta) >= rand()
        config.pos[i_atom] = copy(trial_pos)          #update position of the atom moved
        en_tot = en_tot - en_unmoved + en_moved       #update total energy
        dist2_mat[i_atom, :] = copy(dist2_new)        #distance matrix
        #dist2_mat[:, i_atom] = dist2_mat[i_atom, :]
        dist2_mat[:, i_atom] = copy(dist2_new)        #same                                
        count_acc_adj += 1                            #for adjustement
        count_acc += 1
    end
    return config, dist2_mat, en_tot, count_acc, count_acc_adj
end


function mc_step_atom_eq!(config, beta, dist2_mat, en_atom_mat, en_tot, i_atom, max_displacement, count_acc_adj, pot1)
    #displace atom, until it fulfills boundary conditions
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = move_atom!(config.pos[i_atom], delta_move)
    while check_boundary(config.bc,trial_pos)         #displace the atom until it's inside the sphere
        delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
        trial_pos = move_atom!(config.pos[i_atom], delta_move)
    end
    #find energy difference
    dist2_new = [distance2(trial_pos,b) for b in config.pos]
    en_moved = dimer_energy_atom(i_atom, dist2_new, pot1) 
    #one might want to store dimer energies per atom in vector?
    en_unmoved = dimer_energy_atom(i_atom, dist2_mat[i_atom, :], pot1)
    #decide acceptance
    if metropolis_condition(en_unmoved, en_moved, beta) >= rand()
        config.pos[i_atom] = copy(trial_pos)          #update position of the atom moved
        en_tot = en_tot - en_unmoved + en_moved       #update total energy
        dist2_mat[i_atom, :] = copy(dist2_new)        #distance matrix
        dist2_mat[:, i_atom] = copy(dist2_new)        #same                                
        count_acc_adj += 1                            #for adjustement
    end
    return config, dist2_mat, en_tot, count_acc_adj
end

function mc_step_atom_2!(config, beta, dist2_mat, en_atom_mat, en_tot, i_atom, max_displacement, count_acc, count_acc_adj, eq, Ehistogram, pot1)
    #displace atom, until it fulfills boundary conditions
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = move_atom!(config.pos[i_atom], delta_move)
    while check_boundary(config.bc,trial_pos)         #displace the atom until it's inside the sphere
        delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
        trial_pos = move_atom!(config.pos[i_atom], delta_move)
    end
    #find energy difference
    dist2_new = [distance2(trial_pos,b) for b in config.pos]
    en_moved = dimer_energy_atom(i_atom, dist2_new, pot1) 
    #one might want to store dimer energies per atom in vector?
    en_unmoved = dimer_energy_atom(i_atom, dist2_mat[i_atom, :], pot1)
    #decide acceptance
    if metropolis_condition(en_unmoved, en_moved, beta) >= rand()
        config.pos[i_atom] = copy(trial_pos)          #update position of the atom moved
        en_tot = en_tot - en_unmoved + en_moved       #update total energy
        dist2_mat[i_atom, :] = copy(dist2_new)        #distance matrix
        #dist2_mat[:, i_atom] = dist2_mat[i_atom, :]
        dist2_mat[:, i_atom] = copy(dist2_new)        #same                                
        count_acc_adj += 1 
        if eq==0                           #for adjustement
            count_acc += 1
        end
    end
    if en_tot>Emin && en_tot<Emax && eq==0                   #store energy in histogram
        Ehistogram[Int(floor((en_tot-Emin)/dE))+1]+=1
    end
    return config, dist2_mat, en_tot, count_acc, count_acc_adj
end

function trajectory_exchange!(config1, config2, dist2_mat1, dist2_mat2, en_atom_mat1, en_atom_mat2, en_tot1, en_tot2, eq, count_exc_acc1, count_exc_acc2)
    config1, config2 = config2, config1                       #exchange configurations
    dist2_mat1, dist2_mat2 = dist2_mat2, dist2_mat1           #distance matrices
    en_atom_mat1, en_atom_mat2 = en_atom_mat2, en_atom_mat1   #energy matrices
    en_tot1, en_tot2 = en_tot2, en_tot1                       #total energies
    if eq==0
        count_exc_acc1+=1                                     #number of acceptance for both trajectories
        count_exc_acc2+=1
    end
    return config1, config2, dist2_mat1, dist2_mat2, en_atom_mat1, en_atom_mat2, en_tot1, en_tot2, count_exc_acc1, count_exc_acc2
end

function total_energy_x(x)
    energy_tot=0
    for i = 1:NAtoms-1
        for j = i+1:NAtoms
            energy_tot += dimer_energy(elj_ne, ((x[3*j-2]-x[3*i-2])^2+(x[3*j-1]-x[3*i-1])^2+(x[3*j]-x[3*i])^2))
        end
    end
    return energy_tot
end


#println("initial configuration:")
#println(conf_ne13.bc,conf_ne13.pos)
#println(check_boundary(conf_ne13.bc,conf_ne13.pos[1]))
#println()

dist2_mat_0=get_distance2_mat(conf_ne13)
#println("initial distance matrix: ", dist2_mat_0)

en_atom_mat_0=dimer_energy_config(dist2_mat_0, NAtoms, elj_ne)[1]
en_tot_0=dimer_energy_config(dist2_mat_0, NAtoms, elj_ne)[3]

#println("initial total energy: ", en_tot_0)
#println()



config=Array{Config}(undef,n_traj)
dist2_mat = Array{Matrix}(undef,n_traj) 
en_atom_mat = Array{Array}(undef,n_traj) 
en_tot = zeros(n_traj)
max_displacement=displ_param.max_displacement
for i=1:n_traj
    config[i]=Config(copy(conf_ne13.pos),bc_ne13)
    dist2_mat[i]=copy(dist2_mat_0)
    en_atom_mat[i]=copy(en_atom_mat_0)
    en_tot[i]=en_tot_0
end

function minimization(config, en_tot)
    x=zeros(3*NAtoms)
    for i=1:NAtoms
        x[3i-2]=config.pos[i][1]
        x[3i-1]=config.pos[i][2]
        x[3i]=config.pos[i][3]
    end
    #println(x)
    local od
    od=OnceDifferentiable(total_energy_x, x; autodiff=:forward);
    println("enenrgy before minimization is ", en_tot)
    result = Optim.minimizer(optimize(od, x, ConjugateGradient(),Optim.Options(g_tol=1e-6)))
    e_end=total_energy_x(result)
    println("enenrgy after minimization is ", e_end)
    println()
end




#energies=Array{DataFrame}(undef,n_traj)                #energies for heat capacity calculation
#for i=1:n_traj
    #energies[i]=DataFrame(A=Float64[])
#end

energies=Array{Array}(undef,n_traj)
for i=1:n_traj
    energies[i]=zeros(mc_cycles)
end

cv=Array{Float64}(undef,n_traj)



for i=1:equilibrium
    for j=1:n_traj                   #number of atoms
        for k=1:NAtoms            #trajetcories
            i_atom=rand(1:NAtoms)
            config[j], dist2_mat[j], en_tot[j], count_acc_adj[j]=mc_step_atom_eq!(config[j], temp.beta_grid[j], dist2_mat[j], en_atom_mat[j], en_tot[j], i_atom, displ_param.max_displacement[j], count_acc_adj[j], elj_ne)
        end
    end
    if rem(i,100)==0              #stepsize adjustment
        update_max_stepsize!(displ_param, count_acc_adj, NAtoms)
        for k=1:n_traj
            count_acc_adj[k]=0
        end
    end
    c=rand()                      #exchange
    if c<0.1
        #ex=Int(1+floor((n_traj-1)*rand()))
        ex=rand(1:n_traj-1)
        #count_exc[ex]+=1
        #count_exc[ex+1]+=1
        delta_beta=1/(kB*temp.t_grid[ex])-1/(kB*temp.t_grid[ex+1])
        delta_energy=en_tot[ex]-en_tot[ex+1]
        exc_acc=min(1.0,exp(delta_beta*delta_energy))
        #println(ex, " , ", exc_acc)
        if exc_acc>rand()
            config[ex], config[ex+1], dist2_mat[ex], dist2_mat[ex+1], en_atom_mat[ex], en_atom_mat[ex+1], en_tot[ex], en_tot[ex+1], count_exc_acc[ex], count_exc_acc[ex+1] = trajectory_exchange!(config[ex], config[ex+1], dist2_mat[ex], dist2_mat[ex+1], en_atom_mat[ex], en_atom_mat[ex+1], en_tot[ex], en_tot[ex+1], 1, count_exc_acc[ex], count_exc_acc[ex+1])
        end
    end
end

#println(en_tot)

for i=1:mc_cycles
    for j=1:n_traj                    #number of atoms
        for k=1:NAtoms            #trajetcories
            i_atom=rand(1:NAtoms)
            config[j], dist2_mat[j], en_tot[j], count_acc[j], count_acc_adj[j]=mc_step_atom!(config[j], temp.beta_grid[j], dist2_mat[j], en_atom_mat[j], en_tot[j], i_atom, displ_param.max_displacement[j], count_acc[j],count_acc_adj[j], elj_ne)
        end
        if en_tot[j]>Emin && en_tot[j]<Emax                   #store energy in histogram
            Ehistogram[j][Int(floor((en_tot[j]-Emin)/dE))+1]+=1
        end
    end
    if rem(i,100)==0              #stepsize adjustment
        update_max_stepsize!(displ_param, count_acc_adj, NAtoms)
        for k=1:n_traj
            count_acc_adj[k]=0
        end
        minimization(config[15],en_tot[15])
    end
    c=rand()                      #exchange
    if c<0.1
        ex=rand(1:n_traj-1)
        count_exc[ex]+=1
        count_exc[ex+1]+=1
        delta_beta=1/(kB*temp.t_grid[ex])-1/(kB*temp.t_grid[ex+1])
        delta_energy=en_tot[ex]-en_tot[ex+1]
        exc_acc=min(1.0,exp(delta_beta*delta_energy))
        #println(ex, " , ", exc_acc)
        if exc_acc>rand()
            config[ex], config[ex+1], dist2_mat[ex], dist2_mat[ex+1], en_atom_mat[ex], en_atom_mat[ex+1], en_tot[ex], en_tot[ex+1], count_exc_acc[ex], count_exc_acc[ex+1] = trajectory_exchange!(config[ex], config[ex+1], dist2_mat[ex], dist2_mat[ex+1], en_atom_mat[ex], en_atom_mat[ex+1], en_tot[ex], en_tot[ex+1], 0, count_exc_acc[ex], count_exc_acc[ex+1])
        end
    end

    for j=1:n_traj                #energy and energy^2
        #push!(energies[j],[en_tot[j]])
        energies[j][i]=en_tot[j]
    end
    #println(i)
end

println(now())

println("displacement acceptance: ",count_acc)
println()
println("maximum displacement: ",displ_param.max_displacement)
println()
println("total exchange: ",count_exc)
println()
println("accepted exchange: ",count_exc_acc)
println()
println("Histogram:")
for i=1:n_traj
    println(i)
    println(Ehistogram[i])
end
println()


e_avg=zeros(n_traj)
e2_avg=zeros(n_traj)
for i=1:n_traj
    for j=1:mc_cycles
        if rem(j,block)==0
            #e_avg[i]+=energies[i].A[j]/floor(mc_cycles/block)
            #e2_avg[i]+=(energies[i].A[j])^2/floor(mc_cycles/block)
            #println(energies[i][j])
            e_avg[i]+=energies[i][j]/floor(mc_cycles/block)
            e2_avg[i]+=(energies[i][j])^2/floor(mc_cycles/block)
        end
    end
    #println(e_avg[i])
    #println(e2_avg[i])
    cv[i]=(e2_avg[i]-e_avg[i]^2)/(kB*temp.t_grid[i])
end

println("Heat Capacity: ", cv)

plot(temp.t_grid,cv)

println(now())


for i=1:10
    @btime mc_step_atom!(config[2], temp.beta_grid[2], dist2_mat[2], en_atom_mat[2], en_tot[2], 2, displ_param.max_displacement[2], count_acc[2],count_acc_adj[2], elj_ne)
end
#for i=1:10
    #@btime mc_step_atom_eq!(config[12], temp.beta_grid[12], dist2_mat[12], en_atom_mat[12], en_tot[12], 12, displ_param.max_displacement[12],count_acc_adj[12], elj_ne)
#end
#for i=1:10
    #@btime mc_step_atom_2!(config[2], temp.beta_grid[2], dist2_mat[2], en_atom_mat[2], en_tot[2], 2, displ_param.max_displacement[2], count_acc[2],count_acc_adj[2], 1, Ehistogram[2], elj_ne)
#end
#for i=1:10
    #@btime mc_step_atom_2!(config[2], temp.beta_grid[2], dist2_mat[2], en_atom_mat[2], en_tot[2], 2, displ_param.max_displacement[2], count_acc[2],count_acc_adj[2], 0, Ehistogram[2], elj_ne)
#end
#println()
#for i=1:10
    #@btime trajectory_exchange!(config[1], config[2], dist2_mat[1], dist2_mat[2], en_atom_mat[1], en_atom_mat[2], en_tot[1], en_tot[2], 1, count_exc_acc[1], count_exc_acc[2])
#end
#println()
#for i=1:10
    #@btime update_max_stepsize!(displ_param, count_acc_adj, 13)
#end
#println()
#for i=1:10
    #@btime push!(energies[1],[en_tot[1] en_tot[1]^2]) 
    #@btime energies[1][1]=en_tot[1]
#end 
#println()
for i=1:10
    @btime dimer_energy_atom(1, dist2_mat[1, :], elj_ne) 
end
#println()
#pos_1=copy(config[1].pos[1])
#@btime copy(config[1].pos[1])
#for i=1:10
    #@btime move_atom!(pos_1,[1,1,1])
#end

end