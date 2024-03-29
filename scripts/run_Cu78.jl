
using ParallelTemperingMonteCarlo#potential_variable_struct


using ParallelTemperingMonteCarlo#potential_variable_struct

using Random,DelimitedFiles

#cd("$(pwd())/scripts")
#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms

n_atoms = 78


# temperature grid

ti = 700
tf = 1100


n_traj = 20


temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details


mc_cycles = 10000 #default 20% equilibration cycles on top



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




#starting configurations
#icosahedral ground state of Cu55 (from Cambridge cluster database) converted to angstrom

pos_cu78 = [[9.20810577, 10.6811792, 12.85775222],
[7.72518083, 8.64747448, 13.47423093],
[8.83017234, 6.48185841, 14.26401845],
[5.76926904, 8.5204058, 11.78510919],
[10.3058934, 8.49163418, 13.72488251],
[11.0445983, 6.10374679, 13.03249449],
[10.76908039, 11.68540658, 11.04398002],
[6.86491181, 6.33673147, 12.63535963],
[11.14520009, 4.41652378, 11.06840909],
[10.56131156, 9.20976286, 11.36608831],
[3.8350787, 9.66043706, 10.62564335],
[8.18461754, 11.86457693, 10.78680942],
[12.53103163, 8.13651712, 12.41914688],
[9.10050302, 7.21655723, 11.96507541],
[8.9196861, 4.77149686, 12.37507079],
[8.03726493, 9.35893297, 11.12552957],
[11.75404842, 10.48292508, 13.05807496],
[7.19140363, 7.0877584, 10.30045396],
[6.63954366, 10.80307617, 12.52504858],
[13.20475821, 5.78540789, 11.69596475],
[10.04929714, 10.21021383, 2.74162567],
[8.60808677, 8.21325894, 3.29147489],
[11.13551339, 8.08991071, 3.51785384],
[7.18684228, 6.21910921, 3.93796442],
[7.88437782, 10.5718392, 3.95937956],
[9.71784555, 6.07918378, 4.09394596],
[11.97420313, 10.35137104, 4.33933113],
[6.42017871, 8.55796232, 4.53249281],
[9.97169751, 11.88885922, 4.62524758],
[9.82411407, 9.43691954, 5.10406749],
[5.02772356, 6.55318086, 5.20359556],
[5.71367449, 10.90921526, 5.25590498],
[8.41435724, 7.42843168, 5.68181186],
[11.0285349, 7.27653846, 5.85794355],
[7.79544252, 12.26033475, 5.87610665],
[4.27718147, 8.88880172, 5.86736713],
[11.91867483, 12.00967502, 6.24563146],
[6.9561607, 5.37238684, 6.26938662],
[7.67700429, 9.79132879, 6.32467407],
[9.50228326, 5.2069596, 6.45258385],
[11.87164769, 9.54494266, 6.68244807],
[12.02789873, 5.12581568, 6.74109416],
[9.88844903, 13.49922651, 6.57922127],
[3.59896936, 11.21261099, 6.58760866],
[6.25574475, 7.76131662, 6.94310978],
[9.77298468, 11.08697928, 7.00896014],
[5.64304702, 12.56406942, 7.20157052],
[9.65009283, 8.66311539, 7.41738673],
[12.95719784, 7.37463975, 7.57489662],
[3.97662306, 6.96385649, 7.4780558],
[5.5593229, 10.10255239, 7.6386882],
[7.73820851, 13.84622135, 7.8527848],
[8.1830049, 6.63840522, 8.0168064],
[10.70190431, 6.53159817, 8.22675048],
[13.72586678, 9.68714235, 8.3981167],
[7.98154103, 4.19765578, 8.32731731],
[3.27977734, 9.3022518, 8.18896381],
[7.62359206, 11.43808642, 8.27752581],
[10.55129337, 4.0805155, 8.5609616],
[11.76414234, 11.28499926, 8.65854603],
[5.91012759, 5.75557787, 8.56871289],
[7.49034308, 9.0057639, 8.68353281],
[11.54936709, 8.80691453, 9.05320905],
[9.7137393, 12.78636073, 8.95977573],
[12.67456914, 5.40926942, 9.22361494],
[4.12936341, 11.58348871, 9.01710543],
[5.22409251, 8.16647139, 9.25872611],
[9.56094879, 10.34182511, 9.36067818],
[9.37880896, 7.93821131, 9.68957149],
[6.19508081, 12.92482629, 9.64387612],
[13.5372612, 7.72382715, 10.06363071],
[4.95459032, 6.27620264, 10.88163222],
[9.20140742, 5.5571462, 10.04127471],
[6.0873292, 10.48531258, 10.10256552],
[6.96384327, 4.65312172, 10.66958722],
[9.02696245, 3.15942277, 10.40158812],
[11.28333689, 6.86978501, 10.68020117],
[12.79744324, 10.11163506, 10.75172547]]

#convert to Bohr

nmtobohr = 18.8973



# copperconstant = 0.36258*nmtobohr
# pos_cu55 = copperconstant*ico_55
AtoBohr = 1.8897259886
pos_cu78 .*= AtoBohr 
 
 # we do not have a centred config, correcting this now
 
 cofm = [sum([pos[1] for pos in pos_cu78]),sum([pos[2] for pos in pos_cu78]),sum([pos[3] for pos in pos_cu78])]./78

for element in pos_cu78
    element .-=cofm 
end



length(pos_cu78) == n_atoms || error("number of atoms and positions not the same - check starting config")


#boundary conditions 
bc_cu55 = SphericalBC(radius=14*AtoBohr)   #5.32 Angstrom

#starting configuration

start_config = Config(pos_cu78, bc_cu55)

#histogram information
n_bin = 100

#----------------------------------------------------------------------------#
evtohartree = 0.0367493
#parameters taken from L Vocadlo etal J Chem Phys V120N6 2004
n = 8.482
m = 4.692
ϵ = evtohartree*0.0370
a = 0.25*nmtobohr
C = 27.561

pot = EmbeddedAtomPotential(n,m,ϵ,C,a)
# sutton chen potential as used by Doye et al for the cambridge cluster database energy landscape

suttonchenpot = EmbeddedAtomPotential(9.0,6.0,0.0126*evtohartree,39.432,0.3612*nmtobohr)
#----------------------------------------------------------------------------#

#-------------------------------------------#
#--------Vector of radial symm values-------#
#-------------------------------------------#
X = [ 1    1              0.001   0.000  11.338
 1    0              0.001   0.000  11.338
 1    1              0.020   0.000  11.338
 1    0              0.020   0.000  11.338
 1    1              0.035   0.000  11.338
 1    0              0.035   0.000  11.338
 1    1              0.100   0.000  11.338
 1    0              0.100   0.000  11.338
 1    1              0.400   0.000  11.338
 1    0              0.400   0.000  11.338]
radsymmvec = []


#--------------------------------------------#
#--------Vector of angular symm values-------#
#--------------------------------------------#
V = [[0.0001,1,1,11.338],[0.0001,-1,2,11.338],[0.003,-1,1,11.338],[0.003,-1,2,11.338],[0.008,-1,1,11.338],[0.008,-1,2,11.338],[0.008,1,2,11.338],[0.015,1,1,11.338],[0.015,-1,2,11.338],[0.015,-1,4,11.338],[0.015,-1,16,11.338],[0.025,-1,1,11.338],[0.025,1,1,11.338],[0.025,1,2,11.338],[0.025,-1,4,11.338],[0.025,-1,16,11.338],[0.025,1,16,11.338],[0.045,1,1,11.338],[0.045,-1,2,11.338],[0.045,-1,4,11.338],[0.045,1,4,11.338],[0.045,1,16,11.338],[0.08,1,1,11.338],[0.08,-1,2,11.338],[0.08,-1,4,11.338],[0.08,1,4,11.338]]

T = [[1.,1.,1.],[1.,1.,0.],[1.,0.,0.]]

angularsymmvec = []
#-------------------------------------------#
#-----------Including scaling data----------#
#-------------------------------------------#
file = open("$(pwd())/scaling.data")
scalingvalues = readdlm(file)
close(file)
G_value_vec = []
for row in eachrow(scalingvalues[1:88,:])
    max_min = [row[4],row[3]]
    push!(G_value_vec,max_min)
end


for symmindex in eachindex(eachrow(X))
    row = X[symmindex,:]
    radsymm = RadialType2{Float64}(row[3],row[5],[row[1],row[2]],G_value_vec[symmindex])
    push!(radsymmvec,radsymm)
end


let n_index = 10

for element in V
    for types in T 

        n_index += 1

        symmfunc = AngularType3{Float64}(element[1],element[2],element[3],11.338,types,G_value_vec[n_index])

        push!(angularsymmvec,symmfunc)
    end
end
end
#---------------------------------------------------#
#------concatenating radial and angular values------#
#---------------------------------------------------#

totalsymmvec = vcat(radsymmvec,angularsymmvec)


#--------------------------------------------------#
#-----------Initialising the nnp weights-----------#
#--------------------------------------------------#
num_nodes::Vector{Int32} = [88, 20, 20, 1]
activation_functions::Vector{Int32} = [1, 2, 2, 1]
file = open("weights.029.data","r+")
weights=readdlm(file)
close(file)
weights = vec(weights)
nnp = NeuralNetworkPotential(num_nodes,activation_functions,weights)

runnerpotential = RuNNerPotential(nnp,totalsymmvec)
#------------------------------------------------------------#
#============================================================#
#------------------------------------------------------------#


mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot) for i in 1:n_traj]



#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

@time ptmc_run!((mc_states, move_strat, mc_params, pot, ensemble, results));

