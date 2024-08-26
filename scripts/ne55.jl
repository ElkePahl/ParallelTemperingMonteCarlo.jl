using ParallelTemperingMonteCarlo
using Random

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

Random.seed!(1234)

# number of atoms
n_atoms = 55

# temperature grid
ti = 6.
tf = 20.
n_traj = 24

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 1000000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)


#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)


a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608]
b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751]
c1=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203]
potB = ELJPotentialB{6}(a,b,c1)
#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
ensemble = NVT(n_atoms)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne55 = [[-18.12272377969806, -2.8885121126311164, -1.0785986289370968], 
[-15.082336090495254, 8.461893729333122, -2.772516296352573], 
[-13.97479347892338, 3.3708602986270364, 7.744372171934367], 
[-10.705830054491752, -7.223554859294805, 6.885740644077007], 
[-12.316683712073452, 0.48997343801849313, -10.883233130266527], 
[-9.48582717041148, -9.41927729763756, -5.186712240003359], 
[-5.271574296449036, 10.312911450915326, 4.009531394290461], 
[-3.9245720730871403, 8.419425338948441, -8.298124854597294], 
[0.14347252002730343, -7.962244596738602, 1.5673173085128722], 
[-2.2416568351909856, 0.2182893291002686, 9.374005316894582], 
[-0.7932616472324334, -2.9766646014955653, -8.821102103446126], 
[3.23229791926128, 3.7045329918043213, -0.5392909934679209], 
[-16.782560878346867, 2.8505131527760144, -1.8420035713106138], 
[-14.45796426261625, -5.206921324082057, 2.9518906489820056], 
[-14.129865877391534, -5.9706638471499875, -3.842326021528425], 
[-15.692471050773447, -1.2353591654746434, -5.996699189948502], 
[-16.4226895330066, 0.5040039844927909, 3.3141708368955087], 
[-10.974869777160817, -4.671645764421954, -8.434980796104126], 
[-12.12315450106367, -1.8767402392696917, 7.4133511560585985], 
[-5.178874679434518, -8.64131935780649, -1.7047852256155787], 
[-14.583996551578903, 6.006820337022638, 2.822929195647825], 
[-8.471940502995933, 4.4574553409007915, -9.433064437927777], 
[-9.937832907351632, 7.220948943326906, 5.6651178270712155], 
[-8.009945930725, 2.2795246694040228, 8.658908104976408], 
[-5.80586677014992, -7.785115823363022, 4.027754790113613], 
[-9.640237465455519, 8.884727900168004, -5.191738610955646], 
[-6.24941303337707, -0.9957079773920928, -10.260332267544259], 
[-13.818788015022808, 4.5441773072625695, -6.846771003740379], 
[-5.159393050812876, -6.37342938570516, -7.113487881701231], 
[-10.574803948661117, -8.881201778937994, 0.6901167758162368], 
[-6.886656084288847, -3.3924394106446587, 8.543596613687987], 
[-10.326405022254868, 9.6444341142385, 0.7031366201913135], 
[1.2666189311375193, 0.14281204344724835, -4.515860068160085], 
[-5.010395370766757, 9.54400639104859, -2.0297340691201704], 
[-1.6654928577640074, -4.2212328415084235, 5.684783375958839], 
[0.24269529181928148, 1.7997374526095993, 4.299420910564888], 
[-0.9463371496307672, 6.792757019021627, 1.5207205212335662], 
[-0.8840969278216083, -5.3956823903527935, -3.6477793043095765], 
[-2.3900517129249508, 2.860802658434241, -8.286225056658665], 
[-3.405820737075789, 5.308842861814757, 6.396728325437435], 
[-0.43591977546303884, 6.605496962910492, -4.22693866769965], 
[1.3271801160752972, -2.19398965774533, 0.47941319893889417], 
[-7.688258287644187, 0.5710341723052527, -0.7390403000744364], 
[-12.850904470885903, -1.4903047326739016, -0.9792323138228062], 
[-11.448194714888132, 4.3354019635318295, -1.913486806552684], 
[-9.182830650401401, -3.6079010567729277, 2.7881248164509094], 
[-10.691821738385105, 2.102555484987307, 3.390909130926336], 
[-8.571070481953322, -4.519765508629754, -2.955532848563941], 
[-9.863756376073564, 0.1724190085133224, -5.788323874133968], 
[-2.364577645391471, 1.8567901981001682, -0.5399175052394061], 
[-4.109590537191895, -3.5903373425596303, 0.5346190102935704], 
[-5.7486276891518795, 4.355936402408257, -4.037147326564243], 
[-6.44037315748041, 5.355402110703938, 1.655822931025658], 
[-4.2518479464483505, -1.1555663973348989, -4.822239916008089], 
[-4.95442487926296, 0.05292679599723511, 4.3523193622717]]

#convert to Bohr
AtoBohr = 1.8897259886
#pos_ne55 = pos_ne55 * AtoBohr

#binding sphere
bc_ne55 = SphericalBC(radius=14*AtoBohr) 

length(pos_ne55) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne55, bc_ne55)


#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#


#to check code in REPL
Out = ptmc_run!(mc_params,temp,start_config,pot,ensemble;save=1000)
#Out = ptmc_run!(false;save=1000) 
#rm("checkpoint",recursive=true)
## 