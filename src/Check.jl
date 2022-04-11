module Check

using StaticArrays

using Reexport

include("BoundaryConditions.jl")
include("Configurations.jl")
include("EnergyEvaluation.jl")
include("InputParams.jl")

@reexport using .InputParams
@reexport using .BoundaryConditions
@reexport using .Configurations
@reexport using .EnergyEvaluation
#@reexport using .Initialization


config_1_pos=[5.027408713597405, 0.8225487208608, 0.8002074984766196], 
[3.8103704397072757, -4.550276898524101, 1.9246323750116117], [3.8958673084302764, -1.696766636200351, -4.41810819064243], [1.7263854231760167, 4.419662861469047, -3.5870220871483305], [1.6498750899477201, -0.40314163121280217, 5.503606288915084], [0.5224424023158618, 4.50285868417844, 2.268650460128306], [-5.493157829142178, -1.7795328404374653, -1.0179293697841345], [-0.39420302761571624, -5.12240322796646, -2.7031178303271393], [-2.1199937267916824, -4.461177592594844, 3.2228302623759597], [-1.8538692117577136, -0.008587287490898408, -5.429932573587306], [-3.78495931086242, 4.196872632173511, -1.378630754843927], [-4.055107715334226, 1.0088787031633608, 3.6204086332043772], [-0.17521792511350928, -0.3088096517092877, -0.11071332794246591]

bc_1 = SphericalBC(radius=10.)

config_1=Config(config_1_pos, bc_ne13)

println(config_1)
println(config_1.pos[4])

println("old distances are:")
dist2 = [distance2(config_1.pos[4],b) for b in config_1.pos]
println(dist2)
dist2_mat_0=get_distance2_mat(config_1)
#println(dist2_mat_0)
en_unmoved = dimer_energy_atom(4, dist2_mat_0[4, :], elj_ne)
println("en_unmoved= ", en_unmoved)


trial_pos=[1.7284331683366974, 4.413552808225706, -3.5930835155632166]

println("new distances are:")
dist2 = [distance2(trial_pos,b) for b in config_1.pos]
println(dist2)
dist2_mat=get_distance2_mat(config_1)
#println(dist2_mat)
en_moved = dimer_energy_atom(4, dist2, elj_ne)
println("en_moved= ", en_moved)



end