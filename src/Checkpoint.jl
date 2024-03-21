"""
    Checkpoint
Module designed to save relevant parameters and configurations throughout the simulation to allow restarting.
"""

module Checkpoint 

using StaticArrays,DelimitedFiles
using ..Configurations
using ..InputParams
using ..Ensembles
using ..EnergyEvaluation
using ..MCStates

#---------------------------------------------------------------#
#-----------------------Static Parameters-----------------------#
#---------------------------------------------------------------#
"""
    writeparams(savefile,params,temp)
Function to write the `mc_params` and `temp` data into a `savefile`. These are static parameters that define how the simulation is to proceed such as the number of cycles, trajectories and the temperatures to be covered.
"""
function writeparams(savefile,params,temp)
    headersvec = ["cycles:" "sample_rate:" "n_traj:" "n_atoms:" "t_i:" "t_f"]
    paramsvec = [params.mc_cycles params.mc_sample params.n_traj params.n_atoms first(temp.t_grid) last(temp.t_grid)]
    writedlm(savefile, [headersvec, paramsvec], ' ' )
end
"""
    writeensemble(savefile,ensemble::NVT)
    writeensemble(savefile,ensemble::NPT)
Function to write the `ensemble` data into a savefile including the move types. First method is for the NVT ensemble which does not include volume changes, second method is NPT ensemble and does inclue volume moves.
"""
function writeensemble(savefile,ensemble::NVT)

    headersvec = ["ensemble" "n_atom_moves" "n_atom_swaps" ]
    valuesvec = ["NVT{$(ensemble.n_atoms)}" ensemble.n_atom_moves ensemble.n_atom_swaps]
    writedlm(savefile, [headersvec, valuesvec], ' ' )
end
function writeensemble(savefile,ensemble::NPT)

    headersvec = ["ensemble" "n_atom_moves" "n_volume_moves" "n_atom_swaps" ]
    valuesvec = ["NPT{$(ensemble.n_atoms)}" ensemble.n_atom_moves ensembles.n_volume_moves emsemble.n_atom_swaps]
    writedlm(savefile, [headersvec, valuesvec], ' ' )
end
"""
    writepotential(savefile,potential::Ptype)
    Ptype = AbstractDimerPotential,AbstractDimerPotentialB,EmbeddedAtomPotential,AbstractMachineLearningPotential
Function to write potential surface information into `savefile`. implemented methods are the Embedded Atom Model, Extended Lennard-Jones and ELJ in Magnetic Field. This does not work for machine learning potentials.
"""
function writepotential(savefile,potential::Ptype) where Ptype <: AbstractDimerPotential
    coeff_vec = transpose([potential.coeff[i] for i in eachindex(potential.coeff)])

    write(savefile,"ELJ_coeffs: " )
    writedlm(savefile, coeff_vec, ' ')
end
function writepotential(savefile,potential::Ptype) where Ptype <: AbstractDimerPotentialB

    coeff_vec_a = transpose([potential.coeff_a[i] for i in eachindex(potential.coeff_a)])
    coeff_vec_b = transpose([potential.coeff_b[i] for i in eachindex(potential.coeff_b)])
    coeff_vec_c = transpose([potential.coeff_c[i] for i in eachindex(potential.coeff_c)])
    write(savefile,"ELJB \n")
    writedlm(savefile, [coeff_vec_a, coeff_vec_b, coeff_vec_c])
end
function writepotential(savefile,potential::Ptype) where Ptype <: EmbeddedAtomPotential
    write(savefile,"EAM: $(potential.n) $(potential.m) $(potential.ean) $(potential.eCam) \n")
end 
function writepotential(savefile,potential::Ptype) where Ptype <: AbstractMachineLearningPotential
    write(savefile,"Define the potential elsewhere, it's too complicated for simple I/O \n")
end
"""
    save_init(potential,ensemble,params,temp)
Function to write all static parameters into a single parameters file. If a params file does not exist, it is created in ./checkpoint as params.data. This contains the `mc_params` `ensemble` and `potential` data.
"""
function save_init(potential,ensemble,params,temp)
    if ispath("./checkpoint/params.data") == true
    else
        mkpath("./checkpoint/")
        paramsfile = open("./checkpoint/params.data","w+")
        
        writeparams(paramsfile,params,temp)
        writeensemble(paramsfile,ensemble)
        writepotential(paramsfile,potential)
        
        close(paramsfile)
    end
end
end