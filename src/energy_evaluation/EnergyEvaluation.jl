"""
    module EnergyEvaluation
Structs and functions relating to the calculation of energy. Includes both low and high level functions from individual PES calculations to state-specific functions. The structure is as follows:
-   Define abstract potential and potential_variables
-   Define PES functions
    -   DimerPotential
    -   ELJB
    -   Embedded Atom Model
    -   Machine Learning Potentials
-   EnergyUpdate function
    -   Calculates a new energy based on a trialpos for each PES type
-   InitialiseEnergy function
    -   Calculates potentialvariables and total energy from a new config to be used when initialising MCStates
-   SetVariables function
    -   Initialises the potential variables, aka creates a blank version of the struct for each type of PES
"""
module EnergyEvaluation

using StaticArrays, LinearAlgebra, StructArrays

using ..MachineLearningPotential
using ..Configurations
using ..Ensembles
using ..BoundaryConditions
using ..CustomTypes
import ..BoundaryConditions.long_range_correction

export AbstractPotential,
    AbstractDimerPotential, ELJPotential, ELJPotentialEven, AbstractMachineLearningPotential
export AbstractDimerPotentialB,
    ELJPotentialB,
    EmbeddedAtomPotential,
    RuNNerPotential,
    RuNNerPotential2Atom,
    LookupTablePotential
export AbstractPotentialVariables,
    DimerPotentialVariables, ELJPotentialBVariables, LookupTableVariables
export EmbeddedAtomVariables, NNPVariables, NNPVariables2a

export dimer_energy, dimer_energy_atom, dimer_energy_config, dimer_energy_update!
export energy_update!,
    set_variables,
    initialise_energy,
    dimer_energy_config,
    lrc,
    invrexp,
    calc_components,
    calc_energies_from_components,
    get_new_state_vars!,
    calc_new_runner_energy!

export swap_energy_update

include("abstract_types.jl")
include("ELJPotentials.jl")
include("LookupTablePotential.jl")
include("EmbeddedAtomPotential.jl")
include("MachineLearningPotentials.jl")

end
