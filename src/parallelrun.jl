module ParallelRun

using Distributed
@everywhere begin
    using StaticArrays,DelimitedFiles

    using ..BoundaryConditions
    using ..Configurations
    using ..InputParams
    using ..MCMoves
    using ..EnergyEvaluation
    using ..RuNNer
    using ..MCRun

    import ..MCRun: initialise_histograms!,updatehistogram!,update_max_stepsize,sampling_step!,save_results,save_states
    
end






end