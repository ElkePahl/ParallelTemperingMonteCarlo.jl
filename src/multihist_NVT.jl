module Multihistogram_NVT

using DelimitedFiles, LinearAlgebra, StaticArrays

using ..InputParams
using ..EnergyEvaluation

export multihistogram_NVT

function temp_trajectories(temp)
    tempnumber = length(temp.t_grid)
    tempnumber_result = tempnumber * 10
    return tempnumber,tempnumber_result
end

function histogram_initialise_en(ensemble::NVT,temp,results)
    k=3.166811429/10^6
    temp_o=temp.t_grid
    beta=temp.beta_grid
    Emin=results.en_min
    Emax=results.en_max
    Ebins=results.n_bin
    dEhist=(Emax-Emin)/Ebins
    ENhistogram=results.en_histogram
    return k,temp_o,beta,Emin,Ebins,dEhist,ENhistogram
end

function Temp_grid_result(ti,tf,tempnumber_result)
    temp_grid_result = TempGrid{tempnumber_result}(ti,tf) 
    temp_result=temp_grid_result.t_grid
    beta_result=temp_grid_result.beta_grid
    return temp_result,beta_result
end

function free_energy_initialise(ENhistogram,Ebins,tempnumber,tempnumber_result)
    free_energy=Array{Float64}(undef,tempnumber)
    new_free_energy=Array{Float64}(undef,tempnumber)
    normalconst=Array{Float64}(undef,tempnumber_result)
    ncycles=Array{Float64}(undef,tempnumber)

    for i=1:tempnumber
        free_energy[i]=0
        new_free_energy[i]=0
        ncycles[i]=0
        for m=1:Ebins
            ncycles[i]=ncycles[i]+ENhistogram[i][m+1]
        end
    end
    
    for i=1:tempnumber_result
        normalconst[i]=0
    end
    return free_energy, new_free_energy, normalconst, ncycles
end

function quasiprob(betat,m,ncycles,dEhist,Emin,tempnumber,ENhistogram,beta,free_energy)
    energy_t=Emin+(m-0.5)*dEhist
    quasiprob=0
    denom=0
    offset=-10^6
    for i=1:tempnumber
        offset=max(offset,-beta[i]*energy_t-free_energy[i])
    end
    offset=offset+log(10^3)
    for i=1:tempnumber
        quasiprob=quasiprob+ENhistogram[i][m+1]
        denom=denom+ncycles[i]*exp(-beta[i]*energy_t-free_energy[i]-offset)
    end

    quasiprob=quasiprob/denom*exp(-betat*energy_t-offset)
    return quasiprob
end

"""
Multihistogram analysis for NPT
    multihistgram_NPT(ensemble, temp, results, conv_threshold, readfile)
    conv_threshold is the convergence threshold, which user can choose.
    Now "readfile" can only be false.
    Example: multihistgram_NPT(ensemble, temp, results, 10^(-3), false)
"""
function multihistogram_NVT(ensemble, temp, results, conv_threshold, readfile)
    if readfile==false
        tempnumber,tempnumber_result = temp_trajectories(temp)
        k,temp_o,beta,Emin,Ebins,dEhist,ENhistogram = histogram_initialise_en(ensemble,temp,results)
    end
    temp_result,beta_result = Temp_grid_result(temp_o[1],temp_o[tempnumber],tempnumber_result)

    free_energy, new_free_energy, normalconst, ncycles = free_energy_initialise(ENhistogram,Ebins,tempnumber,tempnumber_result)
    
    for it=1:1000
        println("iteration=",it)
        for i=1:tempnumber
            local betat
            betat=beta[i]
            new_free_energy[i]=0
            for m=1:Ebins
                new_free_energy[i]=new_free_energy[i]+quasiprob(betat,m,ncycles,dEhist,Emin,tempnumber,ENhistogram,beta,free_energy)
            end
            
            #println(new_free_energy[i])
            new_free_energy[i]=log(new_free_energy[i])
            #println(new_free_energy[i])
        end
        
        local delta
        delta=0
        for i=1:tempnumber
            delta=delta+abs(new_free_energy[i]-free_energy[i])^2
            free_energy[i]=new_free_energy[i]
        end
        println(delta)
        println()
    
        if delta<conv_threshold
            println("iteration finished")
            break             #if converged, exit the loop
        end
    end
    
    for i=1:tempnumber_result
        betat=beta_result[i]
        for m=1:Ebins
            normalconst[i]=normalconst[i]+quasiprob(betat,m,ncycles,dEhist,Emin,tempnumber,ENhistogram,beta,free_energy)
        end
    end
    
    cv=zeros(tempnumber_result)
    for i=1:tempnumber_result
        betat=beta_result[i]
        eenergy=0
        eenergy2=0
        for m=1:Ebins
            energy_t=Emin+(m-0.5)*dEhist
    
            eenergy=eenergy+quasiprob(betat,m,ncycles,dEhist,Emin,tempnumber,ENhistogram,beta,free_energy)/normalconst[i]*energy_t
            eenergy2=eenergy2+quasiprob(betat,m,ncycles,dEhist,Emin,tempnumber,ENhistogram,beta,free_energy)/normalconst[i]*energy_t^2
        end
        println("temperature: ",temp_result[i])
        println("energy: ",eenergy)
        println("heat capacity: ", (eenergy2-eenergy^2)/(k*temp_result[i]^2))
        println()
        cv[i]=(eenergy2-eenergy^2)/(k*temp_result[i]^2)
    end
    println("temperature array: ",temp_result)
    println("heat capacity array: ",cv)
end



end