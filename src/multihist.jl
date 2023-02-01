module Multihistogram


using DelimitedFiles, LinearAlgebra, StaticArrays

using ..InputParams

export multihistogram

"""
    readfile(xdir::String)
method 1: xdir::String -reads output files for the FORTRAN PTMC code written by Edison Florez.
method 2: Output::Output,Tvals::TempGrid - designed to receive output data from the Julia PTMC program: as the beta vector and NBins are defined in the structs they can be directly unpacked as output.

xdir is the directory containing the histogram information usually /path/to/output/histograms

`HistArray` is the NTrajxNBins array containing all histogram counts
`energyvector` is an NBins length vector containing the energy value of each bin
`beta` is an NTraj length vector of 1/(kBT)
`NBins,NTraj,kB` are constant values required throughout
"""
function readfile(xdir::String)
    f = open("$(xdir)histE.data", "r+")
    datafile=readdlm(f)
    kB = datafile[1]
    NTraj = datafile[2]
    T = copy(datafile[3,:])
    emin = datafile[4,1]
    emax = datafile[4,2]
    NBins = datafile[4,3]
    beta = 1 ./(kB.*T)
    de = (emax-emin)/(NBins-1)
    #Below we initialise the histogram array
    HistArray = Array{Float64}(undef,NTraj,NBins)
    
    for i in 1:NTraj
        c = open("$(xdir)histE.$i", "r")
        hist = readdlm(c)
        HistArray[i,:] = hist[1:NBins,2] 
    end

    println("Files Read")
    energyvector = [(j-1)*de + emin for j=1:NBins]

    return HistArray,energyvector,beta,NTraj,NBins,kB
end

function readfile(Output::Output, Tvals::TempGrid )

    kB = 3.16681196E-6  # in Hartree/K (3.166811429E-6)

    NTraj = length(Tvals.beta_grid)

    de = ( Output.en_max - Output.en_min )/(Output.n_bin - 1)

    energyvector = [(j-1)*de + Output.en_min for j=1:Output.n_bin ]

    HistArray = Array{Float64}(undef,NTraj,Output.n_bin)
    nbin_actual = length(Output.en_histogram[1])
    for i in 1:NTraj

        if nbin_actual == Output.n_bin
            HistArray[i,:] = Output.en_histogram[i]
        else
            HistArray[i,:] = Output.en_histogram[i][2:end-1]
        end
    end

    return HistArray, energyvector, Tvals.beta_grid, NTraj, Output.n_bin , kB
end
""" 
    processhist!(HistArray,energyvector,beta,NBins)
This function normalises the histograms, collates the bins into their total counts and then deletes any energy bin containing no counts -- this step is required to prevent NaN errors when doing the required calculations.

`HistArray,energyvector` are the total histograms and values of the energy bins respectively, they are only changed by normalisation and removal of unnecessary rows
`nsum` is merely the total histogram count for each energy bin

"""
function processhist!(HistArray,energyvector,NBins,NTraj)
    for i in 1:NTraj
        #NB in Florent's original code this factor of NBins*i normalised everything
        HistArray[i,:] = HistArray[i,:]./(NBins)#*i)
    end
    nsum = zeros(NBins)
    for j = 1:NBins
        nsum[j] = sum(HistArray[:,j])
    end
    #as it causes colossal headaches, we will now delete all rows
    #which have exactly 0 histogram counts. Trust me it's needed.
    k=1
    while k <= NBins

        if nsum[k] == 0
            deleteat!(nsum,k)
            deleteat!(energyvector,k)
            HistArray= HistArray[1:end,1:end .!=k]
            NBins -= 1
        else
            k=k+1
        end
    end
    return HistArray,energyvector,nsum,NBins
end
"""
    initialise(xdir::String)
    (Output::Output,Tvec::TempGrid)
function to retrieve all histogram information from the histogram directory outputted by Edison's PTMC code for method one, or directly from the output data given from the Julia PTMC code.
    
    We read the files with readfile, process the file with processhist! and output all relevant arrays and constants as defined in the constituent functions.

"""
function initialise(xdir::String)
    HistArray,energyvector,beta,NTraj,NBins,kB = readfile(xdir::String)

    HistArray,energyvector,nsum,NBins = processhist!(HistArray,energyvector,NBins,NTraj)

    return HistArray,energyvector,beta,nsum,NTraj,NBins,kB
    
end
function initialise(Output::Output,Tvec::TempGrid)

    HistArray,energyvector,beta,NTraj,NBins,kB = readfile(Output,Tvec)

    HistArray,energyvector,nsum,NBins = processhist!(HistArray,energyvector,NBins,NTraj)

    return HistArray,energyvector,beta,nsum,NTraj,NBins,kB

end
"""
    nancheck(X::Vector)
    nancheck(X::Matrix)
function to ensure no vector or matrix contains NaN as this ruins the linear algebra.
"""
function nancheck(X :: Vector)
    N = length(X)
    check = 1
    for i=1:N
        if isnan(X[i]) == true
            check = 0
        end
    end
    return check
end
function nancheck(X::Matrix)
    check = 1
    N1 = size(X)[1]
    N2 = size(X)[2]
    for i = 1:N1
        for j = 1:N2
            if isnan(X[i,j]) == true
                check = 0
            end
        end
    end
    return check
end
""" 
    bvector(HistArray::Matrix,energyvector::Vector,beta::Vector,nsum::Vector,NTraj,NBins)
function to calculate the b vector relevant to solving the RHS of the multihistogram equation. 

"""
function bvector(HistArray::Matrix,energyvector,beta,nsum,NTraj,NBins)
    #Below we find the matrix of values n_{ij}*(ln(n_{ij} + beta_iE_j)
    #which appears frequently
    logmat = Array{Float64}(undef,NTraj,NBins)
    for i in 1:NTraj
        for j in 1:NBins
            logmat[i,j] = log(HistArray[i,j]) + beta[i]*energyvector[j]
        end
    end
    bmat = Array{Float64}(undef, NTraj,NBins)
    B = zeros(NTraj)
    rhvec = zeros(NBins)
    #The loop below guarantees we do not get values of NaN, as -inf*0.0 != -inf*false
    for i = 1:NTraj
        for j = 1:NBins
            if HistArray[i,j] == 0.0
                bmat[i,j] = 0.0
            else
                bmat[i,j] = HistArray[i,j]*logmat[i,j]
            end
        end
    end
    #the penultimate step is the vectors of length j to be summed in the rhterm
    for j=1:NBins
        rhvec[j] = sum(bmat[:,j])
    end
#Now we have our two matrices, the ith element is a sum over the energies for bmat-nij*rhvec
    for i in 1:NTraj 
        B[i] = sum( bmat[i,:] .- HistArray[i,:] .*rhvec./nsum )
    end
    println("B Vector Calculated")
    return B,bmat
end
"""
    amatrix(HistArray :: Matrix,nsum,NTraj)
This function calculates the LHS of the multihistogram equation, the A matrix.
"""

function amatrix(HistArray :: Matrix,nsum,NTraj)
    A = Array{Float64}(undef,NTraj,NTraj)

    for i = 1:NTraj
        for ip = 1:NTraj
            
            A[i,ip] = -sum(HistArray[i,:].*HistArray[ip,:]./nsum[:])
            
            if i == ip
                A[i,ip] += sum(HistArray[i,:])
            end
        end
    end
    println("A Matrix Calculated")

    return A
end
""" 
    systemsolver(HistArray,energyvector,beta,nsum,NTraj,NBins,kB)
systemsolver is used to determine the solution Alpha to the linear equation Ax = b where A and b are the A matrix and b vector described above. This is fundamentally how the multihistogram method works
"""
function systemsolver(HistArray,energyvector,beta,nsum,NTraj,NBins)
    #solve the b vector
    b,bmat = bvector(HistArray,energyvector,beta,nsum,NTraj,NBins)
    #solve the A matrix
    A = amatrix(HistArray,nsum,NTraj)
    #Check NaN
    c1 = nancheck(A)
    c2 = nancheck(b)
    if c1 == 1 && c2 == 1
        #If there isn't NaN we solve the system and get entropy
        alpha = A \ b
        #alpha = alpha.- alpha[NTraj]
        println("system solved!")
        S = Entropycalc(alpha, bmat, HistArray,nsum,NBins)
        println("Entropy Found")
        return alpha, S
    else #if NaN is present we return an error
        println("system cannot be solved")
    end

end
"""
    Entropycalc(alpha::Vector, bmat:: Matrix, HistArray::Matrix,nsum,NBins)
Having determined the vector solution to Ax=b, we input alpha and the "b-matrix" the term n_{ij}*(ln(n_{ij} + beta_iE_j) we can find the entropy as a function of Energy.
"""
function Entropycalc(alpha::Vector, bmat:: Matrix, HistArray::Matrix,nsum,NBins)
    S_E = []
    for j = 1:NBins 
        var = (sum(bmat[:,j] .- HistArray[:,j].*alpha))/nsum[j]
        push!(S_E,var)
    end

    return S_E
end
"""
    analysis(energyvector:: Vector, S_E :: Vector, beta::Vector,kB::Float64; NPoints=600)
NB: NPoints is an optional keyword expressing how dense the points should be populated. 

analysis takes in the energy bin values, entropy per energy and inverse temperatures beta. It calculates the temperatures T, and then finds the partition function -- note that the boltzmann factors XP are self-scaling so they vary from 1 to 100, this is not necessary but prevents numerical errors in regions where the partition function would otherwise explode in value. 

Output is the partition function, heat capacity and its first derivative as a function of temperature.
"""
function analysis(energyvector, S_E :: Vector, beta,kB::Float64, NPoints)
    
    NBins = length(energyvector)
    Tvec = 1 ./ (kB*beta)
    dT = (last(Tvec) - first(Tvec))/NPoints
    T = [(i-1)*dT + first(Tvec) for i = 1:NPoints]
   #Initialise all relevant vectors
   y = Array{Float64}(undef,NPoints,NBins)
   XP = Array{Float64}(undef,NPoints,NBins)
   nexp = 0

   
   Z = Array{Float64}(undef,NPoints)
   U = Array{Float64}(undef,NPoints)
   U2 = Array{Float64}(undef,NPoints)
   S_T = Array{Float64}(undef,NPoints)
   Cv = Array{Float64}(undef,NPoints)
   dCv = Array{Float64}(undef,NPoints)
   r2 = Array{Float64}(undef,NPoints)
   r3 = Array{Float64}(undef,NPoints)
   #below we begin the calculation of thermodynamic quantities

   for i = 1:NPoints
       #y is a matrix of free energy
       y[i,:] = S_E[:] .- energyvector[:]./(T[i]*kB)
       #here we set the zero of free energy

       Stest = nancheck(S_E)
       energyvectest = nancheck(energyvector)
       ytest = nancheck(y[i,:])

       if Stest == 0
        println("Entropy is a problem")
       elseif energyvectest == 0
        println("energyvector is a problem")
       elseif ytest == 0
        println("vector $i at temperature $(T[i]) is a problem")
       end

       #count=0  this variable was included for bug testing and should be excluded from the main program

       #below we calculate the partition function
       
       @label start
       XP[i,:] = exp.(y[i,:].-nexp)
       Z[i] = sum(XP[i,:] )
       
       #this loop exists to make sure the scale of our partition function is sensible
       #the numbers are utterly arbitrary, they have been chosen so that they don't create a loop
        
        if Z[i] < 1.
            #count += 1

            nexp -= 1.  
            @goto start
        elseif Z[i] > 100.

            #count += 1
            nexp += 0.7

            @goto start
        end
        
       U[i] = sum(XP[i,:].*energyvector[:])/Z[i]
       U2[i] = sum(XP[i,:].*energyvector[:].*energyvector[:])/Z[i]       
       r2[i] = sum(XP[i,:].*(energyvector[:].-U[i] ).*(energyvector[:].-U[i] ) )/Z[i]
       r3[i] = sum(XP[i,:].*(energyvector[:].-U[i] ).*(energyvector[:].-U[i] ).*(energyvector[:].-U[i] ) )/Z[i]
       Cv[i] = (U2[i] - U[i]*U[i])/kB/(T[i]^2)
       S_T[i] = U[i]/T[i] + kB*log(Z[i])
       dCv[i] = r3[i]/kB^2/T[i]^4 - 2*r2[i]/kB/T[i]^3
   end
return Z,Cv,dCv,S_T,T
end
""" 
    runmultihistogram(HistArray,energyvector,beta,nsum,NTraj,NBins,kB,outdir::String)

This function completely determines the properties of a system given by the output of the initialise function and a specified directory to write to. It outputs four files with the following information:

    histograms.data The top line are the corresponding energy values and the next NTraj lines are the raw histogram data. This file can be used to plot the histograms if needed. 
    
    Sol.X containing the solution to the linear equation Ax=B, 

    S.data containing the energy values and corresponding entropies 

    analysis.NVT containing the temperatures, partition function, heat capacity and its derivative. NB now includes the temperature dependent Entropy function.
    
"""
function run_multihistogram(HistArray,energyvector,beta,nsum,NTraj,NBins,kB,outdir::String,NPoints)

    #HistArray,energyvector,beta,nsum,NTraj,NBins,kB = initialise(xdir)

    #hist=histplot(HistArray,energyvector,NTraj)
    #png(hist,"$(xdir)histo")
    alpha,S = systemsolver(HistArray,energyvector,beta,nsum,NTraj,NBins)

    Z,C,dC,S_T,T = analysis(energyvector,S,beta,kB,NPoints)
    println("Quantities found")
    #cvplot = plot(T,C,xlabel="Temperature (K)",ylabel="Heat Capacity")
    #png(cvplot,"$(xdir)Cv")
    #dcvplot = plot(T,dC,xlabel="Temperature(K)",ylabel="dCv")
    #png(dcvplot,"$(xdir)dC")
    println("analysis complete")
    histfile = open("$(outdir)/histograms.data", "w")
    writedlm(histfile,[energyvector])
    writedlm(histfile,HistArray)
    close(histfile)

    solfile = open("$(outdir)/Sol.X", "w")
    writedlm(solfile,["alpha"])
    writedlm(solfile,[alpha])
    close(solfile)

    entropyfile = open("$(outdir)/S.data", "w")
    writedlm(entropyfile, ["E" "Entropy"])
    writedlm(entropyfile, [energyvector S ])
    close(entropyfile)

    cvfile = open("$(outdir)/analysis.NVT", "w")
    writedlm(cvfile, ["T" "Z" "Cv" "dCv" "S(T)"])
    writedlm(cvfile, [T Z C dC S_T])
    close(cvfile)
       
end

"""
    multihistogram(xdir::String)
    multihistogram(Output::Output,Tvec::TempGrid)
Function has two methods which vary only in how the initialise function is called: one takes a directory and writes the output of the multihistogram analysis to that directory, the other takes the output and temperature grid and writes to the current directory unless specified otherwise.
    
    The output of this function are the four files defined in run_multihistogram.

"""
function multihistogram(xdir::String)
    HistArray,energyvector,beta,nsum,NTraj,NBins,kB = initialise(xdir)
    run_multihistogram(HistArray,energyvector,beta,nsum,NTraj,NBins,kB, xdir)
end

function multihistogram(Output::Output,Tvec::TempGrid; outdir = pwd(), NPoints=600)
    HistArray,energyvector,beta,nsum,NTraj,NBins,kB = initialise(Output,Tvec)
    run_multihistogram(HistArray,energyvector,beta,nsum,NTraj,NBins,kB,outdir,NPoints)



end

end