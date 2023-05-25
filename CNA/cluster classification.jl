module Cluster_Classification
# Classify clusters according to CNA

export atomic_classify
export cluster_classify


function atomic_profile(bond_profile, N)
    atomicProfile = [Dict{String,Int}() for i in 1:N]
    l=length(bond_profile)
    for i=1:l
        atom1=bond_profile[i][1][1]
        atom2=bond_profile[i][1][2]
        key=bond_profile[i][2]
        push!(atomicProfile[atom1], key => get!(atomicProfile[atom1],key,0)+1)
		push!(atomicProfile[atom2], key => get!(atomicProfile[atom2],key,0)+1)
    end
    return atomicProfile
end

function total_profile(bond_profile,N)
    totalProfile=Dict{String,Int}()
    l=length(bond_profile)
    for i=1:l
        key=bond_profile[i][2]
        push!(totalProfile, key => get!(totalProfile,key,0)+1)
    end
    return totalProfile
end
				
function atomic_classify(bond_profile, N)
    atomic_classification=Array{String}(undef,N)
    atomicProfile = [Dict{String,Int}() for i in 1:N]
    atomicProfile = atomic_profile(bond_profile, N)
    for i=1:N
        atomic_classification[i]="unclassified"
    end
    for i=1:N
        if haskey(atomicProfile[i],"(5,5,5)")==true 
            if atomicProfile[i]["(5,5,5)"]>=6     #icoso core
                atomic_classification[i]="J"
            elseif atomicProfile[i]["(5,5,5)"]>=2    #icoso internal
                atomic_classification[i]="F"
            else
                atomic_classification[i]="G"
            end
        elseif haskey(atomicProfile[i],"(3,1,1)")==true
            if atomicProfile[i]["(3,1,1)"]>=3 && haskey(atomicProfile[i],"(3,2,2)")==false
                atomic_classification[i]="B"
            end
            if atomicProfile[i]["(3,1,1)"]>=3 && haskey(atomicProfile[i],"(3,2,2)")==true && atomicProfile[i]["(3,2,2)"]>=1
                atomic_classification[i]="H"
            end
            if atomicProfile[i]["(3,1,1)"]==2 && haskey(atomicProfile[i],"(2,1,1)")==true && atomicProfile[i]["(2,1,1)"]==2
                atomic_classification[i]="D"
            end
        elseif haskey(atomicProfile[i],"(4,2,2)")==true && atomicProfile[i]["(4,2,2)"]>=5
            atomic_classification[i]="E"
        elseif haskey(atomicProfile[i],"(4,2,1)")==true && atomicProfile[i]["(4,2,1)"]>=4
            atomic_classification[i]="A"
        elseif haskey(atomicProfile[i],"(2,1,1)")==true && atomicProfile[i]["(2,1,1)"]>=3
            atomic_classification[i]="C"
        end
        if haskey(atomicProfile[i],"(3,1,1)")==true && atomicProfile[i]["(3,1,1)"]==2 && haskey(atomicProfile[i],"(2,0,0)")==true && haskey(atomicProfile[i],"(2,0,0)")==2 && haskey(atomicProfile[i],"(2,1,1)")==true && atomicProfile[i]["(2,1,1)"]>=1
            atomic_classification[i]="I"       #anti_mackay
        end
    end
    return atomic_classification
end

#println(atomic_classify(bond_profile,N))
#println()


function cluster_classify(bond_profile, N)
    atomic_classification = Array{String}(undef,N)
    atomic_classification = atomic_classify(bond_profile, N)
    totalProfile=Dict{String,Int}()
    totalProfile=total_profile(bond_profile,N)
    ico_atom_total=0
    ico_core_total=0
    ico_edge_total=0
    fcc_atom_total=0
    hcp_atom_total=0
    unc_atom_total=0
    antim_atom_total=0
    ico_bond_max=0

    cluster="ambiguous"

    for i=1:N
        if atomic_classification[i]=="J"
            ico_core_total+=1
            #println(i)
        end
        if atomic_classification[i]=="F" || atomic_classification[i]=="G" || atomic_classification[i]=="J"
            #println("i=",i)
            ico_bond_atom=0
            ico_atom_total+=1
            l=length(bond_profile)
            local another_atom
            for j=1:l
                if bond_profile[j][1][1]==i
                    another_atom=bond_profile[j][1][2]
                    if atomic_classification[another_atom]=="F" || atomic_classification[another_atom]=="G" || atomic_classification[i]=="J"
                        ico_bond_atom+=1
                    end
                end
                if bond_profile[j][1][2]==i
                    another_atom=bond_profile[j][1][1]
                    if atomic_classification[another_atom]=="F" || atomic_classification[another_atom]=="G" #|| atomic_classification[i]=="H"
                        ico_bond_atom+=1
                    end
                end
            end
            #println(ico_bond)
            if ico_bond_atom>ico_bond_max
                ico_bond_max=ico_bond_atom
            end

        elseif atomic_classification[i]=="A" || atomic_classification[i]=="B" || atomic_classification[i]=="C" || atomic_classification[i]=="D"
            fcc_atom_total+=1
        elseif atomic_classification[i]=="E"
            hcp_atom_total+=1
        elseif atomic_classification[i]=="H"
            ico_edge_total+=1
        elseif atomic_classification[i]=="I"
            antim_atom_total+=1
        else
            unc_atom_total+=1
        end
    end
    #println(ico_bond_max)


    if ico_bond_max>=6
        cluster="ico"
        if ico_core_total>4
            cluster="ico_4+_core"
        elseif ico_core_total==4
            cluster="ico_4_core"
        elseif ico_core_total>=3
            cluster="ico_3_core"
        elseif ico_core_total>=2
            cluster="ico_2_core"
        end
    end


    #println("total ico atoms=", ico_atom_total)
    #println("total ico edge atoms=", ico_edge_total)
    #println("total ico core atoms=", ico_core_total)
    #println("total fcc atoms=", fcc_atom_total)
    #println("total hcp atoms=", hcp_atom_total)
    #println("total unc atoms=", unc_atom_total)
    #println("total anti-mackay atoms=", antim_atom_total)
    if ico_bond_max<=2 && haskey(totalProfile,"(5,5,5)")==true && totalProfile["(5,5,5)"]==ico_atom_total-1
        cluster="dec"
    end

    if ico_atom_total==0
        if fcc_atom_total + hcp_atom_total > unc_atom_total
            if fcc_atom_total >= hcp_atom_total*5
                cluster="fcc"
            elseif hcp_atom_total >= fcc_atom_total*5
                cluster="hcp"
            else
                cluster="twinned"
            end
        end
    end

    if antim_atom_total>=1
        cluster="anti_mackay"
    end

    return cluster
end

#println(cluster_classify(bond_profile_ico,N))

end

#@benchmark cluster_classify(bond_profile,N)

