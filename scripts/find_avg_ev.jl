#Find the average energy and volume from histograms

N_atoms=108

e_avg=[]
v_avg=[]

for i=1:24
    e_index=findmax(results.ev_histogram[i])[2][1]
    v_index=findmax(results.ev_histogram[i])[2][2]

    e = results.en_min + results.delta_en_hist * (e_index+0.5)
    v = results.v_min + results.delta_v_hist * (v_index+0.5)

    push!(e_avg,e/N_atoms)
    push!(v_avg,v/N_atoms/AtoBohr^3)
end

println(e_avg)
println(v_avg)