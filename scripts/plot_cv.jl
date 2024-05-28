using JLD2
using Plots
path = "C:/Users/shado/OneDrive - The University of Auckland/Documents/2024/Phy399/ParallelTemperingMonteCarlo.jl/scripts/saved_results/";
files = vcat(filter(x->contains(x, "n10"), readdir(path)))
files1 = filter(x->contains(x, "result_"), readdir(path))

results_dict = Dict(f =>load_object(join([path;f])) for f in files1)

# resultit = results_dict|>collect

cv, temp_result = multihistogram_NVT(ensemble, temp, results_dict["result_3.32.jdl2"], 10^(-3), false);
format_label(s) = s |> s->split(s,"result_")[end] |> s->split(s, "jdl2")[1] |> s->split(s, "n10")[1] |> s->join(["R = ", s])

Plots.plot(temp_result, cv, labels=format_label("result_3.32.jdl2"))
for (f) in files
    cv, temp_result = multihistogram_NVT(ensemble, temp, results_dict[f], 10^(-3), false);
    Plots.plot!(temp_result, cv, label=format_label(f))
end

cv, temp_result = multihistogram_NVT(ensemble, temp, results_dict["result_5.32.jdl2"], 10^(-3), false);
Plots.plot!(temp_result, cv, labels=format_label("result_5.32.jdl2"))

cv, temp_result = multihistogram_NVT(ensemble, temp, results_dict["result_4.52n10.jdl2"], 10^(-3), false);
Plots.plot(temp_result, cv, labels="Helium Heat Capacity")
vline!([temp_result[argmax(cv)]], labels="Maximum Heat Capacity")

radii = [3.32, 4.32, 4.42, 4.52, 4.62, 4.82, 5.32]
format_key(s) = s |> s->split(s,"result_")[end] |> s->split(s, ".jdl2")[1] |> s->split(s, "n10")[1] |> s->parse(Float64, s)
function maxcv(obj) 
    cv, temp_result = multihistogram_NVT(ensemble, temp, obj, 10^(-3), false);
    return temp_result[argmax(cv)]
end

# results_dict2 = Dict(format_key(f) =>maxcv(load_object(join([path;f]))) for f in files1)

plotdict = sort(collect(results_dict2), by=x->x[1])
ke = plotdict |> s->map(x->x[1], s)
va = plotdict |> s->map(x->x[2], s)

Plots.plot(ke,va, labels="Heat Capacity Maximum")
    