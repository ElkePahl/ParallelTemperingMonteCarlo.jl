using ParallelTemperingMonteCarlo

link="C:\\Users\\shado\\OneDrive - The University of Auckland\\Documents\\2024\\Phy399\\ParallelTemperingMonteCarlo.jl\\scripts\\look-up_table_he_B0.3.txt";
potlut=LookuptablePotential(link);

θ = (potlut.start_angle:potlut.d_angle:(potlut.l_angle-1)*potlut.d_angle)*π/180
r = (potlut.start_dist:potlut.d_dist:(potlut.l_dist-1)*potlut.d_dist)

E = Iterators.map(
    t -> 
    (dimer_energy.(fill(potlut, potlut.l_dist), r.^2, fill(tan(t),potlut.l_dist)) |> (
        minimum
    ))
    , θ
) |> collect

rmin = Iterators.map(
    t -> 
    (dimer_energy.(fill(potlut, potlut.l_dist), r.^2, fill(tan(t),potlut.l_dist)) |> (
        s-> r[argmin(s)]
    ))
    , θ
) |> collect

plot(θ, rmin)