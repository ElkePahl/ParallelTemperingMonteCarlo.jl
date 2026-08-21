using StaticArrays
using DelimitedFiles
using LinearAlgebra

export magic_cluster, face_centred_cubic, body_centred_cubic

"""
    face_cented_cubic(cell_size; r_min=1, boundary_condition=Cubic)

Create face-centred cubic configuration with `cell_size` cells. The atoms are placed such
that the smallest distance between atoms is `r_min`.
`boundary_condition` can be set to [`CubicBC`](@ref) or [`RectangularBC`](@ref).

Number of atoms per `cell_size`:
- `cell_size=1`: 32
- `cell_size=2`: 108
- `cell_size=3`: 256
There is no limit to `cell_size`.
"""
function face_centred_cubic(cell_size; r_min=1, boundary_condition=CubicBC)
    T = SVector{3,Float64}
    points = T[]
    for x in 0:cell_size, y in 0:cell_size, z in 0:cell_size
        push!(points, T(x, y, z))
        push!(points, T(x, y + 0.5, z + 0.5))
        push!(points, T(x + 0.5, y, z + 0.5))
        push!(points, T(x + 0.5, y + 0.5, z))
    end
    map!(pt -> check_boundary(CubicBC(cell_size + 1), pt), points)
    map!(pt -> √2 * r_min * pt, points)

    box_size = √2 * r_min * (cell_size + 1)
    if boundary_condition ≡ CubicBC
        bc = CubicBC(box_size)
    elseif boundary_condition ≡ RectangularBC
        bc = RectangularBC(box_size, box_size)
    else
        throw(ArgumentError("invalid `boundary_condition=$boundary_condition`"))
    end
    return Config(points, bc)
end

"""
    body_centred_cubic(cell_size; r_min=1, boundary_condition=CubicBC)

Create body-centred cubic configuration with `cell_size` cells. The atoms are placed such
that the smallest distance between atoms is `r_min`.
`boundary_condition` can be set to [`CubicBC`](@ref) or [`RectangularBC`](@ref).

Number of atoms per `cell_size`:
- `cell_size=1`: 16
- `cell_size=2`: 54
- `cell_size=3`: 128
There is no limit to `cell_size`.
"""
function body_centred_cubic(cell_size; r_min=1, boundary_condition=CubicBC)
    T = SVector{3,Float64}
    points = T[]
    for x in 0:cell_size, y in 0:cell_size, z in 0:cell_size
        push!(points, T(x, y, z))
        push!(points, T(x + 0.5, y + 0.5, z + 0.5))
    end
    map!(pt -> check_boundary(CubicBC(cell_size + 1), pt), points)
    map!(pt -> 2√3 / 3 * r_min * pt, points)

    box_size = 2√3 / 3 * r_min * (cell_size + 1)
    if boundary_condition ≡ CubicBC
        bc = CubicBC(box_size)
    elseif boundary_condition ≡ RectangularBC
        bc = RectangularBC(box_size, box_size)
    else
        throw(ArgumentError("invalid `boundary_condition=$boundary_condition`"))
    end
    return Config(points, bc)
end

"""
    min_distance(positions)

Return the minimum distance between any pair of `positions`.
"""
function min_distance(positions)
    dist = Inf
    for i in 1:length(positions)
        p1 = positions[i]
        for j in (i + 1):length(positions)
            p2 = positions[j]
            dist = min(dist, norm(p1 - p2))
        end
    end
    return dist
end
"""
    radius(positions)

Return the radius of cluster.
"""
function radius(positions)
    return maximum(norm, positions)
end

"""
    magic_cluster(magic_number_index; r_min=1, binding_sphere_radius=r_min/2)

Create magic number cluster configuration cells. The atoms are placed such that the smallest
distance between atoms is `r_min`.
The first argument is the magic number index (see below).
`binding_sphere_radius` sets the how much the radius of binding sphere is extended top of
the radius of the cluster.

## Number of atoms per `magic_number_index`:
- `magic_number_index=1`: 13
- `magic_number_index=2`: 55
- `magic_number_index=3`: 147
- `magic_number_index=4`: 309 (currently missing)
- `magic_number_index=5`: 561
- `magic_number_index=6`: 923
"""
function magic_cluster(magic_number_index; r_min=1, binding_sphere_radius=r_min / 2)
    if magic_number_index == 1
        filename = "13.txt"
    elseif magic_number_index == 2
        filename = "55.txt"
    elseif magic_number_index == 3
        filename = "147.txt"
    elseif magic_number_index == 4
        error("`magic_number_index = 4` currently not supported")
        filename = "309.txt"
    elseif magic_number_index == 5
        filename = "561.txt"
    elseif magic_number_index == 6
        filename = "923.txt"
    else
        throw(ArgumentError("`magic_number_index` should be an integer between 1 and 6"))
    end
    data = readdlm(joinpath(@__DIR__, "../data/", filename))
    points = map(axes(data, 1)) do i
        SVector{3,Float64}(data[i, :]...)
    end
    recentre!(points)
    scale_factor = r_min / min_distance(points)
    map!(p -> scale_factor * p, points)

    bc = SphericalBC(; radius=binding_sphere_radius + radius(points))

    if any(p -> isnothing(check_boundary(bc, p)), points)
        throw(ArgumentError("`binding_sphere_radius` too small for cluster."))
    end

    return Config(points, bc)
end
