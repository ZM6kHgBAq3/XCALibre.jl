module BlockMesher2D

using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
using KernelAbstractions
# using CUDA, AMDGPU

include("Mesh_0_types.jl")
include("Mesh2D_1_builder_types.jl")
include("Mesh2D_2_builder.jl")
include("Mesh2D_3_connectivity.jl")
include("Mesh2D_4_geometry.jl")
include("Mesh2D_5_access_functions.jl")
include("Mesh2D_6_elements.jl")
include("Mesh2D_7_generate.jl")


# include("Mesh3D/Mesh3D_0_types.jl")

include("Mesh_1_functions.jl")

end