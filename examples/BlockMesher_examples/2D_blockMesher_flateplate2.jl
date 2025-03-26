using XCALibre

n_vertical      = 20 #400
n_horizontal1   = 20 #500

#(x,y,z)
p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(0.0,1.0,0.0)
p4 = Point(1.0,1.0,0.0)
p5 = Point(0.0,2.0,0.0)
p6 = Point(1.0,2.0,0.0)
points = [p1, p2, p3, p4, p5, p6]

# Edges in x-direction
e1 = line!(points,1,2,n_horizontal1)
e2 = line!(points,3,4,n_horizontal1)
e3 = line!(points,5,6,n_horizontal1)

# Edges in y-direction
e4 = line!(points,1,3,n_vertical)
e5 = line!(points,2,4,n_vertical)
e6 = line!(points,3,5,n_vertical)
e7 = line!(points,4,6,n_vertical)
edges = [e1, e2, e3, e4,e5,e6,e7]

b1 = quad(edges, [1,2,4,5])
b2 = quad(edges, [2,3,6,7])
blocks = [b1,b2]

patch1 = Patch(:inlet,  [4,6])
patch2 = Patch(:outlet, [5,7])
patch3 = Patch(:wall, [1])
patch4 = Patch(:top,    [3])
patches = [patch1, patch2, patch3, patch4]


builder = MeshBuilder2D(points, edges, patches, blocks)
mesh = generate!(builder)
mesh_new = XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64) #move to end of genrerate function

# Set up case for flat plate 

velocity = [0.0, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_new
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0)
)

 @assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.8,
        rtol = 1e-1
    ),
    p = set_solver(
        model.momentum.p;
        solver = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.2,
        rtol = 1e-2
    )
)

runtime = set_runtime(iterations=5000, write_interval=1000, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs