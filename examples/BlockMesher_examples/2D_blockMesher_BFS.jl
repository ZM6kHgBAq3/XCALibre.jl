using XCALibre

n_vertical      = 500 #400
n_horizontal1   = 500 #500
n_horizontal2   = 500 #500
n_horizontal3   = 500 #500

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(2.0,0.0,0.0)
p4 = Point(0.0,1.0,0.0)
p5 = Point(1.0,1.0,0.0)
p6 = Point(2.0,1.0,0.0)
p7 = Point(3.0,1.0,0.0)
p8 = Point(0.0,2.0,0.0)
p9 = Point(1.0,2.0,0.0)
p10 = Point(2.0,2.0,0.0)
p11 = Point(3.0,2.0,0.0)

points = [p1, p2, p3, p4, p5,p6,p7,p8,p9,p10,p11]

# Edges in x-direction
e1 = line!(points,1,2,n_horizontal1)
e2 = line!(points,2,3,n_horizontal2)
e3 = line!(points,4,5,n_horizontal1)
e4 = line!(points,5,6,n_horizontal2)
e5 = line!(points,6,7,n_horizontal3)
e6 = line!(points,8,9,n_horizontal1)
e7 = line!(points,9,10,n_horizontal2)
e8 = line!(points,10,11,n_horizontal3)

# Edges in y-direction
e9 = line!(points,1,4,n_vertical)
e10 = line!(points,2,5,n_vertical)
e11 = line!(points,3,6,n_vertical)
e12 = line!(points,4,8,n_vertical)
e13 = line!(points,5,9,n_vertical)
e14 = line!(points,6,10,n_vertical)
e15 = line!(points,7,11,n_vertical)
edges = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15]

b1 = quad(edges, [1,3,9,10])
b2 = quad(edges, [2,4,10,11])
b3 = quad(edges, [3,6,12,13])
b4 = quad(edges, [4,7,13,14])
b5 = quad(edges, [5,8,14,15])
blocks = [b1, b2, b3, b4, b5]

patch1 = Patch(:inlet,  [9,12])
patch2 = Patch(:outlet, [15])
patch3 = Patch(:wall, [1,2,5,11])
patch4 = Patch(:top,    [6,7,8])
patches = [patch1, patch2, patch3, patch4]


builder = MeshBuilder2D(points, edges, patches, blocks)
mesh = generate!(builder)
mesh_new = XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64) #move to end of genrerate function

# Set up case for flat plate 

velocity = [0.2, 0.0, 0.0]
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

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs