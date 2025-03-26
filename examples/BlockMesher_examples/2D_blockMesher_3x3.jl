using XCALibre

n_vertical      = 500 #400
n_horizontal1   = 500 #500

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(2.0,0.0,0.0)
p4 = Point(3.0,0.0,0.0)
p5 = Point(0.0,1.0,0.0)
p6 = Point(1.0,1.0,0.0)
p7 = Point(2.0,1.0,0.0)
p8 = Point(3.0,1.0,0.0)
p9 = Point(0.0,2.0,0.0)
p10 = Point(1.0,2.0,0.0)
p11 = Point(2.0,2.0,0.0)
p12 = Point(3.0,2.0,0.0)
p13 = Point(0.0,3.0,0.0)
p14 = Point(1.0,3.0,0.0)
p15 = Point(2.0,3.0,0.0)
p16 = Point(3.0,3.0,0.0)

points = [p1, p2, p3, p4, p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]

# Edges in x-direction
e1 = line!(points,1,2,n_horizontal1)
e2 = line!(points,2,3,n_horizontal1)
e3 = line!(points,3,4,n_horizontal1)
e4 = line!(points,5,6,n_horizontal1)
e5 = line!(points,6,7,n_horizontal1)
e6 = line!(points,7,8,n_horizontal1)
e7 = line!(points,9,10,n_horizontal1)
e8 = line!(points,10,11,n_horizontal1)
e9 = line!(points,11,12,n_horizontal1)
e10 = line!(points,13,14,n_horizontal1)
e11 = line!(points,14,15,n_horizontal1)
e12 = line!(points,15,16,n_horizontal1)


# Edges in y-direction
e13 = line!(points,1,5,n_vertical)
e14 = line!(points,2,6,n_vertical)
e15 = line!(points,3,7,n_vertical)
e16 = line!(points,4,8,n_vertical)
e17 = line!(points,5,9,n_vertical)
e18 = line!(points,6,10,n_vertical)
e19 = line!(points,7,11,n_vertical)
e20 = line!(points,8,12,n_vertical)
e21 = line!(points,9,13,n_vertical)
e22 = line!(points,10,14,n_vertical)
e23 = line!(points,11,15,n_vertical)
e24 = line!(points,12,16,n_vertical)

edges = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16,e17,e18,e19,e20,e21,e22,e23,e24]

b1 = quad(edges, [1,4,13,14])
b2 = quad(edges, [2,5,14,15])
b3 = quad(edges, [3,6,15,16])
b4 = quad(edges, [4,7,17,18])
b5 = quad(edges, [5,8,18,19])
b6 = quad(edges, [6,9,19,20])
b7 = quad(edges, [7,10,21,22])
b8 = quad(edges, [8,11,22,23])
b9 = quad(edges, [9,12,23,24])

blocks = [b1, b2, b3, b4, b5, b6, b7, b8, b9]

patch1 = Patch(:inlet,  [13,17,21])
patch2 = Patch(:outlet, [16,20,24])
patch3 = Patch(:wall, [1,2,3])
patch4 = Patch(:top,    [10,11,12])
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

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs