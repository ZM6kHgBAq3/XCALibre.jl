export dirichlet, neumann

# LAPLACIAN TERM (CONSTANT)

@inline (bc::Dirichlet)(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where T<:AbstractFloat = begin
    ap = term.sign[1]*(-term.J*face.area/face.delta)
    A[cellID,cellID] += ap
    b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann)(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where T<:AbstractFloat = begin
    phi = term.phi 
    values = phi.values
    A[cellID,cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta)
    b[cellID] += 0.0*term.sign[1]*(-term.J*face.area/face.delta*values[cellID])
    nothing
end

# LAPLACIAN TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where T<:AbstractScalarField = begin
    ap = term.sign[1]*(-term.J(fID)*face.area/face.delta)
    A[cellID,cellID] += ap
    b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann)(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where T<:AbstractScalarField = begin
    phi = term.phi 
    values = phi.values
    A[cellID,cellID] += 0.0*term.sign[1]*(-term.J(fID)*face.area/face.delta)
    b[cellID] += 0.0*term.sign[1]*(-term.J(fID)*face.area/face.delta*values[cellID])
    nothing
end

# DIVERGENCE TERM (CONSTANT)

@inline (bc::Dirichlet)(
    term::Divergence{Linear, SVector{3, Float64}}, A, b, cellID, cell, face, fID
    ) = begin
    A[cellID,cellID] += 0.0 #term.sign[1]*(term.J⋅face.normal*face.area)/2.0
    b[cellID] += term.sign[1]*(-term.J⋅face.normal*face.area*bc.value)
    nothing
end

@inline (bc::Neumann)(
    term::Divergence{Linear, SVector{3, Float64}}, A, b, cellID, cell, face, fID
    ) = begin
    ap = term.sign[1]*(term.J⋅face.normal*face.area)
    A[cellID,cellID] += ap # 2.0*ap
    b[cellID] += 0.0*-ap*0.5
    nothing
end

# DIVERGENCE TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Divergence{Linear, FaceVectorField{I,F}}, A, b, cellID, cell, face, fID
    ) where {I,F} = begin
    A[cellID,cellID] += 0.0 #term.sign[1]*(term.J⋅face.normal*face.area)/2.0
    b[cellID] += term.sign[1]*(-term.J(fID)⋅face.normal*face.area*bc.value)
    nothing
end

@inline (bc::Neumann)(
    term::Divergence{Linear, FaceVectorField{I,F}}, A, b, cellID, cell, face, fID
    ) where {I,F} = begin
    phi = term.phi 
    values = phi.values
    ap = term.sign[1]*(term.J(fID)⋅face.normal*face.area)
    A[cellID,cellID] += ap # 2.0*ap
    b[cellID] += 0*-ap*values[cellID]*0.5
    nothing
end