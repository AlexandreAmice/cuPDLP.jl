using LinearAlgebra
using SparseArrays

function random_feasible_lp(m, n, A_density = 1)
    s0 = randn(m)
    lamb0 = max.(-s0, 0)
    s0 = max.(s0, 0)
    x0 = randn(n)
    A = randn(m,n)
    mask = rand(m,n) .> A_density
    A[mask] .= 0
    A = sparse(A)
    b = A * x0 + s0
    c = -transpose(A) * lamb0
    return c, A, b
end


# m, n = 15, 10
# c, A, b = random_feasible_lp(m, n, 1)
# println(A)