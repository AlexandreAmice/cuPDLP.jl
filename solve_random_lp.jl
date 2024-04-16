using LinearAlgebra
import cuPDLP
using JuMP
using MosekTools
using Clarabel


include("random_feasible_lp.jl")
function main()
    m, n = Int(1e5), Int(1e3)
    A_density = 0.3
    tolerance = 1e-8
    c, A, b = random_feasible_lp(m, n, A_density)




    lp = cuPDLP.linear_programming_problem(
        -Inf * ones(n), Inf * ones(n), c, 0.0, -A, -b, 0
    )

    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params = cuPDLP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = 120,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    params = cuPDLP.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        true,
        2,
        true,
        40,
        termination_params,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6),  
    )

    println("measuring gpu")

    gpu_time = @elapsed begin
        output::cuPDLP.SaddlePointOutput = cuPDLP.optimize(params, lp)
    end 
    println("GPU time = $gpu_time")



    # model = Model(Mosek.Optimizer)
    # set_attribute(model, "MSK_IPAR_INTPNT_BASIS", 0)
    model = Model(Clarabel.Optimizer)
    set_optimizer_attribute(model, "verbose", true)
    @variable(model, x[1:n])
    @objective(model, Min, transpose(c)*x)
    @constraint(model, A*x <= b)


    cpu_time = @elapsed begin
        optimize!(model)
    end 
    println("cpu_time = $cpu_time")
end
main()
