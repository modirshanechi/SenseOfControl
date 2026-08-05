################################################################################
# Code to correct for FDR of all hypotheses at the same time
################################################################################
using EmpHCA

# Fig 2F
pvals_2F  = [ 1e-4, 1e-4, 1e-4, 1e-4,
              1e-4, 1e-4, 1e-4, 0.004]
# Fig 2G
pvals_2G  = [1e-4]
# Fig 2H
pvals_2H = [0.002, 1e-4,
            1e-4,  0.016]

# Fig 3E
pvals_3E  = [ 1e-4, 1e-4, 1e-4, 1e-4,
              1e-4, 1e-4, 1e-4, 1e-4,
              1e-4, 1e-4, 1e-4, 1e-4,
              1e-4, 1e-4, 1e-4, 1e-4]

# Fig 4
pvals_4  = [  1e-4, 1e-4, 1e-4]

# Fig 5D
pvals_5D  = [ 1e-4, 1e-4, 1e-4, 1e-4,
              1e-4, 1e-4, 1e-4, 1e-4]
# Fig 5E
pvals_5E  = [1e-4]
# Fig 5F
pvals_5F = [1e-4, 1e-4,
            1e-4, 1e-4]

# All main figs together
pvals = vcat(pvals_2F, pvals_2G, pvals_2H,
             pvals_3E,
             pvals_4,
             pvals_5D, pvals_5E, pvals_5F)
println("--------------------------------");
println("All main figs:");
R0, argR0, pval_thresh = FDR_control_pval(pvals;FDR=0.05);
@show pval_thresh
