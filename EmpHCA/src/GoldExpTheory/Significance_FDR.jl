################################################################################
# Code to correct for FDR of all hypotheses at the same time
################################################################################
using EmpHCA

# Fig 2 G
pvals_2G = [0.003, 1e-4,  0.19,
            1e-4,  0.016, 0.083]
# Fig 3 outset
pvals_3out  = [ 1e-4, 1e-4, 1e-4,
                0.24, 1e-4, 0.06]
# Fig 3 inset
pvals_3in  = [  1e-4, 1e-3]
# Fig 4 F
pvals_4F  = [1e-4]
# Fig 5 outset
pvals_5out  = [ 1e-4, 1e-4, 1e-4,
                1e-4, 1e-4, 1e-4,
                1e-4, 1e-4, 0.01,
                0.07, 1e-4, 0.18,]
# Fig 5 inset
pvals_5in  = [  1e-4, 1e-3, 1e-3, 1e-4]
# Fig 6 E
pvals_6E = [1e-4, 1e-4, 0.007,
            1e-4, 1e-4, 0.58]
# Fig 7 outset
pvals_7out  = [ 1e-4, 1e-4, 1e-4,
                0.22, 1e-4, 0.63]
# Fig 7 inset
pvals_7in  = [  1e-4, 1e-4]

# All main figs together
pvals = vcat(pvals_2G, pvals_3out, pvals_3in, 
             pvals_4F, pvals_5out, pvals_5in, 
             pvals_6E, pvals_7out, pvals_7in)
println("--------------------------------");
println("All main figs:");
R0, argR0, pval_thresh = FDR_control_pval(pvals;FDR=0.05);
@show pval_thresh
