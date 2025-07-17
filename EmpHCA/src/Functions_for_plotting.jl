# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for heatmap
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function PlotHeatMap(Y_plot, ax; X_names = "", Y_names = "", TextVal = true,
                        cmap="RdBu",vmin=-1,vmax=1.,
                        rotationX=0, rotationY=0)
        if isnan(vmin) || isnan(vmax)
            cp = ax.imshow(Y_plot,cmap=cmap)
        else
            cp = ax.imshow(Y_plot,cmap=cmap, vmin=vmin,vmax=vmax)
        end
        if TextVal
                for i = 1:size(Y_plot)[1]
                for j = 1:size(Y_plot)[2]
                        ax.text(j - 1, i - 1, 
                        string(round(Y_plot[i,j],digits=2)),
                        fontsize=5, horizontalalignment="center", color = "w")
                end
                end
        end
        if X_names != ""
                ax.set_xticks(0:(size(Y_plot)[2]-1)); 
                ax.set_xticklabels(X_names,fontsize=9,rotation=rotationX)
                ax.set_yticks(0:(size(Y_plot)[1]-1)); 
                ax.set_yticklabels(Y_names,fontsize=9,rotation=rotationY)
        end
        return cp                  
end
export PlotHeatMap
