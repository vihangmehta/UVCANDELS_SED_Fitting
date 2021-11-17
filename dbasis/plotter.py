from useful import *

from dense_basis import calc_percentiles
from matplotlib import gridspec
from corner import hist2d
from matplotlib.ticker import MaxNLocator

def plot_prior_dist(priors,bins=40,savename=None):

    import dense_basis as db

    a,b,c,d = priors.make_N_prior_draws(size=50000, random_seed=10)
    theta = np.vstack((a[0,0:], a[1,0:], a[3:,0:], b,c,d))
    txs = ['t'+'%.0f' %i for i in db.quantile_names(priors.Nparam)]
    labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
    labels[2:2] = txs

    theta  = np.insert(theta,2,theta[1,:]-theta[0,:],axis=0)
    labels = np.insert(labels,2,"sSFR")

    fig,axes = plt.subplots(len(labels),len(labels),figsize=(18,18),dpi=75)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.06,top=0.98,wspace=0.05,hspace=0.05)
    fig.suptitle("%.2f<z<%.2f"%(priors.z_min,priors.z_max),fontsize=18,fontweight=600)

    if "sSFR" in labels:
        idx = np.where(labels=="sSFR")[0][0]
        axes[idx,idx].set_xlim(-13,-7)

    for i in range(len(labels)):
        axes[i,i].hist(theta[i],bins=bins,color='k',histtype="step",lw=1.5)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i>j:
                hist,binsx,binsy = np.histogram2d(theta[i],theta[j],bins=bins)
                bincx = 0.5*(binsx[1:]+binsx[:-1])
                bincy = 0.5*(binsy[1:]+binsy[:-1])
                hist = np.ma.masked_array(hist,mask=hist<3)
                axes[i,j].contourf(bincy,bincx,hist,color='k',cmap=plt.cm.Greys,levels=3)
                axes[i,j].set_xlim(axes[j,j].get_xlim())
                axes[i,j].set_ylim(axes[i,i].get_xlim())
            elif i==j:
                [tick.set_visible(False) for tick in axes[i,j].get_yticklabels()]
            else:
                axes[i,j].set_visible(False)

            if j!=0: [tick.set_visible(False) for tick in axes[i,j].get_yticklabels()]
            elif i!=0: axes[i,j].set_ylabel(labels[i],fontsize=18)

            if i!=len(labels)-1: [tick.set_visible(False) for tick in axes[i,j].get_xticklabels()]
            else: axes[i,j].set_xlabel(labels[j],fontsize=18)

            axes[i,j].tick_params(axis="both",which="both",direction="in",top=True,right=True)
            [tick.set_fontsize(13) for tick in axes[i,j].get_xticklabels()+axes[i,j].get_yticklabels()]

    if savename: fig.savefig(savename)
    else: return fig

def plot_best_fit_summary(ndim):

    fig = plt.figure(figsize=(18, 9), dpi=100)
    fig.subplots_adjust(left=0.05,right=0.98,bottom=0.07,top=0.98)
    ogs = gridspec.GridSpec(2, 2, wspace=0.18, hspace=0.2, height_ratios=[3,1], width_ratios=[1,1])

    igs_spc = gridspec.GridSpecFromSubplotSpec(   1,   1,ogs[0,0])
    igs_crn = gridspec.GridSpecFromSubplotSpec(ndim,ndim,ogs[:,1],hspace=0.08,wspace=0.08)
    igs_sfh = gridspec.GridSpecFromSubplotSpec(   1,   1,ogs[1,0])

    ax_spc = fig.add_subplot(igs_spc[0])
    ax_crn = np.array([[fig.add_subplot(igs_crn[i,j]) for i in range(ndim)] for j in range(ndim)])
    ax_sfh = fig.add_subplot(igs_sfh[0])

    return fig, [ax_spc, ax_crn, ax_sfh]

def plot_corner(axis, params, chi2_array, labels, truths, nbins=20, max_n_ticks=4):

    ndim = len(labels)

    for i in range(ndim):
        for j in range(ndim):

            xlim = [np.min(params[i]), np.max(params[i])]
            ylim = [np.min(params[j]), np.max(params[j])]

            if i < j:
                hist2d(params[i], params[j], ax=axis[i, j], weights=np.exp(-chi2_array/2),
                       bins=[nbins,nbins], color='k', plot_datapoints=False, fill_contours=True, smooth=1.0,
                       levels=[1 - np.exp(-(1/1)**2/2), 1 - np.exp(-(2/1)**2/2)], contour_kwargs={"linewidths":0.3})
                axis[i, j].plot(truths[i], truths[j], marker='s', markersize=3, color='tab:red', alpha=0.9)
                axis[i, j].axvline(truths[i], color="tab:red", lw=1.5, alpha=0.5)
                axis[i, j].axhline(truths[j], color="tab:red", lw=1.5, alpha=0.5)
                axis[i, j].set_xlim(xlim)
                axis[i, j].set_ylim(ylim)

            elif i == j:
                axis[i, j].hist(params[i], bins=nbins, weights=np.exp(-chi2_array/2), color='k', alpha=0.6)
                axis[i, j].axvline(truths[i], color="tab:red", lw=1.5, alpha=1.0)
                axis[i, j].set_xlim(xlim)
                [tick.set_visible(False) for tick in axis[i, j].get_yticklabels()]

            if j != ndim-1:
                [tick.set_visible(False) for tick in axis[i, j].get_xticklabels()]
            if i != 0:
                [tick.set_visible(False) for tick in axis[i, j].get_yticklabels()]

            axis[i, -1].set_xlabel(labels[i], fontsize=10)
            if j > 0:
                axis[0, j].set_ylabel(labels[j], fontsize=10)

            axis[i, j].xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            axis[i, j].yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            [tick.set_fontsize(8) for tick in axis[i,j].get_xticklabels()+axis[i,j].get_yticklabels()]

    [axis[i, j].set_visible(False) for i in range(ndim)
     for j in range(ndim) if i > j]


def plot_spec(axis, filt_centers, sed, sed_err, fit_mask, norm_fac, model_sed, lam_all, spec_all, z_all, wht_all, alpha=0.3):

    for i in range(len(z_all)):
        a = alpha * wht_all[i] / wht_all[0]
        axis.plot(lam_all[i]*(1+z_all[i]), spec_all[i]*norm_fac, color='k', lw=1.5, alpha=a, zorder=1)
    # axis.plot(lam_all[0]*(1+z_all[0]), spec_all[0]*norm_fac, color='tab:red', lw=2, alpha=0.9, zorder=5)

    cond = (sed>0) & fit_mask
    axis.errorbar(filt_centers[sed>0], sed[sed>0], yerr=sed_err[sed>0] * 2,
                    markerfacecolor="none", markeredgecolor="tab:blue",
                    lw=0, elinewidth=0.7, marker='s', markersize=8, capsize=3, zorder=7)
    axis.errorbar(filt_centers[cond], sed[cond], yerr=sed_err[cond] * 2,
                    color="tab:blue", lw=0, elinewidth=1.5, marker='s', markersize=8, capsize=3, zorder=8)
    # axis.plot(filt_centers[cond], model_sed[cond]*norm_fac,
    #                 markerfacecolor="none", markeredgecolor="tab:red",
    #                 lw=0, marker='o', markersize=10, mew=2, alpha=0.9, zorder=10)

    xlim = np.array([np.min(filt_centers)*0.8, np.max(filt_centers)*1.5])
    ylim = np.array([sed[sed>0].min()/10, sed[sed>0].max()*80])

    axis.set_xlabel('Wavelength [$\\AA$]', fontsize=15)
    axis.set_ylabel('Flux Density [$\mu$Jy]', fontsize=15)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xscale("log")
    axis.set_yscale("log")

    xticks = np.array([0.3,0.4,0.6,0.5,0.8,1,1.5,2,3,4,5,6,8,10])
    axis.set_xticks(xticks*1e4)
    axis.set_xticklabels(xticks)

    twax = axis.twinx()
    twax.set_ylim(-2.5*np.log10(ylim)+23.9)
    twax.set_ylabel("Magnitudes [AB]",fontsize=15)

def plot_sfh(axis, common_time, sfh_all, wht_all, alpha=0.3):

    for i in range(len(sfh_all)):
        a = alpha * wht_all[i] / wht_all[0]
        axis.plot(np.amax(common_time)-common_time, sfh_all[i], color='k', lw=1.5, alpha=a)
    # axis.plot(np.amax(common_time)-common_time, sfh_all[0], color='tab:red', lw=2, alpha=0.9)

    sfh_16, sfh_50, sfh_84 = np.zeros((3,len(common_time)))
    for ti in range(len(common_time)):
        qty = np.log10(sfh_all[0:, ti])
        qtymask = (qty > -np.inf) & (~np.isnan(qty))
        if np.sum(qtymask) > 0:
            smallwts = wht_all.copy()[qtymask.ravel()]
            qty = qty[qtymask]
            if len(qty > 0):
                sfh_50[ti], sfh_16[ti], sfh_84[ti] = 10**calc_percentiles(qty, smallwts, bins=50, percentile_values=[50., 16., 84.])

    axis.plot(np.amax(common_time)-common_time,sfh_50, lw=2, color='k', alpha=0.9)
    axis.fill_between(np.amax(common_time)-common_time.ravel(),sfh_16.ravel(), sfh_84.ravel(), alpha=0.3, color='k')

    axis.set_xlabel('Lookback time [Gyr]', fontsize=15)
    axis.set_ylabel('SFR(t)', fontsize=15)
    # axis.set_ylabel('SFR(t) [M$_\odot$/yr]', fontsize=15)
    cond = np.isfinite(sfh_84)
    axis.set_ylim(0, np.amax(sfh_84[cond])*1.3)
