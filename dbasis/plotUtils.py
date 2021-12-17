from useful import *
from utils import *
from utilsDBasis import *

from dense_basis import calc_percentiles
from matplotlib import gridspec
from corner import hist2d
from matplotlib.ticker import MaxNLocator

filtCenters = np.array([filterSet.pivot[x] for x in filterSet.filters])


def quantileNames(N_params):
    return (np.round(np.linspace(0, 100, N_params + 2)))[1:-1]


def plotPriorDist(priors, bins=40, savename=None):

    import dense_basis as db

    a, b, c, d = priors.make_N_prior_draws(size=50000, random_seed=10)
    theta = np.vstack((a[0, 0:], a[1, 0:], a[3:, 0:], b, c, d))
    txs = ["t" + "%.0f" % i for i in db.quantile_names(priors.Nparam)]
    labels = ["log M*", "log SFR", "Z", "Av", "z"]
    labels[2:2] = txs

    theta = np.insert(theta, 2, theta[1, :] - theta[0, :], axis=0)
    labels = np.insert(labels, 2, "sSFR")

    fig, axes = plt.subplots(len(labels), len(labels), figsize=(18, 18), dpi=75)

    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.06, top=0.98, wspace=0.05, hspace=0.05
    )

    fig.suptitle(
        "%.2f<z<%.2f" % (priors.z_min, priors.z_max), fontsize=18, fontweight=600
    )

    if "sSFR" in labels:
        idx = np.where(labels == "sSFR")[0][0]
        axes[idx, idx].set_xlim(-13, -7)

    for i in range(len(labels)):
        axes[i, i].hist(theta[i], bins=bins, color="k", histtype="step", lw=1.5)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i > j:
                hist, binsx, binsy = np.histogram2d(theta[i], theta[j], bins=bins)
                bincx = 0.5 * (binsx[1:] + binsx[:-1])
                bincy = 0.5 * (binsy[1:] + binsy[:-1])
                hist = np.ma.masked_array(hist, mask=hist < 3)
                axes[i, j].contourf(
                    bincy, bincx, hist, color="k", cmap=plt.cm.Greys, levels=3
                )
                axes[i, j].set_xlim(axes[j, j].get_xlim())
                axes[i, j].set_ylim(axes[i, i].get_xlim())
            elif i == j:
                [tick.set_visible(False) for tick in axes[i, j].get_yticklabels()]
            else:
                axes[i, j].set_visible(False)

            if j != 0:
                [tick.set_visible(False) for tick in axes[i, j].get_yticklabels()]
            elif i != 0:
                axes[i, j].set_ylabel(labels[i], fontsize=18)

            if i != len(labels) - 1:
                [tick.set_visible(False) for tick in axes[i, j].get_xticklabels()]
            else:
                axes[i, j].set_xlabel(labels[j], fontsize=18)

            axes[i, j].tick_params(
                axis="both", which="both", direction="in", top=True, right=True
            )
            [
                tick.set_fontsize(13)
                for tick in axes[i, j].get_xticklabels() + axes[i, j].get_yticklabels()
            ]

    if savename:
        fig.savefig(savename)
    else:
        return fig


def setupBestFitPlot(ndim):

    fig = plt.figure(figsize=(18, 9), dpi=100)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.98)
    ogs = gridspec.GridSpec(
        2, 2, wspace=0.18, hspace=0.2, height_ratios=[3, 1], width_ratios=[1, 1]
    )

    igs_spc = gridspec.GridSpecFromSubplotSpec(1, 1, ogs[0, 0])
    igs_crn = gridspec.GridSpecFromSubplotSpec(
        ndim, ndim, ogs[:, 1], hspace=0.08, wspace=0.08
    )
    igs_sfh = gridspec.GridSpecFromSubplotSpec(1, 1, ogs[1, 0])

    ax_spc = fig.add_subplot(igs_spc[0])
    ax_crn = np.array(
        [[fig.add_subplot(igs_crn[i, j]) for i in range(ndim)] for j in range(ndim)]
    )
    ax_sfh = fig.add_subplot(igs_sfh[0])

    return fig, [ax_spc, ax_sfh, ax_crn]


def plotCorner(
    axis,
    params,
    labels,
    truths=None,
    color="k",
    chi2Array=None,
    truthsOnly=False,
    nbins=20,
    maxTicks=4,
):

    ndim = len(labels)

    for i in range(ndim):
        for j in range(ndim):

            xlim = [np.min(params[i]), np.max(params[i])]
            ylim = [np.min(params[j]), np.max(params[j])]

            if i < j:
                if not truthsOnly:
                    hist2d(
                        params[i],
                        params[j],
                        ax=axis[i, j],
                        weights=np.exp(-chi2Array / 2),
                        bins=[nbins, nbins],
                        color="k",
                        plot_datapoints=False,
                        fill_contours=True,
                        smooth=1.0,
                        levels=[
                            1 - np.exp(-((1 / 1) ** 2) / 2),
                            1 - np.exp(-((2 / 1) ** 2) / 2),
                        ],
                        contour_kwargs={"linewidths": 0.3},
                    )
                if truths is not None:
                    axis[i, j].plot(
                        truths[i],
                        truths[j],
                        marker="s",
                        markersize=3,
                        color="tab:red",
                        alpha=0.9,
                    )
                    axis[i, j].axvline(truths[i], color=color, lw=1.5, alpha=0.5)
                    axis[i, j].axhline(truths[j], color=color, lw=1.5, alpha=0.5)
                axis[i, j].set_xlim(xlim)
                axis[i, j].set_ylim(ylim)

            elif i == j:
                if not truthsOnly:
                    axis[i, j].hist(
                        params[i],
                        bins=nbins,
                        weights=np.exp(-chi2Array / 2),
                        color="k",
                        alpha=0.6,
                    )
                if truths is not None:
                    axis[i, j].axvline(truths[i], color=color, lw=1.5, alpha=1.0)
                axis[i, j].set_xlim(xlim)
                [tick.set_visible(False) for tick in axis[i, j].get_yticklabels()]

            if j != ndim - 1:
                [tick.set_visible(False) for tick in axis[i, j].get_xticklabels()]
            if i != 0:
                [tick.set_visible(False) for tick in axis[i, j].get_yticklabels()]

            axis[i, -1].set_xlabel(labels[i], fontsize=10)
            if j > 0:
                axis[0, j].set_ylabel(labels[j], fontsize=10)

            axis[i, j].xaxis.set_major_locator(MaxNLocator(maxTicks, prune="lower"))
            axis[i, j].yaxis.set_major_locator(MaxNLocator(maxTicks, prune="lower"))
            [
                tick.set_fontsize(8)
                for tick in axis[i, j].get_xticklabels() + axis[i, j].get_yticklabels()
            ]

    [axis[i, j].set_visible(False) for i in range(ndim) for j in range(ndim) if i > j]


def plotObsPhotometry(axis, obsFlux, obsFerr, obsMask):

    cond = (obsFlux > 0) & obsMask
    axis.errorbar(
        filtCenters[obsFlux > 0],
        obsFlux[obsFlux > 0],
        yerr=obsFerr[obsFlux > 0] * 2,
        markerfacecolor="none",
        markeredgecolor="tab:blue",
        lw=0,
        elinewidth=0.7,
        marker="s",
        markersize=8,
        capsize=3,
        zorder=7,
    )

    axis.errorbar(
        filtCenters[cond],
        obsFlux[cond],
        yerr=obsFerr[cond] * 2,
        color="tab:blue",
        lw=0,
        elinewidth=1.5,
        marker="s",
        markersize=8,
        capsize=3,
        zorder=8,
    )

    xlim = np.array([np.min(filtCenters) * 0.8, np.max(filtCenters) * 1.5])
    ylim = np.array([obsFlux[obsFlux > 0].min() / 10, obsFlux[obsFlux > 0].max() * 80])

    axis.set_xlabel("Wavelength [$\\AA$]", fontsize=15)
    axis.set_ylabel("Flux Density [$\mu$Jy]", fontsize=15)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xscale("log")
    axis.set_yscale("log")

    xticks = np.array([0.3, 0.4, 0.6, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 8, 10])
    axis.set_xticks(xticks * 1e4)
    axis.set_xticklabels(xticks)

    twax = axis.twinx()
    twax.set_ylim(-2.5 * np.log10(ylim) + 23.9)
    twax.set_ylabel("Magnitudes [AB]", fontsize=15)


def plotModel(modelMags, model, sfh, axisSpec, axisSFH, obsFlux, obsMask, color):

    cond = (obsFlux > 0) & obsMask
    modFlux = np.array(
        [modelMags["model_Fnu_{:s}".format(filt)] for filt in filterSet.filters]
    )

    axisSpec.plot(
        filtCenters[cond],
        modFlux[cond],
        markerfacecolor="none",
        markeredgecolor=color,
        lw=0,
        marker="o",
        markersize=10,
        mew=2,
        alpha=0.9,
        zorder=10,
    )

    axisSpec.plot(
        model["obswave"], model["Fnu"], color=color, lw=2, alpha=0.9, zorder=5
    )

    axisSFH.plot(sfh["time"], sfh["sfr"], color=color, lw=2, alpha=0.9)

    axisSFH.set_xlabel("Lookback time [Gyr]", fontsize=15)
    axisSFH.set_ylabel("SFR(t)", fontsize=15)


def getCigaleBestFit(field, objID, cigaleDir):

    fitFile = os.path.join(cigaleDir, "{:d}_best_model.fits".format(objID))
    sfhFile = os.path.join(cigaleDir, "{:d}_SFH.fits".format(objID))

    fit = fits.getdata(fitFile)
    sfh = fits.getdata(sfhFile)

    return fit, sfh


def plotCigaleFit(objID, cigaleCat, cigaleDir, axisSpec, axisSFH, axisCorner):

    entry = cigaleCat[cigaleCat["ID"] == objID][0]

    cigaleFit, cigaleSFH = getCigaleBestFit(
        field=field, objID=objID, cigaleDir=cigaleDir
    )

    axisSpec.plot(
        cigaleFit["wavelength"] * 10,
        cigaleFit["Fnu"] * 1e3,
        color="tab:green",
        lw=2.5,
        alpha=0.9,
        zorder=4,
    )

    wcen, msed = [], []
    for x in cigaleCat.dtype.names:
        if "best.uvc." in x:
            wcen.append(filterSet.pivot[x.split(".")[-1]])
            msed.append(entry[x] * 1e3)

    axisSpec.plot(
        wcen,
        msed,
        markerfacecolor="none",
        markeredgecolor="tab:green",
        lw=0,
        marker="o",
        markersize=10,
        mew=2,
        alpha=0.9,
        zorder=9,
    )

    axisSFH.plot(
        (entry["best.sfh.age"] - cigaleSFH["time"]) * 1e-3,
        cigaleSFH["SFR"],
        color="tab:green",
        lw=2.5,
        alpha=0.9,
    )

    bestfit = (
        "CIGALE Best-fit\n"
        "log(M$^{{*}}$) = {0:.3f} M$_\\odot$\n"
        "log(SFR) = {1:.2f} M$_\\odot$/yr\n"
        "log(sSFR) = {2:.2f} yr$^{{-1}}$\n"
        "log(age) = {3:.3f} Gyr\n"
        "log(tau) = {4:.3f} Gyr\n"
        "$f_{{burst}}$ = {5:.1f}\n"
        "A$_{{V}}$ = {6:.3f}\n"
        "log(Z/Z$_\\odot$) = {7:.3f}\n".format(
            np.log10(entry["best.stellar.m_star"]),
            np.log10(entry["best.sfh.sfr"]),
            np.log10(entry["best.sfh.sfr"]) - np.log10(entry["best.stellar.m_star"]),
            entry["best.sfh.age"] * 1e-3,
            entry["best.sfh.tau_main"] * 1e-3,
            entry["best.sfh.f_burst"],
            entry["best.attenuation.Av_ISM"],
            np.log10(entry["best.stellar.metallicity"] / 0.02),
        )
    )
    fig.text(
        0.835,
        0.68,
        bestfit,
        fontsize=15,
        fontweight=400,
        color="tab:green",
        va="top",
        ha="left",
    )


# def plotSavedFit(objID,
#                  saveDir,
#                  axisSpec,
#                  axisSFH,
#                  obsFlux,
#                  obsMask,
#                  verbose=False):

#     bestfit = fits.open(
#         os.path.join(saveDir, "{:d}_best_model.fits.gz".format(objID)))

#     fitParams = bestfit["FIT_PARAMS"].data[0]
#     print("".join(["="] * 10 + [" Best-fit params "] + ["="] * 10))
#     print("Galaxy ID: {:d}".format(objID))
#     print("".join(["-"] * 30))
#     print("Mass:", fitParams["logM"])
#     print("SFR:", fitParams["logSFR100"])
#     print("Av:", fitParams["Av"])
#     print("Z:", fitParams["logZsol"])
#     print("zred:", fitParams["zfit"])
#     print("SFH tuple:", [
#         fitParams["t20"], fitParams["t40"], fitParams["t60"], fitParams["t80"]
#     ])

#     axisSpec.plot(bestfit["MODEL_BEST"].data["obswave"],
#                   bestfit["MODEL_BEST"].data["Fnu"],
#                   color='tab:red',
#                   lw=2,
#                   alpha=0.9,
#                   zorder=6)
#     axisSpec.plot(bestfit["MODEL_CHI2"].data["obswave"],
#                   bestfit["MODEL_CHI2"].data["Fnu"],
#                   color='tab:orange',
#                   lw=2,
#                   alpha=0.9,
#                   zorder=6)

#     cond = (obsFlux > 0) & obsMask
#     modFlux = np.array([
#         fitParams["model_Fnu_{:s}".format(filt)] for filt in filterSet.filters
#     ])
#     axisSpec.plot(filtCenters[cond],
#                   modFlux[cond],
#                   markerfacecolor="none",
#                   markeredgecolor="tab:red",
#                   lw=0,
#                   marker='o',
#                   markersize=10,
#                   mew=2,
#                   alpha=0.9,
#                   zorder=10)

#     axisSFH.plot(bestfit["SFH_BEST"].data["time"],
#                  bestfit["SFH_BEST"].data["sfr"],
#                  lw=2,
#                  color='tab:red',
#                  alpha=0.9)
#     axisSFH.plot(bestfit["SFH_CHI2"].data["time"],
#                  bestfit["SFH_CHI2"].data["sfr"],
#                  lw=2,
#                  color='tab:orange',
#                  alpha=0.9)

# def plotSpectrum(axis,
#                  obsFlux,
#                  obsMask,
#                  normFactor,
#                  modFlux,
#                  modWavelengths,
#                  modSpectra,
#                  modRedshifts,
#                  modWeights,
#                  alpha=0.5,
#                  delAlpha=0.4):

#     for i in range(len(modRedshifts)):
#         a = alpha - delAlpha * (i / len(modRedshifts))
#         axis.plot(modWavelengths[i] * (1 + modRedshifts[i]),
#                   modSpectra[i] * normFactor,
#                   color='k',
#                   lw=1.5,
#                   alpha=a,
#                   zorder=1)
#     axis.plot(modWavelengths[0] * (1 + modRedshifts[0]),
#               modSpectra[0] * normFactor,
#               color='tab:purple',
#               lw=2,
#               alpha=0.9,
#               zorder=5)

#     cond = (obsFlux > 0) & obsMask
#     axis.plot(filtCenters[cond],
#               modFlux[cond] * normFactor,
#               markerfacecolor="none",
#               markeredgecolor="tab:purple",
#               lw=0,
#               marker='o',
#               markersize=10,
#               mew=2,
#               alpha=0.9,
#               zorder=10)

# def plotSFH(axis, modTimeScale, modSFHs, modWeights, alpha=0.3):

#     for i in range(len(modSFHs)):
#         a = alpha * modWeights[i] / modWeights[0]
#         axis.plot(np.amax(modTimeScale) - modTimeScale,
#                   modSFHs[i],
#                   color='k',
#                   lw=1.5,
#                   alpha=a)
#     axis.plot(np.amax(modTimeScale) - modTimeScale,
#               modSFHs[0],
#               color='tab:purple',
#               lw=2,
#               alpha=0.9)

#     sfh_16, sfh_50, sfh_84 = np.zeros((3, len(modTimeScale)))
#     for ti in range(len(modTimeScale)):
#         qty = np.log10(modSFHs[0:, ti])
#         qtymask = (qty > -np.inf) & (~np.isnan(qty))
#         if np.sum(qtymask) > 0:
#             smallwts = modWeights.copy()[qtymask.ravel()]
#             qty = qty[qtymask]
#             if len(qty > 0):
#                 sfh_50[ti], sfh_16[ti], sfh_84[ti] = 10**calc_percentiles(
#                     qty, smallwts, bins=50, percentile_values=[50., 16., 84.])

#     axis.plot(np.amax(modTimeScale) - modTimeScale,
#               sfh_50,
#               lw=2,
#               color='k',
#               alpha=0.9)
#     axis.fill_between(np.amax(modTimeScale) - modTimeScale.ravel(),
#                       sfh_16.ravel(),
#                       sfh_84.ravel(),
#                       alpha=0.3,
#                       color='k')

#     axis.set_xlabel('Lookback time [Gyr]', fontsize=15)
#     axis.set_ylabel('SFR(t)', fontsize=15)
#     # axis.set_ylabel('SFR(t) [M$_\odot$/yr]', fontsize=15)
#     cond = np.isfinite(sfh_84)
#     axis.set_ylim(0, np.amax(sfh_84[cond]) * 1.3)
