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

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98, wspace=0.05, hspace=0.05)

    fig.suptitle("%.2f<z<%.2f" % (priors.z_min, priors.z_max), fontsize=18, fontweight=600)

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
                axes[i, j].contourf(bincy, bincx, hist, color="k", cmap=plt.cm.Greys, levels=3)
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

            axes[i, j].tick_params(axis="both", which="both", direction="in", top=True, right=True)
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
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.07, top=0.98)
    ogs = gridspec.GridSpec(
        2, 2, wspace=0.2, hspace=0.18, height_ratios=[3, 1], width_ratios=[1, 1]
    )

    igs_spc = gridspec.GridSpecFromSubplotSpec(1, 1, ogs[0, 0])
    igs_crn = gridspec.GridSpecFromSubplotSpec(ndim, ndim, ogs[:, 1], hspace=0.08, wspace=0.08)
    igs_sfh = gridspec.GridSpecFromSubplotSpec(1, 1, ogs[1, 0])

    ax_spc = fig.add_subplot(igs_spc[0])
    ax_crn = np.array([[fig.add_subplot(igs_crn[i, j]) for i in range(ndim)] for j in range(ndim)])
    ax_sfh = fig.add_subplot(igs_sfh[0])

    return fig, [ax_spc, ax_sfh, ax_crn]


def plotCornerTruths(axis, truths, labels, color):

    ndim = len(labels)
    for i in range(ndim):
        for j in range(ndim):
            if i < j:
                axis[i, j].plot(
                    truths[i],
                    truths[j],
                    marker="s",
                    markersize=3,
                    color=color,
                    alpha=0.9,
                )
                axis[i, j].axvline(truths[i], color=color, lw=1.5, alpha=0.5)
                axis[i, j].axhline(truths[j], color=color, lw=1.5, alpha=0.5)
            elif i == j:
                axis[i, j].axvline(truths[i], color=color, lw=1.5, alpha=1.0)


def plotCorner(
    axis,
    params,
    labels,
    color="k",
    chi2Array=None,
    nbins=25,
    maxTicks=4,
):

    ndim = len(labels)
    whts = np.clip(np.exp(-chi2Array / 2), 1e-10, np.inf)

    for i in range(ndim):

        for j in range(ndim):

            xlim = [np.min(params[i]), np.max(params[i])]
            ylim = [np.min(params[j]), np.max(params[j])]

            if i < j:

                hist2d(
                    params[i],
                    params[j],
                    ax=axis[i, j],
                    weights=whts,
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

                axis[i, j].set_xlim(xlim)
                axis[i, j].set_ylim(ylim)

            elif i == j:

                axis[i, j].hist(
                    params[i],
                    bins=nbins,
                    weights=whts,
                    color="k",
                    alpha=0.6,
                )

                axis[i, j].set_xlim(xlim)
                [tick.set_visible(False) for tick in axis[i, j].get_yticklabels()]

            if j != ndim - 1:
                [tick.set_visible(False) for tick in axis[i, j].get_xticklabels()]
            if i != 0:
                [tick.set_visible(False) for tick in axis[i, j].get_yticklabels()]

            axis[i, -1].set_xlabel(labels[i], fontsize=14)
            if j > 0:
                axis[0, j].set_ylabel(labels[j], fontsize=14)

            axis[i, j].xaxis.set_major_locator(MaxNLocator(maxTicks, prune="lower"))
            axis[i, j].yaxis.set_major_locator(MaxNLocator(maxTicks, prune="lower"))
            [
                tick.set_fontsize(10)
                for tick in axis[i, j].get_xticklabels() + axis[i, j].get_yticklabels()
            ]
            [tick.set_rotation(45) for tick in axis[i, j].get_xticklabels()+axis[i, j].get_yticklabels()]

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
    ylim = np.array([obsFlux[obsFlux > 0].min() / 3, obsFlux[obsFlux > 0].max() * 50])

    axis.set_xlabel("Wavelength [$\\AA$]", fontsize=20)
    axis.set_ylabel("Flux Density [$\mu$Jy]", fontsize=20)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xscale("log")
    axis.set_yscale("log")

    xticks = np.array([0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10], dtype="U3")
    axis.set_xticks(xticks.astype(float) * 1e4)
    axis.set_xticklabels(xticks)
    axis.grid(linestyle=":", linewidth=0.5, color="k", which="both")

    twax = axis.twinx()
    twax.set_ylim(-2.5 * np.log10(ylim) + 23.9)
    twax.set_ylabel("Magnitudes [AB]", fontsize=20)

    [
        tick.set_fontsize(16)
        for tick in axis.get_xticklabels() + axis.get_yticklabels() + twax.get_yticklabels()
    ]


def plotModel(modelMags, model, sfh, axisSpec, axisSFH, obsFlux, obsMask, color):

    cond = (obsFlux > 0) & obsMask
    modFlux = np.array([modelMags["model_Fnu_{:s}".format(filt)] for filt in filterSet.filters])

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

    axisSpec.plot(model["obswave"], model["Fnu"], color=color, lw=2, alpha=0.9, zorder=5)

    if sfh is not None:

        axisSFH.plot(sfh["time"], sfh["sfr"], color=color, lw=2, alpha=0.9)
        axisSFH.set_xlabel("Lookback time [Gyr]", fontsize=20)
        axisSFH.set_ylabel("SFR(t)", fontsize=20)
        axisSFH.grid(linestyle=":", linewidth=0.5, color="k", which="both")
        [tick.set_fontsize(16) for tick in axisSFH.get_xticklabels() + axisSFH.get_yticklabels()]


def getCigaleBestFit(objID, cigaleDir):

    fitFile = os.path.join(cigaleDir, "{:d}_bestmodel_cigale.fits.gz".format(objID))
    sfhFile = os.path.join(cigaleDir, "{:d}_bestSFH_cigale.fits.gz".format(objID))

    fit = fits.getdata(fitFile)
    sfh = fits.getdata(sfhFile)

    return fit, sfh


def plotCigaleFit(
    objID,
    cigaleCat,
    cigaleDir,
    axisSpec,
    axisSFH,
    axisCorner,
    obsFlux,
    obsMask,
    plotLabels,
    color="tab:green",
):

    if isinstance(cigaleCat, str):
        cigaleCat = fits.getdata(cigaleCat)
    entry = cigaleCat[cigaleCat["ID"] == objID][0]

    cigaleFit, cigaleSFH = getCigaleBestFit(objID=objID, cigaleDir=cigaleDir)

    axisSpec.plot(
        cigaleFit["wavelength"] * 10,
        cigaleFit["Fnu"] * 1e3,
        color="tab:green",
        lw=2.5,
        alpha=0.9,
        zorder=4,
    )

    cond = (obsFlux > 0) & obsMask
    wcen, msed = [], []
    for x in cigaleCat.dtype.names:
        if "best.uvc." in x:
            wcen = np.append(wcen, filterSet.pivot[x.split(".")[-1]])
            msed = np.append(msed, entry[x] * 1e3)

    axisSpec.plot(
        wcen[cond],
        msed[cond],
        markerfacecolor="none",
        markeredgecolor=color,
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
        color=color,
        lw=2.5,
        alpha=0.9,
    )

    truths = [np.nan] * len(plotLabels)
    truths[0] = np.log10(entry["best.stellar.m_star"])
    truths[1] = np.log10(entry["best.sfh.sfr"])
    truths[6] = np.log10(entry["best.stellar.metallicity"] / 0.02)
    truths[7] = entry["best.attenuation.E_BVs.stellar.young"] * 4.05
    truths[8] = entry["best.universe.redshift"]
    plotCornerTruths(
        axis=axisCorner,
        labels=plotLabels,
        truths=truths,
        color=color,
    )

    bestfitCigale = (
        "$\\mathbf{{Cigale}}$\n"
        "log(M$^{{*}}$) = $\\mathbf{{{0:.3f}}}$ M$_\\odot$\n"
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
            entry["best.attenuation.E_BVs.stellar.young"] * 4.05,
            np.log10(entry["best.stellar.metallicity"] / 0.02),
        )
    )
    return bestfitCigale


mosaicnames = {
    "F275W": {
        "goodsn": "goodsn_epochALL_v1.1_F275W_60mas_sci.fits",
        "goodss": "goodss_epochALL_v1.1_F275W_60mas_sci.fits",
        "cosmos": "cosmos_epochALL_v1.1_F275W_60mas_sci.fits",
        "egs": "egs_epochALL_v1.1_F275W_60mas_sci.fits",
    },
    "F435W": {
        "goodsn": "goodsn_all_acs_wfc_f435w_060mas_v2.0_sci.fits",
        "goodss": "gs_presm4_all_acs_f435w_60mas_v3.0_sci.fits",
        "cosmos": "cosmos_epochPar_v0.3_F435W_60mas_sci.fits",
        "egs": "egs_epochPar_v0.2_F435W_60mas_sci.fits",
    },
    "F606W": {
        "goodsn": "hlsp_candels_hst_acs_gn-tot-60mas_f606w_v1.0_sci.fits",
        "goodss": "hlsp_candels_hst_acs_gs-tot-60mas_f606w_v1.0_sci.fits",
        "cosmos": "hlsp_candels_hst_acs_cos-tot-60mas_f606w_v1.0_sci.fits",
        "egs": "hlsp_candels_hst_acs_egs-tot-60mas_f606w_v1.0_sci.fits",
    },
    "F814W": {
        "goodsn": "hlsp_candels_hst_acs_gn-tot-60mas_f814w_v1.0_drz.fits",
        "goodss": "hlsp_candels_hst_acs_gs-tot_f814w_v1.0_drz.fits",
        "cosmos": "hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits",
        "egs": "hlsp_candels_hst_acs_egs-tot-60mas_f814w_v1.0_drz.fits",
    },
    "F160W": {
        "goodsn": "hlsp_candels_hst_wfc3_gn-tot-60mas_f160w_v1.0_drz.fits",
        "goodss": "hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits",
        "cosmos": "hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits",
        "egs": "hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits",
    },
}


def getStamp(field, filt, pos, size):

    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from astropy.coordinates import SkyCoord
    from astropy.nddata.utils import Cutout2D

    img, hdr = fits.getdata(
        os.path.join("/data/UVCANDELS/mosaics/", mosaicnames[filt][field]), header=True
    )

    try:
        wcs = WCS(hdr)
        wcs.sip = None
    except ValueError:
        hdr = fits.getheader(os.path.join("/data/UVCANDELS/mosaics/", mosaicnames["F160W"][field]))
        wcs = WCS(hdr)
        wcs.sip = None
        print("Using backup F160W WCS for %s" % filt)

    pxscale = proj_plane_pixel_scales(wcs)[0] * 3600
    size = size / pxscale

    pos = SkyCoord(*(pos * u.deg), frame="fk5")
    cutout = Cutout2D(img, pos, size, wcs=wcs)

    aperture = size * 2 / 3
    aper0, aper1 = size / 2.0 - aperture / 2.0, size / 2.0 + aperture / 2.0
    _stamp = cutout.data[
        int(np.floor(aper0)) : int(np.ceil(aper1)), int(np.floor(aper0)) : int(np.ceil(aper1))
    ]

    _stamp = np.ma.masked_array(_stamp, mask=~np.isfinite(_stamp))
    med, std = np.ma.median(_stamp), np.ma.std(_stamp)
    _stamp = np.ma.clip(_stamp, med - 3 * std, med + 3 * std)
    med, std = np.ma.median(_stamp), np.ma.std(_stamp)
    vmin, vmax = med - 3 * std, med + 3 * std
    return cutout.data, vmin, vmax, size


def plotStamps(fig, axSpec, ra, dec, field):

    from matplotlib.lines import Line2D

    x0, y0, dx, dy = axSpec.get_position().bounds
    x1, y1 = x0 + dx, y0 + dy
    w, h = fig.get_size_inches()

    dax1 = fig.add_axes([x1 - 0.105 * (h / w) * 5, y1 - 0.1, 0.1 * (h / w), 0.1])
    dax2 = fig.add_axes([x1 - 0.105 * (h / w) * 4, y1 - 0.1, 0.1 * (h / w), 0.1])
    dax3 = fig.add_axes([x1 - 0.105 * (h / w) * 3, y1 - 0.1, 0.1 * (h / w), 0.1])
    dax4 = fig.add_axes([x1 - 0.105 * (h / w) * 2, y1 - 0.1, 0.1 * (h / w), 0.1])
    dax5 = fig.add_axes([x1 - 0.105 * (h / w) * 1, y1 - 0.1, 0.1 * (h / w), 0.1])

    for dax, filt in zip(
        [dax1, dax2, dax3, dax4, dax5],
        ["F275W", "F435W", "F606W", "F814W", "F160W"],
    ):
        cutout, vmin, vmax, size = getStamp(field=field, filt=filt, pos=(ra, dec), size=5)

        if np.any(np.isfinite(cutout)):
            cutout = np.ma.masked_array(cutout, mask=~np.isfinite(cutout))
            dax.pcolormesh(cutout, cmap=plt.cm.Greys, vmin=vmin, vmax=vmax, rasterized=True)
            dax.text(
                0.5,
                -0.03,
                filt,
                color="k",
                fontsize=12,
                va="top",
                ha="center",
                transform=dax.transAxes,
            )

        dax.add_line(
            Line2D(
                [size / 2.0, size / 2.0],
                [size / 2 - 0.2 * size, size / 2.0 - 0.35 * size],
                lw=3,
                c="red",
            )
        )
        dax.add_line(
            Line2D(
                [size / 2.0 + 0.2 * size, size / 2.0 + 0.35 * size],
                [size / 2.0, size / 2.0],
                lw=3,
                c="red",
            ),
        )
        dax.xaxis.set_visible(False)
        dax.yaxis.set_visible(False)
        dax.set_aspect(1.0)
