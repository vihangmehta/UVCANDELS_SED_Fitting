from useful import *
from utils import *
from utilsDBasis import *
from plotUtils import *

import dense_basis as db
from schwimmbad import MultiPool
from functools import partial


def plotBestFit(
    objID,
    flux,
    ferr,
    mask,
    zred,
    zerr,
    params,
    priors,
    atlas,
    saveDir=None,
    cigaleCat=None,
    cigaleDir=None,
    overwrite=False,
    nModels=25,
):

    savename = "{:d}_bestmodel.png".format(objID)
    if (not overwrite) and os.path.isfile(os.path.join(saveDir, savename)):
        print(
            "Skipping objID#%d because output exists (not overwriting)" % objID,
            flush=True,
        )
        return

    fig, axes = setupBestFitPlot(ndim=params["Npars"] + 5)

    plotObsPhotometry(axis=axes[0], obsFlux=flux, obsFerr=ferr, obsMask=mask)

    plotSavedFit(
        objID=objID,
        saveDir=saveDir,
        axisSpec=axes[0],
        axisSFH=axes[2],
        obsFlux=flux,
        obsMask=mask,
        verbose=True,
    )

    fitter = db.SedFit(
        sed=flux, sed_err=ferr, fit_mask=mask, zbest=zred, deltaz=zerr, atlas=atlas
    )

    fitter.evaluate_likelihood()
    fitter.evaluate_posterior_percentiles()
    fitter.evaluate_posterior_SFH(zval=fitter.zbest)

    print("".join(["="] * 10 + [" Fitter params "] + ["="] * 10))
    print("Galaxy ID: {:d}".format(objID))
    print("".join(["-"] * 30))
    print("Mass:", fitter.mstar)
    print("SFR:", fitter.sfr)
    print("Av:", fitter.Av)
    print("Z:", fitter.Z)
    print("zred:", fitter.z)
    print("SFH tuple:", fitter.sfh_tuple)

    plotParams = np.vstack(
        [
            atlas["mstar"],
            atlas["sfr"],
            atlas["sfh_tuple"][0:, 3:].T,
            atlas["met"].ravel(),
            atlas["dust"].ravel(),
            atlas["zval"].ravel(),
        ]
    )
    txs = ["t" + "%.0f" % i for i in quantileNames(plotParams.shape[0] - 5)]
    plotLabels = ["log M*", "log SFR", "Z", "Av", "z"]
    plotLabels[2:2] = txs
    cornerParams = plotParams.copy()
    cornerParams[0, 0:] += np.log10(fitter.norm_fac)
    cornerParams[1, 0:] += np.log10(fitter.norm_fac)

    plotCorner(
        axis=axes[1],
        params=cornerParams,
        chi2Array=fitter.chi2_array,
        labels=plotLabels,
        truths=cornerParams[:, np.argmin(fitter.chi2_array)],
    )

    bestModels = np.argsort(fitter.likelihood)[::-1]
    modWavelengths, modSpectra, modRedshifts, modSFHs, modWeights = [], [], [], [], []

    modTimeScale = np.linspace(0, db.cosmo.age(fitter.zbest).value, 100)
    if priors.dynamic_decouple == True:
        priors.decouple_sfr_time = (
            100 * db.cosmo.age(fitter.zbest).value / db.cosmo.age(0.1).value
        )

    for i in range(nModels):

        modWavelength, modSpectrum = db.makespec_atlas(
            fitter.atlas,
            bestModels[i],
            priors,
            db.mocksp,
            db.cosmo,
            filter_list=[],
            filt_dir=[],
            return_spec=True,
        )
        modWavelengths.append(modWavelength)
        modSpectra.append(modSpectrum)
        modRedshifts.append(fitter.atlas["zval"][bestModels[i]])

        sfh, timeax = db.tuple_to_sfh(
            fitter.atlas["sfh_tuple"][bestModels[i], 0:],
            fitter.atlas["zval"][bestModels[i]],
        )
        sfh = sfh * fitter.norm_fac
        sfhInterp = np.interp(modTimeScale, timeax, sfh)
        modSFHs.append(sfhInterp)
        modWeights.append(fitter.likelihood[bestModels[i]])

    plotSpectrum(
        axis=axes[0],
        obsFlux=fitter.sed,
        obsMask=fitter.fit_mask,
        normFactor=fitter.norm_fac,
        modFlux=fitter.atlas["sed"][bestModels[0]],
        modWavelengths=np.array(modWavelengths),
        modSpectra=np.array(modSpectra),
        modRedshifts=np.array(modRedshifts),
        modWeights=np.array(modWeights),
    )

    plotSFH(
        axis=axes[2],
        modTimeScale=modTimeScale,
        modSFHs=np.array(modSFHs),
        modWeights=np.array(modWeights),
    )

    for i in [1, 2]:
        fitter.mstar[i] -= fitter.mstar[0]
        fitter.sfr[i] -= fitter.sfr[0]
        fitter.Av[i] -= fitter.Av[0]
        fitter.Z[i] -= fitter.Z[0]

    summary = "ID: {:d}\n" "z$_{{phot}}$ = {:.3f}\n".format(objID, zred)
    bestfit = (
        "log(M$^{{*}}$) = {0[0]:.3f}$_{{{0[1]:.3f}}}^{{+{0[2]:.3f}}}$ M$_\\odot$\n"
        "log(SFR) = {1[0]:.2f}$_{{{1[1]:.2f}}}^{{+{1[2]:.2f}}}$ M$_\\odot$/yr\n"
        "log(sSFR) = {4:.2f} yr$^{{-1}}$\n"
        "A$_{{V}}$ = {2[0]:.3f}$_{{{2[1]:.3f}}}^{{+{2[2]:.3f}}}$\n"
        "log(Z/Z$_\\odot$) = {3[0]:.3f}$_{{{3[1]:.3f}}}^{{+{3[2]:.3f}}}$\n".format(
            fitter.mstar,
            fitter.sfr,
            fitter.Av,
            fitter.Z,
            fitter.sfr[0] - fitter.mstar[0],
        )
    )
    fig.text(0.8, 0.98, summary, fontsize=18, fontweight=600, va="top", ha="left")
    fig.text(0.8, 0.90, bestfit, fontsize=16, fontweight=400, va="top", ha="left")

    if cigaleCat is not None:

        entry = cigaleCat[cigaleCat["ID"] == objID][0]

        cigaleFit, cigaleSFH = getCigaleBestFit(
            field=field, objID=objID, cigaleDir=cigaleDir
        )

        axes[0].plot(
            cigaleFit["wavelength"] * 10,
            cigaleFit["Fnu"] * 1e3,
            color="tab:green",
            lw=2.5,
            alpha=0.9,
            zorder=5,
        )

        wcen, msed = [], []
        for x in cigaleCat.dtype.names:
            if "best.uvc." in x:
                wcen.append(filterSet.pivot[x.split(".")[-1]])
                msed.append(entry[x] * 1e3)

        axes[0].plot(
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

        axes[2].plot(
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
                np.log10(entry["best.sfh.sfr"])
                - np.log10(entry["best.stellar.m_star"]),
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

    if saveDir:
        fig.savefig(os.path.join(saveDir, savename))
        plt.close(fig)
    else:
        plt.show()


def worker(
    catalogIndex,
    catalog,
    obscat,
    params,
    priors,
    atlas,
    saveDir,
    cigaleCat,
    cigaleDir,
    overwrite,
):

    parsDict = getObsCatParsDict(
        catalog=catalog, obscat=obscat, catalogIndex=catalogIndex
    )

    plotBestFit(
        **parsDict,
        params=params,
        priors=priors,
        atlas=atlas,
        saveDir=saveDir,
        cigaleCat=cigaleCat,
        cigaleDir=cigaleDir,
        overwrite=overwrite
    )


def main(field, catalog, params, cigaleCat, cigaleDir, overwrite=False):

    obscat = setupDBasisInput(catalog=catalog, redshiftErrScale=0.025)

    catalogIndex = np.arange(len(obscat["zred"]))
    redshiftIndexs = getRedshiftIndex(params=params, zred=obscat["zred"])

    for redshiftIndex in np.unique(redshiftIndexs):

        priors = getPriors(
            Npars=params["Npars"],
            zmin=params["zbins"][redshiftIndex][0],
            zmax=params["zbins"][redshiftIndex][1],
        )

        atlas = db.load_atlas(
            getPregridName(
                params=params,
                z0=params["zbins"][redshiftIndex][0],
                z1=params["zbins"][redshiftIndex][1],
                returnFull=False,
            ),
            path=pregrid_path,
            N_pregrid=params["Npts"],
            N_param=params["Npars"],
        )

        with MultiPool(processes=35) as pool:
            pool.map(
                partial(
                    worker,
                    catalog=catalog,
                    obscat=obscat,
                    params=params,
                    priors=priors,
                    atlas=atlas,
                    saveDir=os.path.join(outputPath, field),
                    cigaleCat=cigaleCat,
                    cigaleDir=cigaleDir,
                    overwrite=overwrite,
                ),
                catalogIndexs[redshiftIndexs == redshiftIndex],
            )


def test(catalogIndex, field, catalog, params, cigaleCat, cigaleDir, overwrite=False):

    import dense_basis as db

    obscat = setupDBasisInput(catalog=catalog, redshiftErrScale=0.025)

    redshiftIndex = getRedshiftIndex(params=params, zred=obscat["zred"][catalogIndex])

    priors = getPriors(
        Npars=params["Npars"],
        zmin=params["zbins"][redshiftIndex][0],
        zmax=params["zbins"][redshiftIndex][1],
    )

    atlas = db.load_atlas(
        getPregridName(
            params=params,
            z0=params["zbins"][redshiftIndex][0],
            z1=params["zbins"][redshiftIndex][1],
            returnFull=False,
        ),
        path=pregridPath,
        N_pregrid=params["Npts"],
        N_param=params["Npars"],
    )

    worker(
        catalogIndex=catalogIndex,
        catalog=catalog,
        obscat=obscat,
        params=params,
        priors=priors,
        atlas=atlas,
        saveDir=os.path.join(outputPath, field),
        cigaleCat=cigaleCat,
        cigaleDir=cigaleDir,
        overwrite=overwrite,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--field", type=str, help="specify the field", default=None
    )
    parser.add_argument(
        "-n", "--nproc", type=int, help="specify number of processes", default=35
    )
    parser.add_argument("-t", "--test", help="run test case", action="store_true")
    parser.add_argument(
        "-o", "--overwrite", help="clobber existing output", action="store_true"
    )

    args = parser.parse_args()
    if args.field not in fields:
        raise Exception("Invalid field selected.")

    params = getParams(runVersion="v1")
    catalog = getCatalog(field=args.field)
    cigaleCat = "uvc_v1_%s_cigale.fits" % args.field
    cigaleDir = os.path.join(cwd, "cigale", "output", args.field)

    if args.test:

        test(
            catalogIndex=16553,
            field=args.field,
            catalog=catalog,
            params=params,
            cigaleCat=cigaleCat,
            cigaleDir=cigaleDir,
            overwrite=args.overwrite,
        )

    else:

        main(
            field=args.field,
            catalog=catalog,
            params=params,
            cigaleCat=cigaleCat,
            cigaleDir=cigaleDir,
            nproc=args.nproc,
            overwrite=args.overwrite,
        )
