from useful import *
from utils import *
from utilsDBasis import *
from plotUtils import *

from astropy.cosmology import Planck15


def calcModelBandFlux(wave, flux, band):
    if band in filterSet.filters:
        filtResp = filterSet.getResponse(band)
        filtPivot = filterSet.pivot[band]
    elif band in filterSetJohnson.filters:
        filtResp = filterSetJohnson.getResponse(band)
        filtPivot = filterSetJohnson.pivot[band]
    else:
        raise Exception("Invalid filter in calcModelBandFlux().")

    filtInterp = interp1d(
        filtResp["wave"],
        filtResp["throughput"],
        bounds_error=False,
        fill_value=0,
        kind="linear",
    )
    filtSens = filtInterp(wave)

    flux = flux * (light / wave**2)
    flux = simps(flux * filtSens * wave, wave) / simps(filtSens * wave, wave)
    flux = flux * (filtPivot**2 / light)
    return flux


def populateFitParams(results, atlas, chi2Index, normFactor, sfrErrCutoff=2.0):
    dtype = [
        ("ID", int),
        ("zbest", float),
        ("delz", float),
        ("logM", float),
        ("logM_16", float),
        ("logM_84", float),
        ("logSFR", float),
        ("logSFR_16", float),
        ("logSFR_84", float),
        ("Av", float),
        ("Av_16", float),
        ("Av_84", float),
        ("logZsol", float),
        ("logZsol_16", float),
        ("logZsol_84", float),
        ("zfit", float),
        ("zfit_16", float),
        ("zfit_84", float),
        ("logM2", float),
        ("logM2_16", float),
        ("logM2_84", float),
        ("logSFR2", float),
        ("logSFR2_16", float),
        ("logSFR2_84", float),
        ("t20", float),
        ("t20_16", float),
        ("t20_84", float),
        ("t40", float),
        ("t40_16", float),
        ("t40_84", float),
        ("t60", float),
        ("t60_16", float),
        ("t60_84", float),
        ("t80", float),
        ("t80_16", float),
        ("t80_84", float),
        ("color_nuvu", float),
        ("color_nuvu_16", float),
        ("color_nuvu_84", float),
        ("color_nuvr", float),
        ("color_nuvr_16", float),
        ("color_nuvr_84", float),
        ("color_uv", float),
        ("color_uv_16", float),
        ("color_uv_84", float),
        ("color_vj", float),
        ("color_vj_16", float),
        ("color_vj_84", float),
        ("color_rj", float),
        ("color_rj_16", float),
        ("color_rj_84", float),
        ("logM_chi2", float),
        ("logSFR_chi2", float),
        ("Av_chi2", float),
        ("logZsol_chi2", float),
        ("zfit_chi2", float),
        ("logM2_chi2", float),
        ("logSFR2_chi2", float),
        ("t20_chi2", float),
        ("t40_chi2", float),
        ("t60_chi2", float),
        ("t80_chi2", float),
        ("color_nuvu_chi2", float),
        ("color_nuvr_chi2", float),
        ("color_uv_chi2", float),
        ("color_vj_chi2", float),
        ("color_rj_chi2", float),
        ("nparam", int),
        ("nbands", int),
        ("chi2", float),
        ("flags", int),
    ]

    output = np.recarray(1, dtype=dtype)
    for x in output.dtype.names:
        output[x] = -99

    output["logM2"][0] = results["mstar"][0]
    output["logM2_16"][0] = results["mstar"][1]
    output["logM2_84"][0] = results["mstar"][2]

    output["logSFR2"][0] = results["sfr"][0]
    output["logSFR2_16"][0] = results["sfr"][1]
    output["logSFR2_84"][0] = results["sfr"][2]

    output["Av"][0] = results["Av"][0]
    output["Av_16"][0] = results["Av"][1]
    output["Av_84"][0] = results["Av"][2]

    output["logZsol"][0] = results["Z"][0]
    output["logZsol_16"][0] = results["Z"][1]
    output["logZsol_84"][0] = results["Z"][2]

    output["zfit"][0] = results["z"][0]
    output["zfit_16"][0] = results["z"][1]
    output["zfit_84"][0] = results["z"][2]

    output["logM"][0] = results["sfh"][0][0]
    output["logM_16"][0] = results["sfh"][1][0]
    output["logM_84"][0] = results["sfh"][2][0]
    output["logSFR"][0] = results["sfh"][0][1]
    output["logSFR_16"][0] = results["sfh"][1][1]
    output["logSFR_84"][0] = results["sfh"][2][1]
    output["nparam"][0] = int(results["sfh"][0][2])

    output["t20"][0] = results["sfh"][0][3]
    output["t20_16"][0] = results["sfh"][1][3]
    output["t20_84"][0] = results["sfh"][2][3]
    output["t40"][0] = results["sfh"][0][4]
    output["t40_16"][0] = results["sfh"][1][4]
    output["t40_84"][0] = results["sfh"][2][4]
    output["t60"][0] = results["sfh"][0][5]
    output["t60_16"][0] = results["sfh"][1][5]
    output["t60_84"][0] = results["sfh"][2][5]
    output["t80"][0] = results["sfh"][0][6]
    output["t80_16"][0] = results["sfh"][1][6]
    output["t80_84"][0] = results["sfh"][2][6]

    output["color_nuvu"][0] = results["nuvu"][0]
    output["color_nuvu_16"][0] = results["nuvu"][1]
    output["color_nuvu_84"][0] = results["nuvu"][2]
    output["color_nuvr"][0] = results["nuvr"][0]
    output["color_nuvr_16"][0] = results["nuvr"][1]
    output["color_nuvr_84"][0] = results["nuvr"][2]
    output["color_uv"][0] = results["uv"][0]
    output["color_uv_16"][0] = results["uv"][1]
    output["color_uv_84"][0] = results["uv"][2]
    output["color_vj"][0] = results["vj"][0]
    output["color_vj_16"][0] = results["vj"][1]
    output["color_vj_84"][0] = results["vj"][2]
    output["color_rj"][0] = results["rj"][0]
    output["color_rj_16"][0] = results["rj"][1]
    output["color_rj_84"][0] = results["rj"][2]

    output["chi2"][0] = np.amin(results["chi2"])

    output["logM_chi2"] = atlas["mstar"][chi2Index] + np.log10(normFactor)
    output["logSFR_chi2"] = atlas["sfr"][chi2Index] + np.log10(normFactor)
    output["logZsol_chi2"] = atlas["met"][chi2Index]
    output["Av_chi2"] = atlas["dust"][chi2Index]
    output["zfit_chi2"] = atlas["zval"][chi2Index]
    output["logM2_chi2"] = atlas["sfh_tuple"][chi2Index, 0] + np.log10(normFactor)
    output["logSFR2_chi2"] = atlas["sfh_tuple"][chi2Index, 1] + np.log10(normFactor)
    for i, x in enumerate(["t20", "t40", "t60", "t80"]):
        output["%s_chi2" % x] = atlas["sfh_tuple"][chi2Index, 3 + i]
    output["color_nuvu_chi2"][0] = atlas["nuvu"][chi2Index]
    output["color_nuvr_chi2"][0] = atlas["nuvr"][chi2Index]
    output["color_uv_chi2"][0] = atlas["uv"][chi2Index]
    output["color_vj_chi2"][0] = atlas["vj"][chi2Index]
    output["color_rj_chi2"][0] = atlas["rj"][chi2Index]

    # flagging galaxies that either
    # 1. have nan values for mass
    if np.isnan(output["logM"][0]):
        output["flags"][0] = 1
    # 2. have SFR uncertainties > sfrErrCutoff
    elif np.abs(output["logSFR_84"][0] - output["logSFR_16"][0]) > sfrErrCutoff:
        output["flags"][0] = 2
    # 3. are flagged as a star
    elif catalog["CLASS_STAR"][0] > 0.5:
        output["flags"][0] = 3
    # 4. have extremely large chi2
    elif output["chi2"][0] > 1000:
        output["flags"][0] = 4
    else:
        output["flags"][0] = 0

    return output


def populateModelMags(model, zbest):
    dtype = [("ID", int), ("zbest", float)]

    for band in filterSetJohnson.filters:
        dtype.append(("model_Lnu_rest{:s}".format(band), float))

    for filt in filterSet.filters:
        dtype.append(("model_Fnu_{:s}".format(filt), float))

    output = np.recarray(1, dtype=dtype)

    for x in output.dtype.names:
        output[x] = -99

    lumDist = Planck15.luminosity_distance(z=zbest).cgs.value

    for band in filterSetJohnson.filters:
        output["model_Lnu_rest{:s}".format(band)][0] = (
            calcModelBandFlux(model["restwave"], model["Fnu"], band=band)
            * 1e-23
            / 1e6
            * (4 * np.pi * lumDist**2)
            / (1 + zbest)
        )

    for filt in filterSet.filters:
        output["model_Fnu_{:s}".format(filt)][0] = calcModelBandFlux(
            model["obswave"], model["Fnu"], band=filt
        )

    return output


def fitter(
    objID,
    flux,
    ferr,
    mask,
    zred,
    zerr,
    atlas,
    priors,
    params,
    saveDir,
    saveName,
    overwrite=False,
    cigaleCat=None,
    cigaleDir=None,
):
    import dense_basis as db

    if (
        (not overwrite)
        and os.path.isfile(os.path.join(saveDir, "{:d}_{:s}.fits.gz".format(objID, saveName)))
        and os.path.isfile(os.path.join(saveDir, "{:d}_{:s}.png".format(objID, saveName)))
    ):
        print(
            "Skipping objID#%d because output exists (not overwriting)" % objID,
            flush=True,
        )
        return

    if (zred < priors.z_min) or (priors.z_max < zred):
        print(
            "Skipping objID#%d due to z_phot (%.3f) being out of range [%.3f,%.3f]"
            % (objID, zred, priors.z_min, priors.z_max),
            flush=True,
        )
        return

    print("Processing objID#%d" % objID, flush=True)

    # Setup the best-fit figure
    fig, axes = setupBestFitPlot(ndim=params["Npars"] + 5)

    # Plot the observed photometry
    plotObsPhotometry(axis=axes[0], obsFlux=flux, obsFerr=ferr, obsMask=mask)

    # Setup the SedFitter class
    fitter = db.SedFit(sed=flux, sed_err=ferr, fit_mask=mask, zbest=zred, deltaz=zerr, atlas=atlas)

    # Evaluate the best fit
    fitter.evaluate_likelihood()
    fitter.evaluate_posterior_percentiles()
    # fitter.evaluate_posterior_SFH(zval=fitter.zbest)

    # Populate the relevant results
    results = {
        "mstar": fitter.mstar,
        "sfr": fitter.sfr,
        "Av": fitter.Av,
        "Z": fitter.Z,
        "z": fitter.z,
        "nuvu": fitter.nuvu,
        "nuvr": fitter.nuvr,
        "uv": fitter.uv,
        "vj": fitter.vj,
        "rj": fitter.rj,
        "sfh": fitter.sfh_tuple,
        "chi2": fitter.chi2_array,
    }
    modTimeScale = np.linspace(0, db.cosmo.age(fitter.zbest).value, 100)

    # Plot the corner subplots
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
    cornerParams[0, :] += np.log10(fitter.norm_fac)
    cornerParams[1, :] += np.log10(fitter.norm_fac)
    plotCorner(
        axis=axes[2],
        params=cornerParams,
        chi2Array=fitter.chi2_array,
        labels=plotLabels,
    )

    # Setup the table with best-fit parameters
    fitParams = populateFitParams(
        results=results,
        atlas=atlas,
        chi2Index=np.argmin(fitter.chi2_array),
        normFactor=fitter.norm_fac,
    )
    fitParams["ID"][0] = objID
    fitParams["zbest"][0] = zred
    fitParams["delz"][0] = zerr
    fitParams["nbands"][0] = sum(mask)

    # Save the model and parameters to file
    hdu = fits.HDUList()
    hdu.append(fits.PrimaryHDU())
    hdu.append(fits.BinTableHDU(fitParams, name="FIT_PARAMS"))

    # Compute the best-fit using the median best-fit parameters
    # Compute the best-fit model spectrum
    try:
        specdetails = [fitter.sfh_tuple[0], fitter.Av[0], fitter.Z[0], fitter.z[0]]
        wave, spec = db.makespec(specdetails, priors, db.mocksp, db.cosmo, return_spec=True)
        bestModelEval = np.array(
            list(zip(wave, wave * (1 + fitter.z[0]), spec)),
            dtype=[("restwave", float), ("obswave", float), ("Fnu", float)],
        )
        # Populate the model mags
        modelMagsEval = populateModelMags(model=bestModelEval, zbest=zred)
        modelMagsEval["ID"][0] = objID
        modelMagsEval["zbest"][0] = zred
        # Compute the best-fit SFH
        sfh, timeax = db.tuple_to_sfh(
            fitter.sfh_tuple[0],
            fitter.z[0],
            decouple_sfr=priors.decouple_sfr,
            decouple_sfr_time=priors.decouple_sfr_time,
        )
        sfh = np.interp(modTimeScale, timeax, sfh)
        bestSFHEval = np.array(
            list(zip(np.amax(modTimeScale) - modTimeScale, sfh)),
            dtype=[("time", float), ("sfr", float)],
        )
        # Plot the model
        plotModel(
            model=bestModelEval,
            modelMags=modelMagsEval,
            sfh=bestSFHEval,
            axisSpec=axes[0],
            axisSFH=axes[1],
            obsFlux=flux,
            obsMask=mask,
            color="tab:red",
        )
        # Plot the best-fit parameters on corner plot
        truthsEval = [fitParams[x] for x in ["logM", "logSFR", "logZsol", "Av", "zfit"]]
        truthsEval[2:2] = [fitParams[x] for x in txs]
        plotCornerTruths(
            axis=axes[2],
            labels=plotLabels,
            truths=truthsEval,
            color="tab:red",
        )
        hdu.append(fits.BinTableHDU(modelMagsEval, name="MODEL_MAGS_BEST"))
        hdu.append(fits.BinTableHDU(bestModelEval, name="MODEL_BEST"))
        hdu.append(fits.BinTableHDU(bestSFHEval, name="SFH_BEST"))

    except AssertionError:
        print("Fit error for objID#{:d}".format(objID))

    # Compute the best-fit using the minimum chi2 soln
    # Compute the best-fit model spectrum
    wave, spec = db.makespec_atlas(
        fitter.atlas,
        np.argmax(fitter.likelihood),
        priors,
        db.mocksp,
        db.cosmo,
        return_spec=True,
    )
    bestModelChi2 = np.array(
        list(
            zip(
                wave,
                wave * (1 + fitter.atlas["zval"][np.argmax(fitter.likelihood)]),
                spec * fitter.norm_fac,
            )
        ),
        dtype=[("restwave", float), ("obswave", float), ("Fnu", float)],
    )
    # Populate the model mags
    modelMagsChi2 = populateModelMags(model=bestModelChi2, zbest=zred)
    modelMagsChi2["ID"][0] = objID
    modelMagsChi2["zbest"][0] = zred
    # Compute the best-fit SFH
    sfh, timeax = db.tuple_to_sfh(
        fitter.atlas["sfh_tuple"][np.argmax(fitter.likelihood), 0:],
        fitter.atlas["zval"][np.argmax(fitter.likelihood)],
        decouple_sfr=priors.decouple_sfr,
        decouple_sfr_time=priors.decouple_sfr_time,
    )
    sfh = np.interp(modTimeScale, timeax, sfh * fitter.norm_fac)
    bestSFHChi2 = np.array(
        list(zip(np.amax(modTimeScale) - modTimeScale, sfh)),
        dtype=[("time", float), ("sfr", float)],
    )
    # Plot the model
    plotModel(
        model=bestModelChi2,
        modelMags=modelMagsChi2,
        sfh=bestSFHChi2,
        axisSpec=axes[0],
        axisSFH=axes[1],
        obsFlux=flux,
        obsMask=mask,
        color="tab:orange",
    )
    # Plot the best-fit parameters on corner plot
    truthsChi2 = [fitParams[x + "_chi2"] for x in ["logM", "logSFR", "logZsol", "Av", "zfit"]]
    truthsChi2[2:2] = [fitParams[x + "_chi2"] for x in txs]
    plotCornerTruths(
        axis=axes[2],
        labels=plotLabels,
        truths=truthsChi2,
        color="tab:orange",
    )
    # Add the model and parameters to the output save file
    hdu.append(fits.BinTableHDU(modelMagsChi2, name="MODEL_MAGS_CHI2"))
    hdu.append(fits.BinTableHDU(bestModelChi2, name="MODEL_CHI2"))
    hdu.append(fits.BinTableHDU(bestSFHChi2, name="SFH_CHI2"))

    hdu.writeto(
        os.path.join(saveDir, "{:d}_{:s}.fits.gz".format(objID, saveName)),
        overwrite=True,
    )

    # Plot the best-fit parameters
    fit, lerr, uerr = {}, {}, {}
    for x in ["logM", "logSFR", "Av", "logZsol"]:
        fit[x] = fitParams[x][0]
        uerr[x] = fitParams[x + "_84"][0] - fitParams[x][0]
        lerr[x] = fitParams[x + "_16"][0] - fitParams[x][0]
    ssfr = fitParams["logM"][0] - fitParams["logSFR"][0]

    summary = "ID: {:d}\n" "z$_{{phot}}$ = {:.3f}\n".format(objID, zred)
    bestfit = (
        "log(M$^{{*}}$) = {0[logM]:.3f}$_{{{1[logM]:.3f}}}^{{+{2[logM]:.3f}}}$ M$_\\odot$\n"
        "log(SFR$_{{100}}$) = {0[logSFR]:.2f}$_{{{1[logSFR]:.2f}}}^{{+{2[logSFR]:.2f}}}$ M$_\\odot$/yr\n"
        "A$_{{V}}$ = {0[Av]:.3f}$_{{{1[Av]:.3f}}}^{{+{2[Av]:.3f}}}$\n"
        "log(Z/Z$_\\odot$) = {0[logZsol]:.3f}$_{{{1[logZsol]:.3f}}}^{{+{2[logZsol]:.3f}}}$\n"
        "log(sSFR$_{{100}}$) = {3:.2f} yr$^{{-1}}$\n".format(fit, lerr, uerr, ssfr)
    )
    fig.text(0.8, 0.98, summary, fontsize=18, fontweight=600, va="top", ha="left")
    fig.text(0.8, 0.90, bestfit, fontsize=16, fontweight=400, va="top", ha="left")

    # Plot CIGALE best-fit
    if cigaleCat is not None:
        plotCigaleFit(
            objID=objID,
            cigaleCat=cigaleCat,
            cigaleDir=cigaleDir,
            axisSpec=axes[0],
            axisSFH=axes[1],
            axisCorner=axes[2],
        )

    if saveDir:
        fig.savefig(os.path.join(saveDir, "{:d}_{:s}.png".format(objID, saveName)))
        plt.close(fig)
    else:
        plt.show()


def worker(catalogIndex, catalog, obscat, priors, params, atlas, saveDir, saveName, overwrite):
    parsDict = getObsCatParsDict(catalog=catalog, obscat=obscat, catalogIndex=catalogIndex)

    fitter(
        **parsDict,
        priors=priors,
        params=params,
        atlas=atlas,
        saveDir=saveDir,
        saveName=saveName,
        overwrite=overwrite
    )


def main(field, catalog, params, saveName, nproc, overwrite=False):
    from schwimmbad import MultiPool
    from functools import partial
    import dense_basis as db

    obscat = setupDBasisInput(catalog=catalog, redshiftErrScale=0.025)

    catalogIndexs = np.arange(len(obscat["zred"]))
    redshiftIndexs = getRedshiftIndex(params=params, zred=obscat["zred"])

    for redshiftIndex in range(len(params["zbins"])):
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

        with MultiPool(processes=nproc) as pool:
            results = list(
                pool.map(
                    partial(
                        worker,
                        catalog=catalog,
                        obscat=obscat,
                        priors=priors,
                        params=params,
                        atlas=atlas,
                        saveDir=os.path.join(outputPath, field),
                        saveName=saveName,
                        overwrite=overwrite,
                    ),
                    catalogIndexs[redshiftIndexs == redshiftIndex],
                )
            )


def test(catalogIndex, field, catalog, params, saveName, overwrite=False):
    import dense_basis as db

    obscat = setupDBasisInput(catalog=catalog, redshiftErrScale=0.025)

    redshiftIndex = getRedshiftIndex(params=params, zred=obscat["zred"][catalogIndex])
    print(redshiftIndex, obscat["zred"][catalogIndex])

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
        priors=priors,
        params=params,
        atlas=atlas,
        saveDir=os.path.join(outputPath, field),
        saveName=saveName,
        overwrite=overwrite,
    )


def mkFinalOutput(field, catalog, params, saveName):
    saveDir = os.path.join(outputPath, field)

    tempName = glob.glob("{:s}/*_{:s}.fits.gz".format(saveDir, saveName))[0]
    dtypeFitParams = fits.getdata(tempName, extname="FIT_PARAMS").dtype.descr
    dtypeModelMags = fits.getdata(tempName, extname="MODEL_MAGS_BEST").dtype.descr

    cond = [_ not in dtypeFitParams for _ in dtypeModelMags]
    dtypeModelMags = [dtypeModelMags[i] for i in np.where(cond)[0]]
    dtype = dtypeFitParams + dtypeModelMags

    output = np.recarray(len(catalog), dtype=dtype)
    for x in output.dtype.names:
        output[x] = -99.0

    for k, entry in enumerate(catalog):
        if (k + 1) % 100 == 0:
            print(
                "\rProcessing %s obj#%5d/%5d ... " % (field, k + 1, len(catalog)),
                end="",
                flush=True,
            )

        fname = os.path.join(saveDir, "{:d}_{:s}.fits.gz".format(entry["ID"], saveName))

        if os.path.isfile(fname):
            try:
                modelMags = fits.getdata(fname, extname="MODEL_MAGS_BEST")
                for x in modelMags.dtype.names:
                    output[x][k] = modelMags[x][0]
            except KeyError:
                pass

            fitParams = fits.getdata(fname, extname="FIT_PARAMS")
            for x in fitParams.dtype.names:
                output[x][k] = fitParams[x][0]

        else:
            output["ID"][k] = entry["ID"]
            output["zbest"][k] = entry["zphot"]

    print("done")

    fits.writeto(
        os.path.join(outputPath, "{:s}_{:s}_dbasis.fits".format(params["prefix"], field)),
        output,
        overwrite=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--field", type=str, help="specify the field", default=None)
    parser.add_argument("-n", "--nproc", type=int, help="specify number of processes", default=35)
    parser.add_argument("-t", "--test", help="run test case", action="store_true")
    parser.add_argument("-o", "--overwrite", help="clobber existing output", action="store_true")

    args = parser.parse_args()

    if args.field not in fields:
        raise Exception("Invalid field selected.")

    params = getParams(runVersion="v1")
    catalog = getCatalog(field=args.field)
    saveName = "bestmodel_dbasis"

    if args.test:
        test(
            catalogIndex=28593,
            field=args.field,
            catalog=catalog,
            params=params,
            saveName=saveName,
            overwrite=args.overwrite,
        )

    else:
        # main(
        #     field=args.field,
        #     catalog=catalog,
        #     params=params,
        #     saveName=saveName,
        #     nproc=args.nproc,
        #     overwrite=args.overwrite,
        # )

        mkFinalOutput(field=args.field, catalog=catalog, params=params, saveName=saveName)
