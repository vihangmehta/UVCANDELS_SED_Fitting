from useful import *
from utils import *
from utils_dbasis import *

from astropy.cosmology import Planck15


def calc_model_bandflux(wave, flux, band):

    if band in filterset.filters:
        resp = filterset.get_response(band)
        filt_pivot = filterset.pivotWL[band]
    elif band in filterset_johnson.filters:
        resp = filterset_johnson.get_response(band)
        filt_pivot = filterset_johnson.pivotWL[band]
    else:
        raise Exception("Invalid filter in calc_model_bandflux().")

    filt_interp = interp1d(resp["wave"],
                           resp["throughput"],
                           bounds_error=False,
                           fill_value=0,
                           kind='linear')
    filt_sens = filt_interp(wave)

    flux = flux * (light / wave**2)
    flux = simps(flux * filt_sens * wave, wave) / simps(filt_sens * wave, wave)
    flux = flux * (filt_pivot**2 / light)
    return flux


def setup_table(results, model, zbest, sfr_uncert_cutoff=2.0):

    dtype = [("ID", int), ("zbest", float), ("delz", float), ("logM", float),
             ("logM_16", float), ("logM_84", float), ("logSFRinst", float),
             ("logSFRinst_16", float), ("logSFRinst_84", float), ("Av", float),
             ("Av_16", float), ("Av_84", float), ("logZsol", float),
             ("logZsol_16", float), ("logZsol_84", float), ("zfit", float),
             ("zfit_16", float), ("zfit_84", float), ("logM2", float),
             ("logM2_16", float), ("logM2_84", float), ("logSFR100", float),
             ("logSFR100_16", float), ("logSFR100_84", float), ("t25", float),
             ("t25_16", float), ("t25_84", float), ("t50", float),
             ("t50_16", float), ("t50_84", float), ("t75", float),
             ("t75_16", float), ("t75_84", float), ("nparam", int),
             ("nbands", int), ("chi2", float), ("flags", int)]

    for band in filterset_johnson.filters:
        dtype.append(("model_Lnu_rest%s" % band, float))

    for filt in filterset.filters:
        dtype.append(("model_Fnu_%s" % filt, float))

    output = np.recarray(1, dtype=dtype)

    for x in output.dtype.names:
        output[x] = -99

    output["logM2"][0] = results["mstar"][0]
    output["logM2_16"][0] = results["mstar"][1]
    output["logM2_84"][0] = results["mstar"][2]

    output["logSFRinst"][0] = results["sfr"][0]
    output["logSFRinst_16"][0] = results["sfr"][1]
    output["logSFRinst_84"][0] = results["sfr"][2]

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
    output["logSFR100"][0] = results["sfh"][0][1]
    output["logSFR100_16"][0] = results["sfh"][1][1]
    output["logSFR100_84"][0] = results["sfh"][2][1]
    output["nparam"][0] = int(results["sfh"][0][2])
    output["t25"][0] = results["sfh"][0][3]
    output["t25_16"][0] = results["sfh"][1][3]
    output["t25_84"][0] = results["sfh"][2][3]
    output["t50"][0] = results["sfh"][0][4]
    output["t50_16"][0] = results["sfh"][1][4]
    output["t50_84"][0] = results["sfh"][2][4]
    output["t75"][0] = results["sfh"][0][5]
    output["t75_16"][0] = results["sfh"][1][5]
    output["t75_84"][0] = results["sfh"][2][5]

    output["chi2"][0] = np.amin(results["chi2"])

    # flagging galaxies that either
    # 1. have nan values for mass
    if np.isnan(output["logM"][0]):
        output["flags"][0] = 1
    # 2. have SFR uncertainties > sfr_uncert_cutoff
    elif (np.abs(output["logSFR100_84"][0] - output["logSFR100_16"][0]) >
          sfr_uncert_cutoff):
        output["flags"][0] = 2
    # 3. are flagged as a star
    elif (catalog["CLASS_STAR"][0] > 0.5):
        output["flags"][0] = 3
    # 4. have extremely large chi2
    elif (output["chi2"][0] > 1000):
        output["flags"][0] = 4
    else:
        output["flags"][0] = 0

    lumDist = Planck15.luminosity_distance(z=zbest).cgs.value
    for band in filterset_johnson.filters:
        output["model_Lnu_rest%s" % band][0] = calc_model_bandflux(
            model["restwave"], model["Fnu"],
            band=band) * 1e-23 / 1e6 * (4 * np.pi * lumDist**2) / (1 + zbest)

    for filt in filterset.filters:
        output["model_Fnu_%s" % filt][0] = calc_model_bandflux(
            model["obswave"], model["Fnu"], band=filt)

    return output


def worker(galID, obs, catalog, atlas, priors, savedir, overwrite=False):

    import dense_basis as db

    objID = catalog["ID"][galID]
    sed = obs["sed"][galID, :].copy()
    sed_err = obs["err"][galID, :].copy()
    fit_mask = obs["mask"][galID, :].copy()
    zbest = obs["zbest"][galID]
    deltaz = obs["delz"][galID]

    if (zbest < priors.z_min) or (priors.z_max < zbest):
        print(
            "Skipping objID#%d due to z_phot (%.3f) being out of range [%.3f,%.3f]"
            % (objID, zbest, priors.z_min, priors.z_max),
            flush=True)
        return

    if (not overwrite) and os.path.isfile(
            os.path.join(savedir, "%d_best_model.fits.gz" % objID)):
        return

    # Setup the SedFitter class
    fitter = db.SedFit(sed=sed,
                       sed_err=sed_err,
                       fit_mask=fit_mask,
                       zbest=zbest,
                       deltaz=deltaz,
                       atlas=atlas)

    # Evaluate the best fit
    fitter.evaluate_likelihood()
    fitter.evaluate_posterior_percentiles()
    fitter.evaluate_posterior_SFH(zval=fitter.zbest)

    # Populate the relevant results
    quants = {
        "mstar": fitter.mstar,
        "sfr": fitter.sfr,
        "Av": fitter.Av,
        "Z": fitter.Z,
        "z": fitter.z,
        "sfh": fitter.sfh_tuple,
        "chi2": fitter.chi2_array,
    }

    common_time = np.linspace(0, db.cosmo.age(fitter.zbest).value, 100)

    # Compute the best-fit using the minimum chi2 soln
    # Compute the best-fit model spectrum
    wave, spec = db.makespec_atlas(fitter.atlas,
                                   np.argmax(fitter.likelihood),
                                   priors,
                                   db.mocksp,
                                   db.cosmo,
                                   return_spec=True)
    best_model_chi2 = np.array(list(
        zip(wave,
            wave * (1 + fitter.atlas['zval'][np.argmax(fitter.likelihood)]),
            spec * fitter.norm_fac)),
                               dtype=[("restwave", float), ("obswave", float),
                                      ("Fnu", float)])
    # Compute the best-fit SFH
    sfh, timeax = db.tuple_to_sfh(
        fitter.atlas['sfh_tuple'][np.argmax(fitter.likelihood), 0:],
        fitter.atlas['zval'][np.argmax(fitter.likelihood)])
    sfh = np.interp(common_time, timeax, sfh * fitter.norm_fac)
    best_sfh_chi2 = np.array(list(zip(np.amax(common_time) - common_time,
                                      sfh)),
                             dtype=[("time", float), ("sfr", float)])

    # Compute the best-fit using the median best-fit parameters
    # Compute the best-fit model spectrum
    try:
        specdetails = [
            fitter.sfh_tuple[0], fitter.Av[0], fitter.Z[0], fitter.z[0]
        ]
        wave, spec = db.makespec(specdetails,
                                 priors,
                                 db.mocksp,
                                 db.cosmo,
                                 return_spec=True)
        best_model_eval = np.array(list(
            zip(wave, wave * (1 + fitter.z[0]), spec * fitter.norm_fac)),
                                   dtype=[("restwave", float),
                                          ("obswave", float), ("Fnu", float)])
    except AssertionError:
        print("Increasing age error for objID#%d" % objID,
              fitter.sfh_tuple[0],
              flush=True)
        best_model_eval = best_model_chi2
    # Compute the best-fit SFH
    sfh, timeax = db.tuple_to_sfh(fitter.sfh_tuple[0], fitter.z[0])
    sfh = np.interp(common_time, timeax, sfh * fitter.norm_fac)
    best_sfh_eval = np.array(list(zip(np.amax(common_time) - common_time,
                                      sfh)),
                             dtype=[("time", float), ("sfr", float)])

    # Setup the table with best-fit parameters
    fitTable = setup_table(results=quants, model=best_model_eval, zbest=zbest)
    fitTable["ID"][0] = objID
    fitTable["zbest"][0] = zbest
    fitTable["delz"][0] = deltaz
    fitTable["nbands"][0] = sum(fit_mask)

    # Save the model and parameters to file
    hdu = fitsio.HDUList()
    hdu.append(fitsio.PrimaryHDU())
    hdu.append(fitsio.BinTableHDU(fitTable, name="FIT_PARAMS"))
    hdu.append(fitsio.BinTableHDU(best_model_eval, name="MODEL_BEST"))
    hdu.append(fitsio.BinTableHDU(best_sfh_eval, name="SFH_BEST"))
    hdu.append(fitsio.BinTableHDU(best_model_chi2, name="MODEL_CHI2"))
    hdu.append(fitsio.BinTableHDU(best_sfh_chi2, name="SFH_CHI2"))
    hdu.writeto(os.path.join(savedir, "%d_best_model.fits.gz" % objID),
                overwrite=True)


def createMMap(atlas, pregrid_path):

    for key in atlas.keys():

        mmap_name = os.path.join(pregrid_path, "atlas_%s.mmap" % key)
        if os.path.isfile(mmap_name):
            os.remove(mmap_name)
        mmap = np.memmap(mmap_name,
                         dtype="float32",
                         mode="w+",
                         shape=atlas[key].shape)
        mmap[:] = atlas[key][:]
        mmap.flush()
        mmap.flags.writeable = False
        atlas[key] = mmap

    return atlas


def main(field, catalog, params, priors, nproc, nouv=False, overwrite=False):

    from schwimmbad import MultiPool
    from functools import partial
    import dense_basis as db

    obs = setup_dbasis_input(catalog,
                             nouv=nouv,
                             err_floor=0.02,
                             delz_floor=0.025)

    atlas = db.load_atlas(params["prefix"],
                          path=pregrid_path,
                          N_pregrid=params["Npts"],
                          N_param=params["Npars"])
    atlas = createMMap(atlas=atlas, pregrid_path=pregrid_path)

    savedir = os.path.join(bestfit_path, field)
    galIDs = np.arange(len(obs["zbest"]))

    with MultiPool(processes=nproc) as pool:
        results = list(
            pool.map(
                partial(worker,
                        obs=obs,
                        catalog=catalog,
                        atlas=atlas,
                        priors=priors,
                        savedir=savedir,
                        overwrite=overwrite), galIDs))


def mk_final_outcat(field, catalog, params, nouv=False):

    savedir = os.path.join(bestfit_path, field)

    tmpname = glob.glob("%s/*_best_model.fits.gz" % savedir)[0]
    dtype = fitsio.getdata(tmpname, extname="FIT_PARAMS").dtype.descr

    for k, entry in enumerate(catalog):

        if (k + 1) % 100 == 0:
            print("\rProcessing %s obj#%5d/%5d ... " %
                  (field, k + 1, len(catalog)),
                  end="",
                  flush=True)

        fname = os.path.join(savedir, "%s_best_model.fits.gz" % entry["ID"])

        if os.path.isfile(fname):
            res = fitsio.getdata(fname, extname="FIT_PARAMS")
        else:
            res = np.recarray(1, dtype=dtype)
            for x in res.dtype.names:
                res[x] = -99.
            res["ID"] = entry["ID"]
            res["zbest"] = entry["zphot"]

        if k == 0:
            output = np.recarray(len(catalog), dtype=dtype)

        for x in output.dtype.names:
            output[x][k] = res[x][0]

    print("done")

    fitsio.writeto(os.path.join(
        output_path, "%s_%s%s_out.fits" %
        (field, params["prefix"], "_nouv" if nouv else "")),
                   output,
                   overwrite=True)


def test(galID, field, catalog, params, priors, nouv=False):

    import dense_basis as db

    obs = setup_dbasis_input(catalog,
                             nouv=nouv,
                             err_floor=0.02,
                             delz_floor=0.025)

    atlas = db.load_atlas(params["prefix"],
                          path=pregrid_path,
                          N_pregrid=params["Npts"],
                          N_param=params["Npars"])

    savedir = os.path.join(bestfit_path, field)
    worker(galID=galID,
           obs=obs,
           catalog=catalog,
           atlas=atlas,
           priors=priors,
           savedir=savedir)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--field",
                        type=str,
                        help="specify the field",
                        default=None)
    parser.add_argument("-n",
                        "--nproc",
                        type=int,
                        help="specify number of processes",
                        default=50)
    args = parser.parse_args()
    if args.field not in fields:
        raise Exception("Invalid field selected.")

    params = get_run_params(run_version="test")
    priors = get_priors(Npars=params["Npars"], zmax=params["zmax"])

    catalog = getCatalog(field=args.field)
    # catalog = catalog[(0<catalog["zphot"]) & (catalog["zphot"]<1)]

    # main(field=args.field,
    #      catalog=catalog,
    #      params=params,
    #      priors=priors,
    #      nproc=args.nproc,
    #      nouv=False)

    mk_final_outcat(field=args.field,
                    catalog=catalog,
                    params=params,
                    nouv=False)

    # test(galID=1234,
    #      field=field,
    #      catalog=catalog,
    #      params=params,
    #      priors=priors,
    #      nouv=False)
