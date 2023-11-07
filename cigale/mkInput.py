from useful import *
from utils import *

filterSet = FilterSet()


def convert_uJy_to_mJy(flux_uJy):
    """
    1 micro Jy = 1e-29 ergs/s/cm^2/Hz
    """
    cond = (flux_uJy != -9999) & (flux_uJy != -1)
    flux_mJy = flux_uJy.copy()
    flux_mJy[cond] = flux_uJy[cond] * 1e-3
    return flux_mJy


def mkCigaleInput(field, catalog, magErrScale=None):
    dtype = [("ID", "U15"), ("ZSPEC", float)]
    for filt in filterSet.filters:
        dtype += [("FLUX_%s" % filt, float), ("FLUXERR_%s" % filt, float)]

    inputCat = np.recarray((len(catalog),), dtype=dtype)
    for x in inputCat.dtype.names:
        inputCat[x] = -9999

    inputCat["ID"] = ["%s_%d" % (field.upper(), entry["ID"]) for entry in catalog]
    inputCat["ZSPEC"] = catalog["zphot"]

    #######################################
    #### TEMPORARY FIX FOR ZPHOT = -99 ####
    #######################################
    cond = catalog["zphot"] < 0
    inputCat["ZSPEC"][cond] = 0
    #######################################

    if magErrScale is None:
        magErrScale = getErrScale()
    elif isinstance(magErrScale, float) or isinstance(magErrScale, int):
        magErrScales = np.array([magErrScale] * len(filterSet.filters))
        magErrScale = dict(zip(filterSet.filters, magErrScales))
    else:
        raise Exception("Invalid magErrScale arg.")

    for filt in filterSet.filters:
        inputCat["FLUX_%s" % filt] = catalog["%s_FLUX" % filt]
        inputCat["FLUXERR_%s" % filt] = catalog["%s_FLUXERR" % filt]

        ### Replace 99s with 9999s
        condFlux99s = inputCat["FLUX_%s" % filt] == -99
        inputCat["FLUX_%s" % filt][condFlux99s] = -9999
        condErr99s = inputCat["FLUXERR_%s" % filt] == -99
        inputCat["FLUXERR_%s" % filt][condErr99s] = -9999

        ### Set the upper limits
        condLim = (inputCat["FLUX_%s" % filt] != -9999) & (
            inputCat["FLUX_%s" % filt] / inputCat["FLUXERR_%s" % filt] < 1
        )
        fluxErrScale = (
            magErrScale[filt] * (np.log(10) / 2.5) * inputCat["FLUXERR_%s" % filt][condLim]
        )
        inputCat["FLUX_%s" % filt][condLim] = np.sqrt(
            inputCat["FLUXERR_%s" % filt][condLim] ** 2 + fluxErrScale**2
        )
        inputCat["FLUX_%s" % filt][condLim] *= 3
        inputCat["FLUXERR_%s" % filt][condLim] = -1

        ### Setup the scaling for flux error
        condErrScale = (inputCat["FLUX_%s" % filt] > 0) & (inputCat["FLUXERR_%s" % filt] >= 0)
        fluxErrScale = (
            magErrScale[filt] * (np.log(10) / 2.5) * inputCat["FLUX_%s" % filt][condErrScale]
        )
        inputCat["FLUXERR_%s" % filt][condErrScale] = np.sqrt(
            inputCat["FLUXERR_%s" % filt][condErrScale] ** 2 + fluxErrScale**2
        )

        print(
            "[%6s] %10s: %5d -99s; %5d ulims; %5d err-scaled [%5d/%5d]"
            % (
                field,
                filt,
                sum(condFlux99s),
                sum(condLim),
                sum(condErrScale),
                (sum(condFlux99s) + sum(condLim) + sum(condErrScale)),
                len(inputCat),
            )
        )

        ### Convert to mJy
        inputCat["FLUX_%s" % filt] = convert_uJy_to_mJy(inputCat["FLUX_%s" % filt])
        inputCat["FLUXERR_%s" % filt] = convert_uJy_to_mJy(inputCat["FLUXERR_%s" % filt])

    return inputCat


def main(saveDir):
    for i, field in enumerate(fields):
        catalog = getCatalog(field=field)
        inputCat = mkCigaleInput(field=field, catalog=catalog)

        if i == 0:
            combinedCat = inputCat
        else:
            combinedCat = rfn.stack_arrays(
                [combinedCat, inputCat],
                usemask=False,
                autoconvert=False,
                asrecarray=True,
            )

    combinedCat = np.sort(combinedCat, order="ZSPEC")

    flxcols = np.array(
        [i.replace("FLUX_", "uvc.") for i in combinedCat.dtype.names if "FLUX_" in i]
    )
    hdr = "%18s%10s" % ("id", "redshift")
    for x in flxcols:
        hdr += "%20s%20s" % (x, x + "_err")
    fmt = "%20s%10.4f" + "".join(["%20.6e%20.6e"] * len(flxcols))

    Ntot = 0
    chunkSize = 5000
    chunkedCat = np.array_split(combinedCat, int(np.floor(len(combinedCat) / chunkSize)))
    for i, chunk in enumerate(chunkedCat):
        workDir = os.path.join(saveDir, "worker%02d" % (i + 1))
        Ntot += len(chunk)

        print(
            "\rWriting out sub-file %s (%4.2f<z<%4.2f) -- %5d gals -- %2d/%2d"
            % (
                workDir,
                min(chunk["ZSPEC"]),
                max(chunk["ZSPEC"]),
                len(chunk),
                i + 1,
                len(chunkedCat),
            ),
            flush=True,
        )

        runBashCommand("mkdir -p %s" % workDir, cwd=saveDir, verbose=False)
        runBashCommand("ln -s ../pcigale.ini .", cwd=workDir, verbose=False)
        runBashCommand("ln -s ../pcigale.ini.spec .", cwd=workDir, verbose=False)
        np.savetxt(os.path.join(workDir, "catalog.cat"), chunk, fmt=fmt, header=hdr)

    # Ntot = 0
    # zrange = [
    #     0,
    #     0.25,
    #     0.5,
    #     0.75,
    #     1,
    #     1.25,
    #     1.5,
    #     1.75,
    #     2,
    #     2.25,
    #     2.5,
    #     2.75,
    #     3,
    #     3.5,
    #     4,
    #     4.5,
    #     5,
    #     5.5,
    #     6,
    #     6.5,
    #     7,
    #     7.5,
    #     8,
    #     8.5,
    #     9,
    #     9.5,
    #     10,
    #     10.5,
    #     15,
    # ]

    # for i, (z0, z1) in enumerate(zip(zrange[:-1], zrange[1:])):
    #     workDir = os.path.join(saveDir, "worker%02d" % (i + 1))
    #     cat = combinedCat[(z0 <= combinedCat["ZSPEC"]) & (combinedCat["ZSPEC"] < z1)]
    #     Ntot += len(cat)

    #     print(
    #         "\rWriting out sub-file %s (%4.2f<z<%4.2f) -- %5d gals -- %2d/%2d"
    #         % (workDir, z0, z1, len(cat), i + 1, len(zrange) - 1),
    #         flush=True,
    #     )

    #     runBashCommand("mkdir -p %s" % workDir, cwd=saveDir, verbose=False)
    #     runBashCommand("ln -s ../pcigale.ini .", cwd=workDir, verbose=False)
    #     runBashCommand("ln -s ../pcigale.ini.spec .", cwd=workDir, verbose=False)
    #     np.savetxt(os.path.join(workDir, "catalog.cat"), cat, fmt=fmt, header=hdr)

    print("Total objects: {:d}/{:d}".format(Ntot, len(combinedCat)))


if __name__ == "__main__":
    main(saveDir=os.path.join(cwd, "cigale"))
