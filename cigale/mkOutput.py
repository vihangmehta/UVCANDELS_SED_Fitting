from useful import *
from utils import *

from joblib import Parallel, delayed


def copyBestFitFiles(field, entry, verbose=False):

    modelName = "%s_%d_best_model.fits" % (field.upper(), entry["id"])
    modelPath = glob.glob(os.path.join("worker*", "out", modelName))

    if len(modelPath) > 0:
        modelName = "%d_bestmodel_cigale.fits" % (entry["id"])
        modelDest = os.path.join("output", field, modelName)

        os.system("mv %s %s" % (modelPath[0], modelDest))
        os.system("gzip %s" % modelDest)
    else:
        if verbose:
            print("Warning! No model files found for %s-%d." % (field, entry["id"]))

    sfhName = "%s_%d_SFH.fits" % (field.upper(), entry["id"])
    sfhPath = glob.glob(os.path.join("worker*", "out", sfhName))

    if len(sfhPath) > 0:
        sfhName = "%d_bestSFH_cigale.fits" % (entry["id"])
        sfhDest = os.path.join("output", field, sfhName)

        os.system("mv %s %s" % (sfhPath[0], sfhDest))
        os.system("gzip %s" % sfhDest)
    else:
        print("Warning! No SFH files found for %s-%d." % (field, entry["id"]))


def mkCigaleOutput():

    resList = sorted(glob.glob(os.path.join("worker*", "out", "results.fits")))
    for i, resFile in enumerate(resList):

        results = fits.getdata(resFile)
        print("{:s}: {:5d} objs".format(resFile.split("/")[0], len(results)))

        if i == 0:
            finCat = results
        else:
            finCat = rfn.stack_arrays(
                [finCat, results], usemask=False, asrecarray=True, autoconvert=False
            )

    for field in fields:

        catalog = getCatalog(field=field)

        cond = [field.upper() in _ for _ in finCat["id"].astype(str)]
        outCat = finCat[cond]
        outCat["id"] = [_.split("_")[1] for _ in outCat["id"].astype(str)]

        dtype = outCat.dtype.descr
        dtype[0] = ("id", ">i8")
        outCat = np.array(outCat, dtype=dtype)

        cond = ~np.in1d(catalog["ID"], outCat["id"])
        print("{:s}: {:d} objs ({:d} missing)".format(field, len(catalog), sum(cond)))

        if sum(cond) > 0:
            print("{:s}: Missing IDs -- {:}".format(field, catalog["ID"][cond]))
            missing = np.recarray(sum(cond), dtype=outCat.dtype.descr)
            for x in outCat.dtype.names:
                missing[x] = -99
            missing["id"] = catalog["ID"][cond]
            outCat = rfn.stack_arrays(
                [outCat, missing], usemask=False, asrecarray=True, autoconvert=False,
            )

        outCat.sort(order="id")
        assert np.all(catalog["ID"] == outCat["id"])

        fits.writeto(
            "output/uvc_v1_{:s}_cigale.fits".format(field), outCat, overwrite=True,
        )

        Parallel(n_jobs=25, verbose=5)(
            delayed(copyBestFitFiles)(field=field, entry=entry, verbose=False) for entry in outCat
        )


if __name__ == "__main__":

    mkCigaleOutput()
