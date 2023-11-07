from useful import *
from utils import *

pregridPath = os.path.join(cwd, "dbasis/pregrids/")
plotdirPath = os.path.join(cwd, "dbasis/plots/")
outputPath = os.path.join(cwd, "dbasis/output/")
filterDir = os.path.join(cwd, "dbasis/filters/")
filterList = "filter_list_uvc.dat"


def getPriors(zmin=1e-4, zmax=12, Npars=4, verbose=False):
    import dense_basis as db

    priors = db.Priors()

    priors.mass_min = 7.0
    priors.mass_max = 12.0

    priors.sfr_prior_type = "sSFRflat"  # options are SFRflat, sSFRflat, sSFRlognormal
    priors.ssfr_min = -12.0
    priors.ssfr_max = -7.0

    priors.z_min = zmin
    priors.z_max = zmax

    priors.met_treatment = "flat"  # options are 'flat' and 'massmet'
    priors.Z_min = -1.5
    priors.Z_max = 0.25
    priors.massmet_width = 0.3

    priors.dust_model = "Calzetti"  # options are 'Calzetti' and 'CF00'
    priors.dust_prior = "flat"  # options are 'exp' and 'flat'
    priors.Av_min = 0.0
    priors.Av_max = 4.0
    priors.Av_exp_scale = 1.0

    priors.sfh_treatment = "custom"  # options are custom and TNGlike
    priors.tx_alpha = 5.0  # [22.0,8.0,7.0,7.0]
    priors.Nparam = Npars

    priors.decouple_sfr = True
    priors.decouple_sfr_time = 100  # in Myr
    priors.dynamic_decouple = False  # set decouple time according to redshift (100 Myr at z=0.1)

    if verbose:
        priors.print_priors()

    return priors


def getParams(runVersion, redshiftErrFloor=0.03):
    zbinsOrig = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ]
    )
    overlaps = np.clip(ceilForFloats(redshiftErrFloor * (1 + zbinsOrig), 1), 0.1, 0.5)

    zbins = [
        [
            np.clip(z0 - overlap, min(zbinsOrig), max(zbinsOrig)),
            np.clip(z1 + overlap, min(zbinsOrig), max(zbinsOrig)),
        ]
        for z0, z1, overlap in zip(zbinsOrig[:-1], zbinsOrig[1:], overlaps[1:])
    ]

    if runVersion == "v1":
        params = {
            "Nproc": 35,
            "Npts_iter": 50000,
            "Npars": 4,
            "zbins": zbins,
            "prefix": "uvc_%s" % runVersion,
        }

    elif runVersion == "test":
        params = {
            "Nproc": 35,
            "Npts_iter": 10000,
            "Npars": 4,
            "zbins": zbins,
            "prefix": "uvc_%s" % runVersion,
        }

    params["Npts"] = params["Npts_iter"] * params["Nproc"]
    params["zbins_orig"] = zbinsOrig
    params["overlaps"] = overlaps

    return params


def getRedshiftIndex(params, zred):
    zred = np.atleast_1d(zred)
    zID = np.zeros_like(zred, dtype=int) - 99

    for idx, (z0, z1) in enumerate(zip(params["zbins_orig"][:-1], params["zbins_orig"][1:])):
        cond = (z0 < zred) & (zred <= z1)
        zID[cond] = idx

    if len(zID) == 1:
        return zID[0]

    return zID


def getPregridName(params, z0, z1, chunkIndex=None, returnFull=True):
    suffixRedshift = "_z{:.0f}p{:.0f}_z{:.0f}p{:.0f}".format(
        z0 // 1, 10 * (z0 % 1), z1 // 1, 10 * (z1 % 1)
    )

    if chunkIndex is not None:
        suffixChunk = "_chunk%03d" % (chunkIndex + 1)
        prefix = params["prefix"] + suffixRedshift + suffixChunk
    else:
        prefix = params["prefix"] + suffixRedshift

    fname = "%s_%d_Nparam_%d.dbatlas" % (
        prefix,
        params["Npts_iter"] if chunkIndex is not None else params["Npts"],
        params["Npars"],
    )
    if not returnFull:
        return prefix
    return fname


def setupDBasisInput(catalog, magErrScale=None, redshiftErrScale=0.025):
    obsFlux, obsFerr = np.zeros((2, len(catalog), len(filterSet.filters)), dtype=float)
    obsMask = np.ones((len(catalog), len(filterSet.filters)), dtype=bool)

    if magErrScale is None:
        magErrScale = getErrScale()
    elif isinstance(magErrScale, float) or isinstance(magErrScale, int):
        magErrScales = np.array([magErrScale] * len(filterSet.filters))
        magErrScale = dict(zip(filterSet.filters, magErrScales))
    else:
        raise Exception("Invalid magErrScale arg.")

    for i, filt in enumerate(filterSet.filters):
        obsFlux[:, i] = catalog["{:s}_FLUX".format(filt)]
        obsFerr[:, i] = catalog["{:s}_FLUXERR".format(filt)]
        obsMask[:, i] = catalog["{:s}_MASK".format(filt)]

        ### Setup the scaling for flux error
        cond = (obsFlux[:, i] > 0) & (obsFerr[:, i] > 0)
        fluxErrScale = magErrScale[filt] * (np.log(10) / 2.5) * obsFlux[cond, i]
        obsFerr[cond, i] = np.sqrt(obsFerr[cond, i] ** 2 + fluxErrScale**2)

    obsZred = catalog["zphot"]
    obsZerr = redshiftErrScale * (1 + catalog["zphot"])

    return {
        "flux": obsFlux,
        "ferr": obsFerr,
        "mask": obsMask,
        "zred": obsZred,
        "zerr": obsZerr,
    }


def getObsCatParsDict(catalog, obscat, catalogIndex):
    return {
        "objID": catalog["ID"][catalogIndex],
        "flux": obscat["flux"][catalogIndex, :].copy(),
        "ferr": obscat["ferr"][catalogIndex, :].copy(),
        "mask": obscat["mask"][catalogIndex, :].copy(),
        "zred": obscat["zred"][catalogIndex],
        "zerr": obscat["zerr"][catalogIndex],
    }


def createMMap(atlas, mmapPrefix, pregridPath):
    for key in atlas.keys():
        mmapName = os.path.join(pregridPath, "%s_%s.mmap" % (mmapPrefix, key))
        if os.path.isfile(mmapName):
            os.remove(mmapName)
        mmap = np.memmap(mmapName, dtype="float32", mode="w+", shape=atlas[key].shape)
        mmap[:] = atlas[key][:]
        mmap.flush()
        mmap.flags.writeable = False
        atlas[key] = mmap

    return atlas
