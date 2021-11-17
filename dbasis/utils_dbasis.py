from useful import *
from utils import *

filter_dir = os.path.join(cwd, "dbasis/filters/")
filter_list = "filter_list_uvc.dat"
filterset = FilterSet()
filterset_johnson = JohnsonFilterSet()

pregrid_path = os.path.join(cwd, "dbasis/pregrids.test/")
plotdir_path = os.path.join(cwd, "dbasis/plots/")
output_path = os.path.join(cwd, "dbasis/output.test/")
bestfit_path = os.path.join(cwd, "dbasis/best_fits.test/")


def get_run_params(run_version):

    if run_version == "v1":

        params = {
            "Nproc": 125,
            "Npts_iter": 400000,
            "Npars": 3,
            "prefix": "uvc_%s" % run_version
        }

    elif run_version == "test":

        params = {
            "zmax": 1,
            "Nproc": 50,
            "Npts_iter": 25000,
            "Npars": 3,
            "prefix": "uvc_%s" % run_version
        }

    params["Npts"] = params["Npts_iter"] * params["Nproc"]
    return params


def get_priors(zmin=1e-4, zmax=12, Npars=3, verbose=False):

    import dense_basis as db

    priors = db.Priors()

    priors.mass_min = 6.0
    priors.mass_max = 12.0

    priors.sfr_prior_type = 'sSFRflat'  # options are SFRflat, sSFRflat, sSFRlognormal
    priors.ssfr_min = -12.0
    priors.ssfr_max = -7.0

    priors.z_min = zmin
    priors.z_max = zmax

    priors.met_treatment = 'flat'  # options are 'flat' and 'massmet'
    priors.Z_min = -1.5
    priors.Z_max = 0.25
    priors.massmet_width = 0.3

    priors.dust_model = 'Calzetti'  # options are 'Calzetti' and 'CF00'
    priors.dust_prior = 'exp'  # options are 'exp' and 'flat'
    priors.Av_min = 0.0
    priors.Av_max = 4.0
    priors.Av_exp_scale = 1.0

    priors.sfh_treatment = 'custom'  # options are custom and TNGlike
    priors.tx_alpha = 5.0  # [22.0,8.0,7.0,7.0]
    priors.Nparam = Npars

    priors.decouple_sfr = True
    priors.decouple_sfr_time = 100  # in Myr
    priors.dynamic_decouple = False  # set decouple time according to redshift (100 Myr at z=0.1)

    if verbose: priors.print_priors()

    return priors


def setup_dbasis_input(catalog, nouv=False, err_floor=0.02, delz_floor=0.025):

    zbest = catalog["zphot"]
    deltaz = delz_floor * (1 + catalog["zphot"])

    # if np.all(catalog["zphot_u68"]==-99) and np.all(catalog["zphot_l68"]==-99):
    #     deltaz = delz_floor * (1 + catalog["zphot"])
    # else:
    #     deltaz = (catalog["zphot_u68"] - catalog["zphot_l68"]) / 2
    #     deltaz = np.clip(deltaz, delz_floor * (1 + catalog["zphot"]), 0.5)

    obs_sed, obs_err = np.zeros((2, len(catalog), len(filterset.filters)),
                                dtype=float)
    fit_mask = np.ones((len(catalog), len(filterset.filters)), dtype=bool)

    for i, filt in enumerate(filterset.filters):

        obs_sed[:, i] = catalog["{:s}_FLUX".format(filt)]
        obs_err[:, i] = catalog["{:s}_FLUXERR".format(filt)]
        fit_mask[:, i] = catalog["{:s}_MASK".format(filt)]

        if nouv and filt == "F275W":
            fit_mask[:, i] = False

        ### Setup the floor in flux error
        cond = (obs_sed[:, i] > 0) & (obs_err[:, i] > 0)
        ferr_floor = err_floor * (obs_sed[cond, i] /
                                  obs_err[cond, i]) * (np.log(10) / 2.5)
        obs_err[cond, i] = np.maximum(obs_err[cond, i], ferr_floor)

    return {
        "sed": obs_sed,
        "err": obs_err,
        "mask": fit_mask,
        "zbest": zbest,
        "delz": deltaz
    }


def createMMap(atlas, pregrid_path):

    for key in atlas.keys():

        # mmap_name = os.path.join(pregrid_path, "atlas_%s.mmap" % key)
        mmap_name = "/scratch.global/atlas_%s.mmap" % key
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
