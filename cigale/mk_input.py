from useful import *
from utils import *

f = FilterSet()


def convert_uJy_to_mJy(flux_uJy):
    """
    1 micro Jy = 1e-29 ergs/s/cm^2/Hz
    """
    cond = (flux_uJy != -99)
    flux_mJy = np.zeros(len(flux_uJy)) - 99.
    flux_mJy[cond] = flux_uJy[cond] * 1e-3
    return flux_mJy


def mk_cigale_input(field, catalog, nouv, err_floor=0.02):

    dtype = [('ID', int), ('ZSPEC', float)]
    for filt in f.filters:
        dtype += [('FLUX_%s' % filt, float), ('FLUXERR_%s' % filt, float)]

    input_cat = np.recarray((len(catalog), ), dtype=dtype)
    for x in input_cat.dtype.names:
        input_cat[x] = -99.

    input_cat['ID'] = catalog["ID"]
    input_cat['ZSPEC'] = catalog["zphot"]

    #######################################
    #### TEMPORARY FIX FOR ZPHOT = -99 ####
    #######################################
    cond = catalog["zphot"] < 0
    input_cat['ZSPEC'][cond] = 0
    #######################################

    for filt in f.filters:

        input_cat["FLUX_%s" % filt] = catalog["%s_FLUX" % filt]
        input_cat["FLUXERR_%s" % filt] = catalog["%s_FLUXERR" % filt]

        ### Setup the floor in flux error
        cond = (input_cat["FLUX_%s" % filt] >
                0) & (input_cat["FLUXERR_%s" % filt] > 0)
        ferr_floor = err_floor * (input_cat["FLUX_%s" % filt][cond] /
                                  input_cat["FLUXERR_%s" % filt][cond]) * (
                                      np.log(10) / 2.5)
        input_cat["FLUXERR_%s" % filt][cond] = np.maximum(
            input_cat["FLUXERR_%s" % filt][cond], ferr_floor)

        input_cat["FLUX_%s" % filt] = convert_uJy_to_mJy(input_cat["FLUX_%s" %
                                                                   filt])
        input_cat["FLUXERR_%s" % filt] = convert_uJy_to_mJy(
            input_cat["FLUXERR_%s" % filt])

        if nouv and filt == "F275W":
            input_cat["FLUX_%s" % filt] = -99.
            input_cat["FLUXERR_%s" % filt] = -99.

    flxcols = np.array([
        i.replace("FLUX_", "uvc.") for i in input_cat.dtype.names
        if "FLUX_" in i
    ])
    hdr = "%4s%10s" % ("id", "redshift")
    for x in flxcols:
        hdr += "%20s%20s" % (x, x + "_err")
    fmt = "%6i%10.4f" + "".join(["%20.6e%20.6e"] * len(flxcols))

    # cond = (catalog["F275W_FLUX"] / catalog["F275W_FLUXERR"] > 3) & \
    #        (catalog["zphot"] <= 1)
    # input_cat = input_cat[cond]
    # print("%s: %d gals" % (field, sum(cond)))
    # np.savetxt("goodsn_f275w/catalog.cat", input_cat, fmt=fmt, header=hdr)

    save_dir = field + ("_nouv" if nouv else "")
    # np.savetxt(os.path.join(save_dir, "catalog.cat"),
    #            input_cat,
    #            fmt=fmt,
    #            header=hdr)

    zrange = [0, 2, 4, 6, 8, 12]
    for i, (z0, z1) in enumerate(zip(zrange[:-1], zrange[1:])):

        _save_dir = os.path.join(save_dir, 'worker%d' % (i + 1))
        cat = input_cat[(z0 <= input_cat["ZSPEC"]) & (input_cat["ZSPEC"] < z1)]

        print("\rWriting out sub-file %s - %d/%d ... " %
              (save_dir, i + 1, len(zrange) - 1),
              end="",
              flush=True)

        runBashCommand("mkdir -p %s" % _save_dir,
                       cwd=os.path.join(cwd, "cigale"),
                       verbose=False)
        runBashCommand("ln -s ../../pcigale.ini .",
                       cwd=_save_dir,
                       verbose=False)
        runBashCommand("ln -s ../../pcigale.ini.spec .",
                       cwd=_save_dir,
                       verbose=False)
        np.savetxt(os.path.join(_save_dir, 'catalog.cat'),
                   cat,
                   fmt=fmt,
                   header=hdr)
        writeSlurmScript(save_dir=save_dir, workerID=i + 1)

    print("done.")


def writeSlurmScript(save_dir, workerID):

    script = "#!/bin/bash -l\n" \
             "#SBATCH -p amdsmall\n" \
             "#SBATCH --time=96:00:00\n" \
             "#SBATCH --ntasks=128\n" \
             "#SBATCH --mem=248g\n" \
             "#SBATCH --mail-type=ALL\n" \
             "#SBATCH --mail-user=mehta074@umn.edu\n"
    script += "cd %s\n" % os.path.join(cwd, "cigale", save_dir,
                                       'worker%d' % workerID)
    script += "pcigale run > log\n"

    scriptname = "cg_%s_%d.slurm" % (save_dir, workerID)
    with open(os.path.join(cwd, "cigale", "scripts", scriptname), "w") as f:
        f.write(script)


if __name__ == '__main__':

    for field in ["goodsn", "goodss", "cosmos", "egs"]:

        catalog = getCatalog(field=field)
        mk_cigale_input(field=field, catalog=catalog, nouv=False)

        # catalog = getCatalog(field=field, candels_zphot=True)
        # mk_cigale_input(field=field, catalog=catalog, nouv=True)
