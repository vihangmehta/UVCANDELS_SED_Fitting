from useful import *
import json

cat_dir = "/home/scarlata/mehta074/Documents/UVCANDELS/catalogs"


def parseCatalog(field,
                 catalog,
                 cat_275=None,
                 cat_435=None,
                 cat_zred=None,
                 candels_zphot=False):

    filters = FilterSet().filters
    flxcols = json.load(open("filters/filter_list_uvc.json", "rb"))

    dtype = [
        ("ID", int),
        ("zphot", float),
        ("zphot_l68", float),
        ("zphot_u68", float),
        ("CLASS_STAR", float),
    ]
    for filt in filters:
        dtype.extend([
            ("{:s}_FLUX".format(filt), float),
            ("{:s}_FLUXERR".format(filt), float),
            ("{:s}_MASK".format(filt), bool),
        ])

    obscat = np.recarray(len(catalog), dtype=dtype)
    for x in obscat.dtype.names:
        obscat[x] = -99

    obscat["ID"] = catalog["ID"]
    obscat["CLASS_STAR"] = catalog["CLASS_STAR"]

    for filt in filters:
        if filt in flxcols[field]:
            flxcol = flxcols[field][filt]
            obscat["{:s}_FLUX".format(filt)] = catalog[flxcol]
            obscat["{:s}_FLUXERR".format(filt)] = catalog[flxcol.replace(
                "FLUX", "FLUXERR")]
            obscat["{:s}_MASK".format(filt)] = True
        else:
            obscat["{:s}_MASK".format(filt)] = False

    if cat_275 is not None:
        obscat["F275W_FLUX"] = cat_275["WFC3_F275W_FLUX_IMPROVED"]
        obscat["F275W_FLUXERR"] = cat_275["WFC3_F275W_FLUXERR_IMPROVED"]
        obscat["F275W_MASK"] = True
    if cat_435 is not None:
        obscat["F435W_FLUX"] = cat_435["WFC3_F435W_FLUX_IMPROVED"]
        obscat["F435W_FLUXERR"] = cat_435["WFC3_F435W_FLUXERR_IMPROVED"]
        obscat["F435W_MASK"] = True

    if field == "goodsn":
        obscat["zphot"] = cat_zred["zbest"]
    elif candels_zphot:
        obscat["zphot"] = cat_zred["zbest"]
        obscat["zphot_l68"] = cat_zred["zphot_l68"]
        obscat["zphot_u68"] = cat_zred["zphot_u68"]
    else:
        cat_zred = np.genfromtxt(os.path.join(cat_dir,
                                             "uvcandels_photz_eazy.txt"),
                                skip_header=2,
                                dtype=[("field", "U6"), ("ID", int),
                                       ("zphot", float), ("zphot_l68", float),
                                       ("zphot_u68", float)])
        cond = (cat_zred["field"] == field)
        assert np.all(obscat["ID"] == cat_zred["ID"][cond])
        obscat["zphot"] = cat_zred["zphot"][cond]
        obscat["zphot_l68"] = cat_zred["zphot_l68"][cond]
        obscat["zphot_u68"] = cat_zred["zphot_u68"][cond]

    return obscat


def getCatalog(field, candels_zphot=False):

    if field == "goodsn":
        catalog = "hlsp_uvcandels_goodsn_photometry-cat_v0.9-0.1-1.fits"
        catalog = fitsio.getdata(os.path.join(cat_dir, catalog))
        cat_275, cat_435 = None, None
        cat_zred = "hlsp_candels_hst_wfc3_goodsn-barro19_multi_v1_redshift-cat.fits"
        cat_zred = fitsio.getdata(os.path.join(cat_dir, cat_zred))

    elif field == "goodss":
        catalog = "hlsp_candels_hst_wfc3_goodss-tot-multiband_f160w_v1_cat.fits"
        catalog = fitsio.getdata(os.path.join(cat_dir, catalog), 1)
        cat_275 = "hlsp_uvcandels_goodss_photometry-cat_v0.9-0.25.fits"
        cat_275 = fitsio.getdata(os.path.join(cat_dir, cat_275))
        cat_435 = None
        cat_zred = "hlsp_candels_hst_wfc3_goodss_santini_v1_mass_cat.fits"
        cat_zred = fitsio.getdata(os.path.join(cat_dir, cat_zred), 1)

    elif field == "cosmos":
        catalog = "hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits"
        catalog = fitsio.getdata(os.path.join(cat_dir, catalog))
        cat_275 = "hlsp_uvcandels_cosmos_photometry-cat_v1.0-0.3.fits"
        cat_275 = fitsio.getdata(os.path.join(cat_dir, cat_275))
        cat_435 = "hlsp_uvcandels_cosmos_f435w_photometry-cat_v0.3-0.1.fits"
        cat_435 = fitsio.getdata(os.path.join(cat_dir, cat_435))
        cat_zred = "hlsp_candels_hst_wfc3_cos_v1_mass_cat.fits"
        cat_zred = fitsio.getdata(os.path.join(cat_dir, cat_zred))

    elif field == "egs":
        catalog = "hlsp_candels_hst_wfc3_egs-tot-multiband_f160w_v1_cat.fits"
        catalog = fitsio.getdata(os.path.join(cat_dir, catalog))
        cat_275 = "hlsp_uvcandels_egs_photometry-cat_v1.0-0.4.fits"
        cat_275 = fitsio.getdata(os.path.join(cat_dir, cat_275))
        cat_435 = "hlsp_uvcandels_egs_f435w_photometry-cat_v0.2-0.1.fits"
        cat_435 = fitsio.getdata(os.path.join(cat_dir, cat_435))
        cat_zred = "hlsp_candels_hst_wfc3_egs_v1_mass_cat.fits"
        cat_zred = fitsio.getdata(os.path.join(cat_dir, cat_zred))

    else:
        raise Exception("Invalid field name.")

    catalog = parseCatalog(field=field,
                           catalog=catalog,
                           cat_275=cat_275,
                           cat_435=cat_435,
                           cat_zred=cat_zred,
                           candels_zphot=candels_zphot)
    return catalog


def getDBasisCatalog(catname):

    return fitsio.getdata(catname)


def getCigaleCatalog(catname):

    return fitsio.getdata(catname)
