from useful import *

import json
import sfd
import extinction
import astropy.units as u
from astropy.coordinates import SkyCoord

catDir = "/data/UVCANDELS/catalogs"
filterSet = FilterSet()
filterSetJohnson = JohnsonFilterSet()


def getIndivCatalogs(field):
    if field == "goodsn":
        catalog = "hlsp_candels_hst_wfc3_goodsn-barro19_multi_v1-1_photometry-cat.fits"
        cat_275 = "hlsp_uvcandels_goodsn_f275w_photometry-cat_v1.1-0.5.fits"
        cat_435 = None
        cat_zred = "hlsp_candels_hst_wfc3_goodsn-barro19_multi_v1_redshift-cat.fits"
        cat_mass = "hlsp_candels_hst_wfc3_goodsn-barro19_multi_v1_mass-fast-cat.fits"

        catalog = fits.getdata(os.path.join(catDir, catalog))
        cat_275 = fits.getdata(os.path.join(catDir, cat_275))
        cat_zred = fits.getdata(os.path.join(catDir, cat_zred))
        cat_mass = fits.getdata(os.path.join(catDir, cat_mass))
        cat_phys = cat_mass

    elif field == "goodss":
        catalog = "hlsp_candels_hst_wfc3_goodss-tot-multiband_f160w_v1_cat.fits"
        cat_275 = "hlsp_uvcandels_goodss_f275w_photometry-cat_v1.1-0.5.fits"
        cat_435 = None
        cat_zred = "hlsp_candels_hst_wfc3_goodss_santini_v1_mass_cat.fits"
        cat_phys = "hlsp_candels_hst_wfc3_goodss_santini_v1_physpar_cat.fits"

        catalog = fits.getdata(os.path.join(catDir, catalog), 1)
        cat_275 = fits.getdata(os.path.join(catDir, cat_275))
        cat_zred = fits.getdata(os.path.join(catDir, cat_zred), 1)
        cat_mass = cat_zred
        cat_phys = fits.getdata(os.path.join(catDir, cat_phys), 1)

    elif field == "cosmos":
        catalog = "hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits"
        cat_275 = "hlsp_uvcandels_cosmos_f275w_photometry-cat_v1.1-0.5.fits"
        cat_435 = "hlsp_uvcandels_cosmos_f435w_photometry-cat_v0.3-0.1.fits"
        cat_zred = "hlsp_candels_hst_wfc3_cos_v1_mass_cat.fits"
        cat_phys = "hlsp_candels_hst_wfc3_cos_v1_physpar_cat.fits"

        catalog = fits.getdata(os.path.join(catDir, catalog))
        cat_275 = fits.getdata(os.path.join(catDir, cat_275))
        cat_435 = fits.getdata(os.path.join(catDir, cat_435))
        cat_zred = fits.getdata(os.path.join(catDir, cat_zred))
        cat_mass = cat_zred
        cat_phys = fits.getdata(os.path.join(catDir, cat_phys))

    elif field == "egs":
        catalog = "hlsp_candels_hst_wfc3_egs-tot-multiband_f160w_v1_cat.fits"
        cat_275 = "hlsp_uvcandels_egs_f275w_photometry-cat_v1.1-0.5.fits"
        cat_435 = "hlsp_uvcandels_egs_f435w_photometry-cat_v0.2-0.1.fits"
        cat_zred = "hlsp_candels_hst_wfc3_egs_v1_mass_cat.fits"
        cat_phys = "hlsp_candels_hst_wfc3_egs_v1_physpar_cat.fits"

        catalog = fits.getdata(os.path.join(catDir, catalog))
        cat_275 = fits.getdata(os.path.join(catDir, cat_275))
        cat_435 = fits.getdata(os.path.join(catDir, cat_435))
        cat_zred = fits.getdata(os.path.join(catDir, cat_zred))
        cat_mass = cat_zred
        cat_phys = fits.getdata(os.path.join(catDir, cat_phys))

    else:
        raise Exception("Invalid field name.")

    return {
        "catalog": catalog,
        "cat_275": cat_275,
        "cat_435": cat_435,
        "cat_zred": cat_zred,
        "cat_mass": cat_mass,
        "cat_phys": cat_phys,
    }


def corrGalExtinction(obscat):
    coords = SkyCoord(ra=obscat["RA"] * u.degree, dec=obscat["DEC"] * u.degree)
    ebv = sfd.ebv(coords)
    Rv = 3.1

    for filt in filterSet.filters:
        pivot = filterSet.pivot[filt]
        ext = np.array(
            [extinction.ccm89(np.atleast_1d(pivot), a_v=ebv_ * Rv, r_v=Rv)[0] for ebv_ in ebv]
        )

        condMask = obscat["{:s}_MASK".format(filt)]
        condFlux = obscat["{:s}_FLUXERR".format(filt)] > 0
        cond = condMask & condFlux

        corrFactor = 10 ** (0.4 * ext)
        obscat["{:s}_FLUX".format(filt)][cond] *= corrFactor[cond]
        obscat["{:s}_FLUXERR".format(filt)][cond] *= corrFactor[cond]

    return obscat


def getErrScale(filters=filterSet.filters):
    errScale = {}

    for filt in filters:
        if filterSet.pivot[filt] < 1e4:
            errScale[filt] = 0.02
        elif filterSet.pivot[filt] < 3.5e4:
            errScale[filt] = 0.05
        else:
            errScale[filt] = 0.1

    return errScale


def parseCatalog(
    field,
    catalog,
    cat_275=None,
    cat_435=None,
    cat_zred=None,
    candels_zphot=False,
    corrGalExt=True,
):
    filters = filterSet.filters
    flxcols = json.load(open("filters/filter_list_uvc.json", "rb"))

    dtype = [
        ("FIELD", "U6"),
        ("ID", int),
        ("RA", float),
        ("DEC", float),
        ("zphot", float),
        ("zphot_l68", float),
        ("zphot_u68", float),
        ("CLASS_STAR", float),
    ]
    for filt in filters:
        dtype.extend(
            [
                ("{:s}_FLUX".format(filt), float),
                ("{:s}_FLUXERR".format(filt), float),
                ("{:s}_MASK".format(filt), bool),
            ]
        )

    obscat = np.recarray(len(catalog), dtype=dtype)
    for x in obscat.dtype.names:
        obscat[x] = -99

    obscat["FIELD"] = field
    obscat["ID"] = catalog["ID"]
    obscat["RA"] = catalog["RA"]
    obscat["DEC"] = catalog["DEC"]
    obscat["CLASS_STAR"] = catalog["CLASS_STAR"]

    for filt in filters:
        if filt in flxcols[field]:
            flxcol = flxcols[field][filt]
            obscat["{:s}_FLUX".format(filt)] = catalog[flxcol]
            obscat["{:s}_FLUXERR".format(filt)] = catalog[flxcol.replace("FLUX", "FLUXERR")]
            obscat["{:s}_MASK".format(filt)] = True
        else:
            obscat["{:s}_MASK".format(filt)] = False

    if cat_275 is not None:
        obscat["F275W_FLUX"] = cat_275["WFC3_F275W_FLUX_IMPROVED"]
        obscat["F275W_FLUXERR"] = cat_275["WFC3_F275W_FLUXERR_IMPROVED"]
        obscat["F275W_MASK"] = ~cat_275["FLAG_COVER"].astype(bool)
    if cat_435 is not None:
        obscat["F435W_FLUX"] = cat_435["WFC3_F435W_FLUX_IMPROVED"]
        obscat["F435W_FLUXERR"] = cat_435["WFC3_F435W_FLUXERR_IMPROVED"]
        obscat["F435W_MASK"] = ~cat_435["FLAG_COVER"].astype(bool)

    if candels_zphot:
        obscat["zphot"] = cat_zred["zbest"]
        obscat["zphot_l68"] = cat_zred["zphot_l68"]
        obscat["zphot_u68"] = cat_zred["zphot_u68"]
    else:
        cat_zred = np.genfromtxt(
            os.path.join(catDir, "uvcandels_photz.txt"),
            skip_header=2,
            dtype=[
                ("field", "U6"),
                ("ID", int),
                ("zphot", float),
                ("zphot_l68", float),
                ("zphot_u68", float),
                ("zpeak", float),
                ("zpeak_l68", float),
                ("zpeak_u68", float),
                ("zpeak2", float),
                ("zpeak2_l68", float),
                ("zpeak2_u68", float),
                ("zpeak3", float),
                ("zpeak3_l68", float),
                ("zpeak3_u68", float),
                ("zphot_med", float),
                ("zphot_med_l68", float),
                ("zphot_med_u68", float),
                ("zspec", float),
                ("zspec_ref", "U50"),
            ],
        )
        cond = cat_zred["field"] == field
        assert np.all(obscat["ID"] == cat_zred["ID"][cond])
        obscat["zphot"] = cat_zred["zphot"][cond]
        obscat["zphot_l68"] = cat_zred["zphot_l68"][cond]
        obscat["zphot_u68"] = cat_zred["zphot_u68"][cond]

    if corrGalExt:
        obscat = corrGalExtinction(obscat)

    return obscat


def getCatalog(field, candels_zphot=False, corrGalExt=True, returnIndivCats=False):
    catDict = getIndivCatalogs(field=field)

    catalog = parseCatalog(
        field=field,
        catalog=catDict["catalog"],
        cat_275=catDict["cat_275"],
        cat_435=catDict["cat_435"],
        cat_zred=catDict["cat_zred"],
        candels_zphot=candels_zphot,
        corrGalExt=corrGalExt,
    )

    if returnIndivCats:
        return catalog, catDict

    return catalog
