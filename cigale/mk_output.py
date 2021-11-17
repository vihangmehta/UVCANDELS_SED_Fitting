from useful import *
from utils import *

def mk_cigale_output(field,catalog):

    print("{:s}: Input catalog -- {:d} objects".format(field,len(catalog)))

    finCat = None
    resList = sorted(glob.glob(os.path.join(field,"worker*","out","results.fits")))

    for resFile in resList:

        results = fitsio.getdata(resFile)
        print("{:s}: read output from {:8s} -- {:5d} objects".format(
                field,resFile.split("/")[1],len(results)))

        if finCat is None:
            finCat = results
        else:
            finCat = rfn.stack_arrays([finCat,results],
                        usemask=False,asrecarray=True,autoconvert=False)

    cond = (~np.in1d(catalog["ID"],finCat["id"]))
    print("{:s}: Final catalog -- {:d} objects ({:d} missing)".format(
            field,len(finCat),sum(cond)))
    if sum(cond)>0:
        print("{:s}: Missing IDs -- {:}".format(field,catalog["ID"][cond]))

    missing = np.recarray(sum(cond),dtype=finCat.dtype.descr)
    for x in finCat.dtype.names: missing[x] = -99
    missing["id"] = catalog["ID"][cond]
    finCat = rfn.stack_arrays([finCat,missing],
                usemask=False,asrecarray=True,autoconvert=False)

    finCat.sort(order="id")
    assert np.all(catalog["ID"]==finCat["id"])
    fitsio.writeto("final_cats/uvc_v1_{:s}_out.fits".format(field),finCat,overwrite=True)

if __name__ == '__main__':

    for field in ["goodsn", "goodss", "cosmos", "egs"][2:3]:

        catalog = getCatalog(field=field)
        mk_cigale_output(field=field, catalog=catalog)
