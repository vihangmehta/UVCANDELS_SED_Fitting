from useful import *
from utils import *
from utilsDBasis import *
from plotUtils import *


def generatePregrid(
    Npts, priors, gridPath, gridName, filterList=filterList, filterDir=filterDir
):

    import dense_basis as db

    db.generate_atlas(
        N_pregrid=Npts,
        priors=priors,
        fname=gridName,
        path=gridPath,
        filter_list=filterList,
        filt_dir=filterDir,
        store=True,
    )


def worker(chunkIndex, redshiftIndex, params, priors):

    pregridName = getPregridName(
        params=params,
        z0=params["zbins"][redshiftIndex][0],
        z1=params["zbins"][redshiftIndex][1],
        chunkIndex=chunkIndex,
        returnFull=False,
    )

    generatePregrid(
        Npts=params["Npts_iter"],
        priors=priors,
        gridPath=pregridPath,
        gridName=pregridName,
        filterList=filterList,
        filterDir=filterDir,
    )


def main(params, redshiftIndex, plotDist=False):

    from schwimmbad import MultiPool
    from functools import partial

    priors = getPriors(
        Npars=params["Npars"],
        zmin=params["zbins"][redshiftIndex][0],
        zmax=params["zbins"][redshiftIndex][1],
    )

    if plotDist:
        pregridName = getPregridName(
            params=params,
            z0=params["zbins"][redshiftIndex][0],
            z1=params["zbins"][redshiftIndex][1],
            returnFull=False,
        )
        plotPriorDist(
            priors,
            savename=os.path.join(plotdirPath, "priors", "%s_priors.png" % pregridName),
        )

    with MultiPool(processes=params["Nproc"]) as pool:
        values = list(
            pool.map(
                partial(
                    worker, redshiftIndex=redshiftIndex, params=params, priors=priors
                ),
                range(params["Nproc"]),
            )
        )


def combinePregrids(params, redshiftIndex):

    import hickle

    atlas = {}

    pregridList = [
        getPregridName(
            params=params,
            z0=params["zbins"][redshiftIndex][0],
            z1=params["zbins"][redshiftIndex][1],
            chunkIndex=chunkIndex,
        )
        for chunkIndex in range(params["Nproc"])
    ]

    for i, pregridName in enumerate(pregridList):

        print(
            "\rProcessing pregrid chunk#%3d/%3d ... " % (i + 1, len(pregridList)),
            end="",
        )

        atlasTmp = hickle.load(os.path.join(pregridPath, pregridName))
        for key in atlasTmp:
            if key != "norm_method":
                if key in atlas:
                    atlas[key] = np.append(atlas[key], atlasTmp[key], axis=0)
                else:
                    atlas[key] = atlasTmp[key]

        os.remove(os.path.join(pregridPath, pregridName))

    print("done.")

    pregridName = getPregridName(
        params=params,
        z0=params["zbins"][redshiftIndex][0],
        z1=params["zbins"][redshiftIndex][1],
    )
    hickle.dump(
        atlas,
        os.path.join(pregridPath, pregridName),
        compression="gzip",
        compression_opts=9,
    )


if __name__ == "__main__":

    params = getParams(runVersion="test")

    for redshiftIndex in range(len(params["zbins"])):
        main(params=params, redshiftIndex=redshiftIndex, plotDist=True)
        combinePregrids(params=params, redshiftIndex=redshiftIndex)
