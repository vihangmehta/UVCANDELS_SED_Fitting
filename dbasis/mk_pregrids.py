from useful import *
from utils import *
from utils_dbasis import *
from plotter import *


def generate_pregrid(Npts,
                     priors,
                     gridpath,
                     gridname,
                     filter_list=filter_list,
                     filter_dir=filter_dir):

    import dense_basis as db

    db.generate_atlas(N_pregrid=Npts,
                      priors=priors,
                      fname=gridname,
                      filter_list=filter_list,
                      filt_dir=filter_dir,
                      path=gridpath,
                      store=True)


def worker(pg_index, params, priors):

    pregrid_fname = params["prefix"] + "_chunk%03d"%(pg_index+1)

    generate_pregrid(Npts=params["Npts_iter"],
                     priors=priors,
                     gridpath=pregrid_path,
                     gridname=pregrid_fname,
                     filter_list=filter_list,
                     filter_dir=filter_dir)


def main(params, priors):

    from schwimmbad import MultiPool
    from functools import partial

    with MultiPool(processes=params["Nproc"]) as pool:
        values = list(
            pool.map(partial(worker, params=params, priors=priors), range(params["Nproc"])))


def combine_pregrids(params):

    import hickle

    atlas = {}

    pregrid_list = sorted(glob.glob(os.path.join(pregrid_path,params["prefix"]+"_chunk*")))

    for i,pg_fname in enumerate(pregrid_list):

        print("\rProcessing pregrid chunk#%3d/%3d ... " % (i+1, len(pregrid_list)),end="")

        atlas_ = hickle.load(pg_fname)
        for key in atlas_:
            if key != "norm_method":
                if key in atlas:
                    atlas[key] = np.append(atlas[key], atlas_[key], axis=0)
                else:
                    atlas[key] = atlas_[key]

    print("done.")

    fname = params["prefix"] + '_' \
            + str(params["Npts"]) \
            + '_Nparam_' \
            + str(params["Npars"]) \
            + '.dbatlas'
    fname = os.path.join(pregrid_path, fname)
    hickle.dump(atlas, fname, compression='gzip', compression_opts=9)

if __name__ == '__main__':

    params = get_run_params(run_version="test")
    priors = get_priors(Npars=params["Npars"],zmax=params["zmax"])

    main(params=params, priors=priors)
    combine_pregrids(params=params)

    # plot_prior_dist(priors,
    #     savename=os.path.join(plotdir_path, "priors.png"))

