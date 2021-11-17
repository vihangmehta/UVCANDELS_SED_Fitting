from useful import *
from utils import *


def plot_redshift_comparison(cat,old,new):

    fig, (ax1, ax2) = plt.subplots(2,
                                   1,
                                   figsize=(20, 10),
                                   dpi=75,
                                   tight_layout=False)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.08,top=0.98,hspace=0.)

    ax1.scatter(1 + cat["zphot"],
                (old["zfit"] - cat["zphot"]) / (1 + cat["zphot"]),
                c='tab:red',
                s=5,
                lw=0,
                alpha=0.4)
    ax1.text(0.005,
             0.98,
             "Old Pregrids",
             va='top',
             ha='left',
             fontsize=20,
             fontweight=600,
             color='tab:red',
             transform=ax1.transAxes)

    ax2.scatter(1 + cat["zphot"],
                (new["zfit"] - cat["zphot"]) / (1 + cat["zphot"]),
                c='tab:blue',
                s=5,
                lw=0,
                alpha=0.4)
    ax2.text(0.005,
             0.98,
             "New Pregrids",
             va='top',
             ha='left',
             fontsize=20,
             fontweight=600,
             color='tab:blue',
             transform=ax2.transAxes)

    ticks = np.arange(11)
    for ax in [ax1,ax2]:
        ax.axhline(0,color='k')
        ax.grid(linestyle=':', linewidth='0.5', color='k')
        ax.set_xlim(1,11)
        ax.set_ylim(-0.069,0.069)
        ax.set_xscale("log")
        ax.set_xticks(1+ticks)
        ax.set_xticklabels(ticks)
        [label.set_fontsize(15) for label in ax.get_xticklabels()+ax.get_yticklabels()]

    ax2.set_xlabel("Photo-z",fontsize=18)
    fig.text(0.015,0.54,"$\\Delta(z)/(1+z)$",va="center",ha="center",fontsize=18,color='k',rotation=90)
    [label.set_visible(False) for label in ax1.get_xticklabels()]

def plot_pars_comparison(cat,old,new):

    fig, (ax1, ax2) = plt.subplots(1,
                                   2,
                                   figsize=(15, 7),
                                   dpi=75,
                                   tight_layout=True)

    ax1.scatter(old["logM"],new["logM"],
                c='k',
                s=5,
                lw=0,
                alpha=0.3)

    ax2.scatter(old["logSFR100"],new["logSFR100"],
                c='k',
                s=5,
                lw=0,
                alpha=0.3)

    ax1.set_xlabel("old log(M)",fontsize=18)
    ax1.set_ylabel("new log(M)",fontsize=18)
    ax2.set_xlabel("old log(SFR)",fontsize=18)
    ax2.set_ylabel("new log(SFR)",fontsize=18)
    ax1.set_xlim(7,12.5)
    ax2.set_xlim(-3,2)

    for ax in [ax1,ax2]:
        ax.set_ylim(ax.get_xlim())
        ax.set_aspect(1)
        ax.plot([-10,20],[-10,20],color='k')
        ax.grid(linestyle=':', linewidth='0.5', color='k')
        [label.set_fontsize(15) for label in ax.get_xticklabels()+ax.get_yticklabels()]


if __name__ == '__main__':

    cat = getCatalog(field="egs")
    old = fitsio.getdata('output/old/egs_uvc_v0_out.fits')
    new = fitsio.getdata('output/egs_uvc_v1_out.fits')

    # plot_redshift_comparison(cat=cat,old=old,new=new)
    plot_pars_comparison(cat=cat,old=old,new=new)
    plt.show()
