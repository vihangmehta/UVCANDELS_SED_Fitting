import os
import sys
import glob
import time
import subprocess
import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy.lib.recfunctions as rfn
from scipy.interpolate import interp1d
from scipy.integrate import simps

cwd = "/home/scarlata/mehta074/Documents/UVCANDELS/sedfit"
light = 3e18

fields = ["goodsn","goodss","cosmos","egs"]


class FilterSet():
    def __init__(self):

        self.filters = np.array([
            # HST -- WFC3/UVIS2
            'F275W',
            # HST -- ACS/WFC1
            'F435W',
            'F606W',
            'F775W',
            'F814W',
            'F850LP',
            # HST -- WFC3/IR
            'F098M',
            'F105W',
            'F125W',
            'F140W',
            'F160W',
            # Ground-based u-band
            'LBC_U',
            'CTIO_U',
            'VIMOS_U',
            # CFHT ugriz
            'CFHT_u',
            'CFHT_g',
            'CFHT_r',
            'CFHT_i',
            'CFHT_z',
            # Subaru BVgriz
            'Subaru_B',
            'Subaru_V',
            'Subaru_gp',
            'Subaru_rp',
            'Subaru_ip',
            'Subaru_zp',
            # UVISTA YJHKs
            'UVISTA_Y',
            'UVISTA_J',
            'UVISTA_H',
            'UVISTA_Ks',
            # WIRCAM JHK
            'WIRCAM_J',
            'WIRCAM_H',
            'WIRCAM_Ks',
            # NEWFIRM JHK
            'NEWFIRM_J1',
            'NEWFIRM_J2',
            'NEWFIRM_J3',
            'NEWFIRM_H1',
            'NEWFIRM_H2',
            'NEWFIRM_K',
            # Ground-based K-band
            'HAWKI_Ks',
            'MOIRCS_Ks',
            'ISAAC_Ks',
            # IRAC
            'IRAC_CH1',
            'IRAC_CH2',
            'IRAC_CH3',
            'IRAC_CH4'
        ])

        self.check_response_files()

        self.pivotWL = dict(
            zip(self.filters,
                [self.calc_pivotWL(filt) for filt in self.filters]))

        self.width = dict(
            zip(self.filters,
                [self.calc_width(filt) for filt in self.filters]))

    def get_response_fname(self, filt):

        return "filters/{:s}.pb".format(filt)

    def get_response(self, filt):

        return np.genfromtxt(self.get_response_fname(filt),
                             dtype=[("wave", float), ("throughput", float)])

    def check_response_files(self):

        check = [
            not os.path.isfile(self.get_response_fname(filt))
            for filt in self.filters
        ]
        if any(check):
            print("No filter response found for: ", self.filters[check])

    def calc_width(self, filt):

        response = self.get_response(filt)
        return simps(response["throughput"], response["wave"]) / np.max(
            response["throughput"])

    def calc_pivotWL(self, filt):

        response = self.get_response(filt)
        return np.sqrt(
            simps(response["throughput"], response["wave"]) /
            simps(response["throughput"] / response["wave"] / response["wave"],
                  response["wave"]))

    def plot_responses(self):

        fig, ax = plt.subplots(1,
                               1,
                               figsize=(20, 8),
                               dpi=75,
                               tight_layout=True)
        for filt in self.filters:
            response = self.get_response(filt)
            ax.plot(response["wave"], response["throughput"], c='k')
        ax.set_xscale("log")
        ax.set_xlim(2e3, 1e5)


class JohnsonFilterSet():
    def __init__(self):

        self.filters = np.array(
            ["NUV", "FUV", "U", "B", "V", "R", "I", "J", "K"])
        self.check_response_files()

        self.pivotWL = dict(
            zip(self.filters,
                [self.calc_pivotWL(filt) for filt in self.filters]))

        self.width = dict(
            zip(self.filters,
                [self.calc_width(filt) for filt in self.filters]))

    def get_response_fname(self, filt):

        return "filters_johnson/{:s}.pb".format(filt)

    def get_response(self, filt):

        return np.genfromtxt(self.get_response_fname(filt),
                             dtype=[("wave", float), ("throughput", float)])

    def check_response_files(self):

        check = [
            not os.path.isfile(self.get_response_fname(filt))
            for filt in self.filters
        ]
        if any(check):
            print("No filter response found for: ", self.filters[check])

    def calc_width(self, filt):

        response = self.get_response(filt)
        return simps(response["throughput"], response["wave"]) / np.max(
            response["throughput"])

    def calc_pivotWL(self, filt):

        response = self.get_response(filt)
        return np.sqrt(
            simps(response["throughput"], response["wave"]) /
            simps(response["throughput"] / response["wave"] / response["wave"],
                  response["wave"]))

    def plot_responses(self):

        fig, ax = plt.subplots(1,
                               1,
                               figsize=(20, 8),
                               dpi=75,
                               tight_layout=True)
        for filt in self.filters:
            response = self.get_response(filt)
            ax.plot(response["wave"], response["throughput"], c='k')
        ax.set_xscale("log")
        ax.set_xlim(2e3, 1e5)


def force_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except (OSError, e):
        if e.errno == errno.EEXIST:
            os.remove(dst)
            os.symlink(src, dst)


def runBashCommand(call, cwd, verbose=True):
    """
    Generic function to execute a bash command
    """
    start = time.time()
    if isinstance(verbose, str):
        f = open(verbose, "w")
        p = subprocess.Popen(call, stdout=f, stderr=f, cwd=cwd, shell=True)
    elif verbose == True:
        print("Running command:<{:s}> in directory:<{:s}> ... ".format(
            call, cwd),
              flush=True)
        p = subprocess.Popen(call,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             cwd=cwd,
                             shell=True)
        for line in iter(p.stdout.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    else:
        devnull = open(os.devnull, "w")
        p = subprocess.Popen(call,
                             stdout=devnull,
                             stderr=devnull,
                             cwd=cwd,
                             shell=True)
    p.communicate()
    p.wait()
    return time.time() - start
