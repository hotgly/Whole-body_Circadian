# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2017

@author: john abel
Module set up to perform analysis for Yongli Shan.
"""


import numpy  as np
import pandas as pd
from scipy import signal, interpolate, optimize, sparse
from scipy.sparse import dia_matrix, eye as speye
from scipy.sparse.linalg import spsolve
import spectrum
import matplotlib.pyplot as plt

import collections

from . import PlotOptions as plo
from . import DecayingSinusoid as ds
from . import Bioluminescence as blu


class WholeBodyRecording(object):
    """
    Class to analyze time series data for whole body PER2iLuc recordings.
    """

    def __init__(self, red_file, green_file, imaging_start, imaging_interval,
                    name=None):
        """
        Parameters
        ----------
        red_data : np.ndarray
            should contain a single PER2iLuc time series for analysis.
        green_data : np.ndarray
            should contain a single PER2iLuc time series for analysis.
        imaging_start : str
            start of the imaging, in format '2019-03-12 17:08:01'
        imaging_interval : str
            timespan of each image, in formate 'XYZ min'
        name : str
            just a name for the dataset
        """
        if name is not None:
            self.name = name


        red_f = np.genfromtxt(red_file)
        green_f = np.genfromtxt(green_file)
        imaging_times = red_f[1:,0]
        assert len(imaging_times)==len(green_f[1:,0]), "Imaging files of unequal length."

        # do we need to remove outliers? or can we just use the LSPgram to get it.....
        ry = red_f[1:,1]
        gy = green_f[1:,1]
        xi = pd.date_range(imaging_start, periods=len(ry), freq=imaging_interval)

        imaging = {}
        imaging['ry'] = ry
        imaging['gy'] = gy
        imaging['xi'] = xi
        self.imaging = imaging
        self.imaging_start = imaging_start
        self.imaging_interval = imaging_interval
    
    def excise_imaging_times(self, intervals, t_pre='xr', red_pre='ryr', green_pre='gyr', t_post='xr', red_post='ryr', green_post='gyr', cooldown_ext=5):
        """
        Cuts times. By default, it operates on and modifies xr, ryr, gyr. If these do
        not exist, they are created.
        """

        # if pre is not set, use the already-truncated data
        # if already-truncated data does not exist, take the raw
        if t_pre not in self.imaging.keys():
            self.imaging[t_pre] = self.imaging['xi']

        if red_pre not in self.imaging.keys():
            self.imaging[red_pre] = self.imaging['ry']

        if green_pre not in self.imaging.keys():
            self.imaging[green_pre] = self.imaging['gy']
        
        # get data for editing
        xr = self.imaging[t_pre]
        y1r = self.imaging[red_pre]
        y2r = self.imaging[green_pre]
        
        # cut each pulse
        for pulse in intervals:
            lightson  = pd.to_datetime(pulse[0])
            lightsoff = pd.to_datetime(pulse[1])
            try:
                idx = np.where(xr>=lightson)[0][0]
                idx2 = np.where(xr>=lightsoff)[0][0]+cooldown_ext
                y1r = np.hstack([y1r[:idx], y1r[idx2:]])
                y2r = np.hstack([y2r[:idx], y2r[idx2:]])
                xr = xr[:idx].append(xr[idx2:])
            except IndexError:
                # if there is no data that late
                pass

        # by default drop it in these
        if t_post is None:
            t_post = 'xr'
        if red_post is None:
            red_post = 'ryr'
        if green_post is None:
            green_post is 'gyr'

        self.imaging[red_post] = y1r
        self.imaging[green_post] = y2r
        self.imaging[t_post] = xr

    def import_temperature_humidity(self, filename, start_time='imaging', droplines=18):
        """
        Imports the temperature and humidity file at filename.

        start_time is either "imaging", None, or a date in the format 
        '2019-03-12 17:08:01'
        """

        # load the files
        TH_pd = pd.read_excel(filename, usecols=[0,1,2,3],
                              names=['index','date','temp','humidity'])
        TH_pd = TH_pd.drop(range(droplines))
        # if set up as example file, then this is 
        # the correct number of rows to drop

        # collect the x values for temp and humidity
        xth = pd.DatetimeIndex(pd.to_datetime(TH_pd['date'], yearfirst=True))

        # only start where imaging starts
        if start_time=='imaging':
            imgstart  = pd.to_datetime(self.imaging_start)
        elif start_time==None:
            imgstart = pd.to_datetime('2010-01-01 00:00:00')
        else:
            try:
                imgstart = dp.to_datetime(start_time)
            except:
                imgstart = pd.to_datetime('2010-01-01 00:00:00')
                print "Date format not understood, leaving all times."
        idx = np.where(xth>=imgstart)[0][0]
        xthr = xth[idx:]
        tempr = np.array(TH_pd['temp'], dtype=np.float)[idx:]
        humr = np.array(TH_pd['humidity'], dtype=np.float)[idx:]

        TH = {}
        TH['temp'] = tempr
        TH['hum'] = humr
        TH['x'] = xthr
        self.TH = TH

    def import_actogram(self, filename, start_time='imaging', actogram_interval=1):
        """
        Imports the actogram file at filename.

        actogram_interval is in minutes

        start_time is either "imaging", None, or a date in the format 
        '2019-03-12 17:08:01'
        """
        act_pd = pd.read_csv(filename, header=None)
        total_days = act_pd.shape[1]-1
        act_start = pd.to_datetime(act_pd[1][1][3:]+' 00:00:00', dayfirst=True)

        # only start where imaging starts
        if start_time=='imaging':
            imgstart  = pd.to_datetime(self.imaging_start)
        elif start_time==None:
            imgstart = pd.to_datetime('2010-01-01 00:00:00')
        else:
            try:
                imgstart = dp.to_datetime(start_time)
            except:
                imgstart = pd.to_datetime('2010-01-01 00:00:00')
                print "Date format not understood, leaving all times."

        # assemble all the columns
        intervals = int(60/actogram_interval*24)*total_days
        xa = pd.date_range(act_start, periods=intervals, freq=str(actogram_interval)+' min')
        activity = np.array(
                    act_pd.iloc[np.arange(8,8+(60/actogram_interval*24)),
                    np.arange(1,1+total_days)], dtype=np.float
                             ).flatten('F')
        idx = np.where(xa>=imgstart)[0][0]
        xar = xa[idx:]
        actr = activity[idx:]

        actogram = {}
        actogram['x'] = xar
        actogram['activity'] = actr
        self.actogram = actogram

    def process_imaging_data(self, xname, redname, greenname):
        """
        Performs the analysis of the PER2iLuc data. The xname, redname, 
        greenname arguments tell which of the dict to look at.
        """
        x = self.imaging[xname]
        red = self.imaging[redname]
        green = self.imaging[greenname]

        # convert interval into float
        interval_float = np.float(self.imaging_interval.split()[0])
        times = np.arange(len(red))*interval_float/60 # I think

        hpt, hp_red, hp_redb = hp_detrend(times, red)
        hpt, hp_green, hp_greenb = hp_detrend(times, green)

        self.imaging[redname+'_hp'] = hp_red
        self.imaging[greenname+'_hp'] = hp_green
        self.imaging[redname+'_hpb'] = hp_redb
        self.imaging[greenname+'_hpb'] = hp_greenb
        self.imaging[xname+'_hp'] = hpt

        # now the eigensmooth
        et, ered, evalred = eigensmooth(hpt, hp_red)
        et, egreen, evalgreen = eigensmooth(hpt, hp_green)

        self.imaging[xname+'_es'] = et
        self.imaging[redname+'_es'] = ered
        self.imaging[greenname+'_es'] = egreen

        # what about a butter smooth
        bt, bred = butterworth_lowpass(hpt, hp_red)
        bt, bgreen = butterworth_lowpass(hpt, hp_green)

        self.imaging[xname+'_b'] = bt
        self.imaging[redname+'_b'] = bred
        self.imaging[greenname+'_b'] = bgreen

# functions


def hp_detrend(x, y, est_period=24., ret="both", a=0.05):
    """ Detrend the data using a hodrick-prescott filter. If ret ==
    "mean", return the detrended mean of the oscillation. Estimated
    period and 'a' parameter are responsible for determining the optimum
    smoothing parameter """

    x = np.asarray(x)
    y = np.asarray(y)

    # yt, index = timeseries_boundary(y, opt_b='mir', bdetrend=False)

    # As recommended by Ravn, Uhlig 2004, a calculated empirically
    num_periods = (x.max() - x.min())/est_period
    points_per_period = len(x)/num_periods
    w = a*points_per_period**4


    y_mean = hpfilter(y, w)
    y_detrended = y - y_mean

    if ret == "detrended": return x, y_detrended
    elif ret == "mean": return x, y_mean
    elif ret == "both": return x, y_detrended, y_mean


def hpfilter(X, lamb):
    """ Code to implement a Hodrick-Prescott with smoothing parameter
    lambda. Code taken from statsmodels python package (easier than
    importing/installing, https://github.com/statsmodels/statsmodels """

    X = np.asarray(X, float)
    if X.ndim > 1:
        X = X.squeeze()
    nobs = len(X)
    I = speye(nobs,nobs)
    offsets = np.array([0,1,2])
    data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    K = dia_matrix((data, offsets), shape=(nobs-2,nobs))

    trend = spsolve(I+lamb*K.T.dot(K), X, use_umfpack=True)
    return trend


def eigensmooth(times, data, ev_threshold=0.05, dim=600, min_ev=2):
    """
    Uses an eigendecomposition to keep only elements with >threshold of the
    data. Then it returns the denoised data.

    Notes: This should take the covariance matrix of the data using fwd-backward method. Then it
    eigendecomposes it, then it finds the biggest (2+) eigenvalues and returns only the
    components associated with them.
    For an intuitive explanation of how this works:
    http://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/
    """
    # remove nans from times, d1. keep nans in d0
    t1 = times[~np.isnan(data)]
    d1 = np.copy(data[~np.isnan(data)])

    # using spectrum to get the covariance matrix
    X = spectrum.linalg.corrmtx(d1, dim-1, method='autocorrelation')
    # the embedding matrix
    X = (1/np.sqrt(len(X.T)))*np.array(X)
    XT = np.transpose(X)
    C = XT.dot(X)

    # now eigendecompose
    evals, Q = np.linalg.eig(C)

    # find evals that matter, use a minimum of 2
    eval_goodness = np.max([2,
                    np.sum(evals/np.sum(evals) >= ev_threshold)])
    QT = np.transpose(Q)

    # and return the reconstruction
    P = QT.dot(XT)
    denoised = np.sum(P[:eval_goodness],0)

    # find alignment - for some reason the signal can be flipped.
    # this catches it
    # truncate the first 24h during alignment
    align, atype = alignment(d1, denoised, d=dim, dstart=96)

    # fix alignment if leading nans
    nanshift=0
    if np.isnan(data[0]):
        nanshift=np.argmin(np.isnan(data))

    # get the correctly-shaped denoised data
    denoised = denoised[align:align+len(d1)]*atype

    return times, denoised, evals


def alignment(original, denoised, d=40, dstart=0):
    """
    The eigensmoothing as written truncates some front-back data, as the
    input data input data is extended. This function realigns the data

    The sign is sometimes flipped on the reconstructed signal.
    This function figures out +/- alignment as well.

    dstart (default=0) tells where to start checking for alignment. Some
    of the data has an artifact, so later datapoints are more reliable
    for finding an alignment (otherwise the artifact will dominate).
    """
    original = original[dstart:]
    denoised = denoised[dstart:]
    errs = np.zeros(d-1)
    for idx in range(d-1):
        errs[idx] = np.linalg.norm(original-denoised[idx:-(d-idx-1)])
    errs_neg = np.zeros(d-1)
    for idx in range(d-1):
        errs_neg[idx] = np.linalg.norm(original+denoised[idx:-(d-idx-1)])
    pos_min = np.min(errs)
    neg_min = np.min(errs_neg)
    if neg_min < pos_min:
        return np.argmin(errs_neg), -1
    else:
        return np.argmin(errs), 1


def butterworth_lowpass(x, y, cutoff_period=8., order=10):
    """ Filter the data with a lowpass filter, removing noise with a
    critical frequency corresponding to the number of hours specified by
    cutoff_period. Assumes a period of 24h, with data in x in the units
    of hours. """

    x = np.asarray(x)
    y = np.asarray(y)
    nyquist = (x[1] - x[0])/2.
    cutoff_freq = 1/((cutoff_period/(x.max() - x.min()))*len(x))

    b, a = signal.butter(order, cutoff_freq/nyquist)
    y_filt = signal.filtfilt(b, a, y)

    return x, y_filt