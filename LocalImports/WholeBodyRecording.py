# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2017

@author: john abel
Module set up to perform analysis for Yongli Shan.
"""


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

import collections

from . import PlotOptions as plo
from . import DecayingSinusoid as ds
from . import Bioluminescence as bl


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
            timespan of each image, in formate '2 min'
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