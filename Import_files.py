# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:15:22 2019

@authors: ShanY, AbelJ

This is just a test file to try to get all the times lined up correctly.
"""

import numpy as np
import scipy as sp
import pandas as pd
import dateutil
from datetime import datetime

from matplotlib import gridspec
import matplotlib.pyplot as plt
from LocalImports import PlotOptions as plo
from LocalImports import DecayingSinusoid as ds
from LocalImports import ProcessingFunctions_20180326 as pf


# setup the data
input_folder = 'Data/M1_030119P2R2GALBWT_Piper1/'

# set up the imaging data
imaging_start = '2019-03-12 17:08:01' #time stamp in format like 19/03/01 17:19:30
imaging_interval = '2 min'

# collect the data
red_file = np.genfromtxt(input_folder+'031219_pump_light_red_body.xls')
green_file = np.genfromtxt(input_folder+'031219_pump_light_green_liver.xls')
imaging_times = red_file[1:,0]
assert len(imaging_times)==len(green_file[1:,0]), "Imaging files of unequal length."

# do we need to remove outliers? or can we just use the LSPgram to get it.....
ry = red_file[1:,1]
gy = green_file[1:,1]
xi = pd.date_range(imaging_start, periods=len(ry), freq=imaging_interval)

# options to try: forward-backward butterworth, wavelet transform, LSPgram?
#lpx, lpy = pf.butterworth_lowpass(gx, np.vstack([ry,gy]).T, cutoff_period=4) 
# butterworth does not work, ugh
# LSPgram - also does not seem to work
#pers, pgram_data, circadian_peaks, circadian_peak_periods, rhythmic_or_not = pf.LS_pgram(gx, np.vstack([ry,gy]).T)

# let's just extract times
def extract_times(x, y1, y2, intervals, cooldown_ext=8):
    """
    Cuts times where light is on from the data. intervals are given 
    in 'YYYY-MM-DD XX:XX:XX' format.

    cooldown_ext is number of datapoints.
    """
    # start with full dataset
    xr = x
    y1r = y1
    y2r = y2
    dd='13'
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
    return xr, y1r, y2r 

# extract the data where the lights are on from days 12-18
xr = xi
gyr = gy
ryr = ry
for day in [12,13,14,15,16,17,18]:
    dd = str(day)
    intervals_1 = [# 12 pulses per day
                 ['2019-03-'+dd+' 07:52:00', '2019-03-'+dd+' 08:24:00'],
                 ['2019-03-'+dd+' 08:52:00', '2019-03-'+dd+' 09:24:00'],
                 ['2019-03-'+dd+' 09:52:00', '2019-03-'+dd+' 10:24:00'],
                 ['2019-03-'+dd+' 10:52:00', '2019-03-'+dd+' 11:24:00'],
                 ['2019-03-'+dd+' 11:52:00', '2019-03-'+dd+' 12:24:00'],
                 ['2019-03-'+dd+' 12:52:00', '2019-03-'+dd+' 13:24:00'],
                 ['2019-03-'+dd+' 13:52:00', '2019-03-'+dd+' 14:24:00'],
                 ['2019-03-'+dd+' 14:52:00', '2019-03-'+dd+' 15:24:00'],
                 ['2019-03-'+dd+' 15:52:00', '2019-03-'+dd+' 16:24:00'],
                 ['2019-03-'+dd+' 16:52:00', '2019-03-'+dd+' 17:24:00'],
                 ['2019-03-'+dd+' 17:52:00', '2019-03-'+dd+' 18:24:00'],
                 ['2019-03-'+dd+' 18:52:00', '2019-03-'+dd+' 19:24:00']
                 ]
    xr, ryr, gyr = extract_times(xr, ryr, gyr, intervals_1, cooldown_ext=5)

# and for days 18 onward
for day in [19,20,21,22,23,24]:
    dd = str(day)
    intervals_2 = [# 4 pulses per day
                 ['2019-03-'+dd+' 07:44:00', '2019-03-'+dd+' 08:32:00'],
                 ['2019-03-'+dd+' 10:44:00', '2019-03-'+dd+' 11:32:00'],
                 ['2019-03-'+dd+' 13:44:00', '2019-03-'+dd+' 14:32:00'],
                 ['2019-03-'+dd+' 16:44:00', '2019-03-'+dd+' 17:32:00']
                 ]
    xr, ryr, gyr = extract_times(xr, ryr, gyr, intervals_2, cooldown_ext=0)

plt.figure()
plt.plot(xr,ryr,'rs', label='red')
plt.plot(xr,gyr,'g.', label='green')
plt.legend()


# temperature and humidity start can be pulled from excel file
TH_file = '190301_P2R2GALBWT_Temperature_humidity.xls'
#contains time axis and time interval is 15 minutes

# load the files
TH_pd = pd.read_excel(input_folder+TH_file, usecols=[0,1,2,3],
                      names=['index','date','temp','humidity'])
TH_pd=TH_pd.drop(range(18))
#if set up as example file, then this is the correct number of rows to drop

# collect the x values for temp and humidity
xth = pd.DatetimeIndex(pd.to_datetime(TH_pd['date'], yearfirst=True))
# only start where imaging starts
imgstart  = pd.to_datetime(imaging_start)
idx = np.where(xth>=imgstart)[0][0]
xthr = xth[idx:]
tempr = np.array(TH_pd['temp'], dtype=np.float)[idx:]
humr = np.array(TH_pd['humidity'], dtype=np.float)[idx:]

plt.figure()
plt.plot(xthr,tempr, label='Temp (C)')
plt.plot(xthr,humr, label='Humidity')
plt.legend()

# actogram is in 1min resolution, constains start date and times
actogram_file = '190301_ALB_P2R2G_homo_water_Actogram_Graph Data.csv'
actogram_interval = 1 # in minutes

#contains time axis and time intercal is 1 minute
act_pd = pd.read_csv(input_folder+actogram_file,header=None)
total_days = act_pd.shape[1]-1
act_start = pd.to_datetime(act_pd[1][1][3:]+' 00:00:00', dayfirst=True)

# assemble all the columns
intervals = int(60/actogram_interval*24)*total_days
xa = pd.date_range(act_start, periods=intervals, freq=str(actogram_interval)+' min')
activity = np.array(act_pd.iloc[np.arange(8,8+(60/actogram_interval*24)),
                                np.arange(1,1+total_days)], dtype=np.float).flatten('F')
idx = np.where(xa>=imgstart)[0][0]
xar = xa[idx:]
actr = activity[idx:]
actr_15bin = [np.mean(actr[idx*15:(idx+1)*15]) for idx in np.arange(int(len(actr)/15))]

plt.figure()
plt.plot(xar[5::15][:-1],actr_15bin,'k', label = 'activity, 15min mean')
plt.legend()


# and finally the feeding
feeding_file =['Feeder_YS_022819_ALB_F097'] #txt format





# repeat exact way with new class 
from LocalImports import WholeBodyRecording as wbr
reload(wbr)

m1g_file = input_folder+'031219_pump_light_green_liver.xls'
m1r_file = input_folder+'031219_pump_light_red_body.xls'
mouse1 = wbr.WholeBodyRecording(m1r_file, m1g_file, imaging_start, imaging_interval)

# cut the light intervals
intervals_1 = []
for day in [12,13,14,15,16,17,18]:
    dd = str(day)
    intervals_1 = intervals_1 + [# 12 pulses per day
                 ['2019-03-'+dd+' 07:52:00', '2019-03-'+dd+' 08:24:00'],
                 ['2019-03-'+dd+' 08:52:00', '2019-03-'+dd+' 09:24:00'],
                 ['2019-03-'+dd+' 09:52:00', '2019-03-'+dd+' 10:24:00'],
                 ['2019-03-'+dd+' 10:52:00', '2019-03-'+dd+' 11:24:00'],
                 ['2019-03-'+dd+' 11:52:00', '2019-03-'+dd+' 12:24:00'],
                 ['2019-03-'+dd+' 12:52:00', '2019-03-'+dd+' 13:24:00'],
                 ['2019-03-'+dd+' 13:52:00', '2019-03-'+dd+' 14:24:00'],
                 ['2019-03-'+dd+' 14:52:00', '2019-03-'+dd+' 15:24:00'],
                 ['2019-03-'+dd+' 15:52:00', '2019-03-'+dd+' 16:24:00'],
                 ['2019-03-'+dd+' 16:52:00', '2019-03-'+dd+' 17:24:00'],
                 ['2019-03-'+dd+' 17:52:00', '2019-03-'+dd+' 18:24:00'],
                 ['2019-03-'+dd+' 18:52:00', '2019-03-'+dd+' 19:24:00']
                 ]

for day in [19,20,21,22,23,24]:
    dd = str(day)
    intervals_1 = intervals_1 +[# 4 pulses per day
                 ['2019-03-'+dd+' 07:44:00', '2019-03-'+dd+' 08:32:00'],
                 ['2019-03-'+dd+' 10:44:00', '2019-03-'+dd+' 11:32:00'],
                 ['2019-03-'+dd+' 13:44:00', '2019-03-'+dd+' 14:32:00'],
                 ['2019-03-'+dd+' 16:44:00', '2019-03-'+dd+' 17:32:00']
                 ]

# now will this work?
mouse1.excise_imaging_times(intervals_1, cooldown_ext=5)
# yes

# and the TH file?
mouse1.import_temperature_humidity(input_folder+
                    '190301_P2R2GALBWT_Temperature_humidity.xls')

# and the activity
mouse1.import_actogram(input_folder+
                '190301_ALB_P2R2G_homo_water_Actogram_Graph Data.csv')

# try the hp detrend
mouse1.process_imaging_data('xr','ryr','gyr')


# HP filter
plt.figure()
plt.plot(mouse1.imaging['xr_hp'], mouse1.imaging['ryr'], color='red', 
         label= 'red')
plt.plot(mouse1.imaging['xr_hp'], mouse1.imaging['ryr_hp'], color='maroon', 
         label= 'red hp')
plt.plot(mouse1.imaging['xr_hp'], mouse1.imaging['ryr_hpb'], color='maroon', 
         label= 'red hp baseline', ls=':')
plt.plot(mouse1.imaging['xr_es'], mouse1.imaging['ryr_es'], color='k', 
         label= 'red hp denoise')
plt.plot(mouse1.imaging['xr_b'], mouse1.imaging['ryr_b'], color='k', ls=':',
         label= 'red hp butter')
plt.legend()

plt.figure()
plt.plot(mouse1.imaging['xr_hp'], mouse1.imaging['gyr'], color='green', 
         label= 'green', ls=':')
plt.plot(mouse1.imaging['xr_hp'], mouse1.imaging['gyr_hp'], color='gold', 
         label= 'green hp')
plt.plot(mouse1.imaging['xr_hp'], mouse1.imaging['gyr_hpb'], color='gold', 
         label= 'green hp baseline', ls=':')
plt.plot(mouse1.imaging['xr_es'], mouse1.imaging['gyr_es'], color='k', 
         label= 'green hp denoise')
plt.plot(mouse1.imaging['xr_b'], mouse1.imaging['gyr_b'], color='k', ls=':',
         label= 'green hp butter')
plt.legend()