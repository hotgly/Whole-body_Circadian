# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:15:22 2019

@authors: ShanY, AbelJ

This is just a test file to try to get all the times lined up correctly.
"""
from __future__ import division

import numpy as np
import scipy as sp
import pandas as pd
import dateutil
from datetime import datetime

from matplotlib import gridspec
import matplotlib.pyplot as plt
from LocalImports import PlotOptions as plo
from LocalImports import DecayingSinusoid as ds
from LocalImports import WholeBodyRecording as wbr

reload(wbr)

# setup the data
input_folder = 'Data/M1_030119P2R2GALBWT_Piper1/'

# set up the TH data
imaging_start = '2019-03-12 17:08:01' #time stamp in format like 19/03/01 17:19:30
imaging_interval = '2 min'
m1g_file = input_folder+'031219_pump_light_green_liver.xls'
m1r_file = input_folder+'031219_pump_light_red_body.xls'

# and the temp and humidity file
TH_file = '190301_P2R2GALBWT_Temperature_humidity.xls'
# and actogram file
actogram_file = '190301_ALB_P2R2G_homo_water_Actogram_Graph Data.csv'

# get the mouse set up
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

# remove imaging times
mouse1.excise_imaging_times(intervals_1, cooldown_ext=5)

# add temp/humidity
mouse1.import_temperature_humidity(input_folder+TH_file)

# add the activity
mouse1.import_actogram(input_folder+actogram_file)

# try the processing of imaging data - use the restricted values
mouse1.process_imaging_data('xr','ryr','gyr')

# try the processing of temp/hum data
mouse1.process_temp_hum_data()

# try the processing of activity data
mouse1.process_activity_data()



# Imaging plots
plo.PlotOptions(ticks='in')

plt.figure()
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['ryr'], color='h',
         label= 'red')
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['ryr_hp'], color='hl', 
         label= 'red hp')
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['ryr_hpb'], color='hl',
         label= 'red hp baseline', ls=':')
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['ryr_es'], color='k', 
         label= 'red hp denoise')
#plt.plot(mouse1.sinusoids['red']['ts'], mouse1.sinusoids['red']['sine_data'], 
#         color='white', label= 'red sine fit')
plt.xlabel('Time (h)')
plt.ylabel('Bioluminescence (AU)')
plt.legend()

plt.figure()
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['gyr'], color='i', 
         label= 'green', ls=':')
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['gyr_hp'], color='il', 
         label= 'green hp')
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['gyr_hpb'], color='il', 
         label= 'green hp baseline', ls=':')
plt.plot(mouse1.imaging['xr_UT'], mouse1.imaging['gyr_es'], color='k', 
         label= 'green hp denoise')
#plt.plot(mouse1.sinusoids['green']['ts'], mouse1.sinusoids['green']['sine_data'], 
#         color='white', label= 'green sine fit')
plt.xlabel('Time (h)')
plt.ylabel('Bioluminescence (AU)')
plt.legend()


# temperature and humidity
plt.figure()
plt.plot(mouse1.TH['x_UT'], mouse1.TH['temp'], color='j', 
         label= 'temp')
plt.plot(mouse1.TH['x_UT'], mouse1.TH['temp_hp'], color='jl', 
         label= 'temp hp')
plt.plot(mouse1.TH['x_UT'], mouse1.TH['temp_hpb'], color='jl', 
         label= 'temp hp baseline', ls=':')
plt.plot(mouse1.TH['x_UT'], mouse1.TH['temp_es'], color='k', 
         label= 'temp hp denoise')
plt.ylabel('Temperature ($^\circ$C)')
plt.xlabel('Time (h)')
plt.legend()

plt.figure()
plt.plot(mouse1.TH['x_UT'], mouse1.TH['hum'], color='f', 
         label= 'hum')
plt.plot(mouse1.TH['x_UT'], mouse1.TH['hum_hp'], color='fl', 
         label= 'hum hp')
plt.plot(mouse1.TH['x_UT'], mouse1.TH['hum_hpb'], color='fl',
         label= 'hum hp baseline', ls=':')
plt.plot(mouse1.TH['x_UT'], mouse1.TH['hum_es'], color='k', 
         label= 'hum hp denoise')
plt.ylabel('Humidity (\%)')
plt.xlabel('Time (h)')
plt.legend()

# activity
plt.figure()
plt.plot(mouse1.activity['x_UT'], mouse1.activity['activity_zero'], color='gray', 
         label= 'act')
plt.plot(mouse1.activity['x_UT'], mouse1.activity['activity_es'], color='k', 
         label= 'act denoise')
plt.plot(mouse1.sinusoids['act']['ts'], mouse1.sinusoids['act']['sine_data'], 
         color='white', label= 'sine fit')
plt.ylabel('Mean-subtracted Activity')
plt.xlabel('Time (h)')
plt.legend()
