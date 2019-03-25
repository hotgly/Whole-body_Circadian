# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:15:22 2019

@author: ShanY
"""
import scipy as sp
import pandas as pd
import matplotlib
import imp
matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
from LocalImports import PlotOptions as plo
from LocalImports import Bioluminescence as blu
from LocalImports import DecayingSinusoid as dsin
from LocalImports import ProcessingFunctions_20190321 as pf
imp.reload(pf)

experiment = '030119_P2R2GALBWT'
input_folder = 'Data/scn46_AVPWT_AAVChR2_021219_SGLE3/'
imaging_start = '19/0312 17:08:01' #time stamp in format like 19/03/01 17:19:30
imaging_interval = 2 #minutes
input_ij_file_type_color =[
                          ['031219_pump_light_red_body.xls', experiment, 'Red'],\
                          ['031219_pump_light_green_liver.xls', experiment, 'Green'],\
                          ]
TH_file = ['190301_P2R2GALBWT_Temperature_humidity.xls'] #contains time axis and time interval is 15 minutes
actogram_file = ['190301_ALB_P2R2G_homo_water_Actogram_Graph Data.csv'] #contains time axis and time intercal is 1 minute
feeding_file =['Feeder_YS_022819_ALB_F097'] #txt format
