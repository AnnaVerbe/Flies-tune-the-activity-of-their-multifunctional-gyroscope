# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:33:17 2024

@author: av8889
"""


import numpy as np

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


import matplotlib.pyplot as plt
import os
import cv2
from scipy import signal
from scipy.signal import butter, lfilter, freqz, argrelmax
from scipy import stats

from matplotlib import  image

import tkinter #path
from tkinter import filedialog

from neo.io import AxonIO
import pandas as pd
import pickle


from moviepy.editor import VideoFileClip, clips_array
import moviepy.editor as mpy


#%% pickle format

def save_object(obj, name):
    try:
        with open(str(name) + ".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
 
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


#%%paths

home_directory = os.path.expanduser( '~' )
flydir = home_directory + '/Dropbox/Data_GitHub'
my_path = flydir + '/Fig1to4'

#%%Fig1  #???????
#%% Open data


rawdata_one = load_object(my_path + '/' + 'rawdata_one.pickle')

time  = rawdata_one[:][0]
Lamp_filt = rawdata_one[:][1]
timeCam = rawdata_one[:][2]
base_test_filt = rawdata_one[:][3]
stalk_test_filt = rawdata_one[:][4]
img_stackImj2= rawdata_one[:][5]
Transformed_contours_S= rawdata_one[:][6]
Transformed_contours_B= rawdata_one[:][7]
im_fly= rawdata_one[:][8]

Fig1_panelGH = load_object(my_path + '/' + 'Fig1_panelGH.pickle')


MeanVal= Fig1_panelGH[:][0]
x_d_20000= Fig1_panelGH[:][1]
Mean_logprob_Stalk= Fig1_panelGH[:][2]
x_d_20000= Fig1_panelGH[:][3]
Mean_logprob_Base= Fig1_panelGH[:][4]



#%% Figure - Pannel D and E 
t = 0 


from moviepy.video.io.bindings import mplfig_to_npimage
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from matplotlib.patches import Arrow
from matplotlib.patches import Polygon
from matplotlib import pyplot, image, transforms
from matplotlib.transforms import Affine2D


#Figure and Movie
plt.rc('font', size=12)
plt.rc('axes',linewidth=1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['axes.linewidth'] = 1

fig = plt.figure(figsize=(7,6))
fig.set_facecolor('None') #fig.set_facecolor('w')
fig.subplots_adjust(hspace=.25, wspace=.01)

scales = (0, 5, 0, 5)

gs = fig.add_gridspec(2,1)#12+12+9)
ax1 = plt.subplot(gs[1,0]) 
ax2 = plt.subplot(gs[0,0]) 

halt = cv2.rotate(img_stackImj2[5], cv2.ROTATE_90_CLOCKWISE)
ax1.clear()
  
ax1.imshow(halt, cmap = 'gray')#, clim = (20, 125)), aspect="auto"   
Stalk_c = Polygon(Transformed_contours_S, 20, facecolor='None', edgecolor='r', lw=1) #Draw circle on halt for the base
Base_c = Polygon(Transformed_contours_B, 20, facecolor='None', edgecolor='b', lw=1) #Draw circle on halt for the base    
ax1.add_patch(Base_c)
ax1.add_patch(Stalk_c)
ax1.set_xticks([])
ax1.set_yticks([])
  
############################
#Movie Fly , frame by frame
fly_p = im_fly[150,:,:]  #cv2.imread(movie_path)
   
ax2.clear()
ax2.imshow(fly_p, cmap = 'gray')#, clim = (20, 125)), aspect="auto"
ax2.set_xticks([])

ax2.set_yticks([])

angle_L = Lamp_filt[37818]

r = 180 #length arrow   
dx_m = -r*np.cos(np.radians(angle_L+90+30))
dy_m = r*np.sin(np.radians(angle_L+90+30))

Arrow_c = Arrow(370, 218, dx_m, dy_m, facecolor='None', edgecolor='g', lw=1)

ax2.add_patch(Arrow_c)

ax1.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
ax1.yaxis.label.set_color('k')          #setting up Y-axis label color to blue
ax1.tick_params(axis='x', colors='k')    #setting up X-axis tick color to red
ax1.tick_params(axis='y', colors='k')  #setting up Y-axis tick color to black

ax2.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
ax2.yaxis.label.set_color('k')          #setting up Y-axis label color to blue
ax2.tick_params(axis='x', colors='k')    #setting up X-axis tick color to red
ax2.tick_params(axis='y', colors='k')  #setting up Y-axis tick color to black




#%% Figure - Pannel F


#Figure and Movie
plt.rc('font', size=12)
plt.rc('axes',linewidth=1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['axes.linewidth'] = 1

fig = plt.figure(figsize=(7,6))
fig.set_facecolor('None') 
fig.subplots_adjust(hspace=.05, wspace=0.05)

scales = (0, 5, 0, 5)


ax3 = plt.subplot2grid((3,10), (0,0), rowspan=1,colspan=10) #Left WBA
ax4 = plt.subplot2grid((3,10), (1,0), rowspan=1,colspan=10, sharex=ax3) #Base
ax5 = plt.subplot2grid((3,10), (2,0), rowspan=1,colspan=10, sharex=ax3) #Stalk Fluorescence
ax3.set_facecolor('None')
ax4.set_facecolor('None')
ax5.set_facecolor('None')



############################
#Fluorescence

#Base
ax4.clear()
ax4.plot(timeCam  , ((base_test_filt - np.mean(base_test_filt[0:11]))/np.mean(abs(base_test_filt[0:11])))*100, 'b')  #time, (base_test-np.mean(base_test[0:100]))/np.mean(base_test[0:100]), 'k')  
ax4.set_ylabel(u'ΔF/F(%) dF2 F') #Base
ax4.spines['left'].set_bounds(-40,0)
ax4.set_ylim((-57,10))
ax4.set_yticks([-40,0])
ax4.set_xticks([])
ax4.tick_params(axis='both', which='major', pad=2)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.get_xaxis().set_visible(False)

ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

############################   
#Stalk
ax5.clear() #Help clear the previous line 
ax5.plot(timeCam  , ((stalk_test_filt - np.mean(stalk_test_filt[0:11]))/np.mean(abs(stalk_test_filt[0:11])))*100, 'r')  #time, (base_test-np.mean(base_test[0:100]))/np.mean(base_test[0:100]), 'k')  
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(True)
ax5.get_xaxis().set_visible(True)
ax5.xaxis.set_ticks_position('bottom')
ax5.yaxis.set_ticks_position('left')
ax5.set_ylabel(u'ΔF/F(%) dF1 F')#stalk

ax5.spines['left'].set_bounds(-40,0)
ax5.set_yticks([-40,0])
ax5.set_ylim((-60,15))
  

############################
#Plot Left WBA
ax3.clear()
#ax3.set_facecolor("w")
ax3.plot(time, Lamp_filt, 'g')
#ax3.axvline(x = time[frameStartInds5[0][0]], color = 'w',  lw=1)
#ax3.axvline(x = timeCam[int(t*22) + frameStartInds3[0][0]], color = 'w',  lw=1) # line that move 
    
ax3.set_ylabel('L WBA (\N{DEGREE SIGN})')
ax3.spines['left'].set_bounds(90,100)
ax3.set_yticks([90,100])   
ax3.set_ylim((73, 118))
ax3.set_xticks([])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.get_xaxis().set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.set_xlim((46, 120))

ax3.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
ax3.yaxis.label.set_color('k')          #setting up Y-axis label color to blue
ax3.tick_params(axis='x', colors='k')    #setting up X-axis tick color to red
ax3.tick_params(axis='y', colors='k')  #setting up Y-axis tick color to black
ax3.spines['left'].set_color('k')        # setting up Y-axis tick color to red
ax3.spines['bottom'].set_color('k')        # setting up Y-axis tick color to red
ax3.tick_params(axis="y", direction='in') 
ax3.spines['left'].set_position(('axes', -0.02))  
  
ax4.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
ax4.yaxis.label.set_color('k')          #setting up Y-axis label color to blue
ax4.tick_params(axis='x', colors='k')    #setting up X-axis tick color to red
ax4.tick_params(axis='y', colors='k')  #setting up Y-axis tick color to black
ax4.spines['left'].set_color('k')        # setting up Y-axis tick color to red

ax4.tick_params(axis="y", direction='in')
ax4.spines['left'].set_position(('axes', -0.02))  





ax5.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
ax5.yaxis.label.set_color('k')          #setting up Y-axis label color to blue
ax5.tick_params(axis='x', colors='k')    #setting up X-axis tick color to red
ax5.tick_params(axis='y', colors='k')  #setting up Y-axis tick color to black
ax5.spines['left'].set_color('k')        # setting up Y-axis tick color to red
ax5.spines['bottom'].set_color('k')   
ax5.tick_params(axis="y", direction='in')
ax5.tick_params(axis="x", direction='in')
ax5.spines['left'].set_position(('axes', -0.02))   


ax5.spines['bottom'].set_color('k')  
ax5.spines['bottom'].set_position(('axes', -0.02))  
ax5.spines['bottom'].set_bounds([50,60])
ax5.set_xticks([50,60])
ax5.set_xticklabels([0,10])
ax5.set_xlabel('Time (s)')#, position=(1,25))
ax5.xaxis.set_label_coords(0.12,-0.12)


ax3.get_yaxis().set_label_coords(-0.08,0.5)
ax4.get_yaxis().set_label_coords(-0.08,0.55)
ax5.get_yaxis().set_label_coords(-0.08,0.55)




#%% Figure - Pannel F and G



#Params figure
plt.rc('font', size=12)
plt.rc('axes',linewidth=1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['axes.linewidth'] = 1


fig = plt.figure(figsize=(4,9))
fig.subplots_adjust(hspace=.05, wspace=0.05)
gs = fig.add_gridspec(3,14)
colorback = 'gainsboro'
colorback2 = 'lightgrey'
fig.set_facecolor('None')

num_bins = 30

polar_ax = plt.subplot2grid((3,10), (0,0), rowspan=1,colspan=10, projection ='polar')
num_bins2 = 30
Tot_valess = []

count, Val = np.histogram(MeanVal, bins=num_bins2)
count2 = np.append(count, count[0])
ticks = np.arange(0,360, 90)
Val1 = Val[:-1]
Val2 = np.append(Val1, Val1[0])*np.pi/180
polar_ax.plot(Val2, count2, color = '#008000')
polar_ax.grid(False)
polar_ax.set_yticklabels([])
polar_ax.set_xticklabels([])

#Stalk
ax0 = plt.subplot2grid((3,10), (2,0), rowspan=1,colspan=10)
ax0.set_facecolor('None')
Nindiv = []

ax0.plot(x_d_20000,Mean_logprob_Stalk, '#FF0000', label="Stalk")
ax0.set_ylim((-0.1, .8))   
ax0.set_ylabel('dF1 Normal density')
ax0.set_xlabel('Z-score')
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(True)
ax0.yaxis.set_ticks_position('left')
ax0.get_xaxis().set_visible(True)
ax0.tick_params(axis="y", direction='in')
ax0.tick_params(axis="x", direction='in')
ax0.set_yticks([0,.5])
ax0.spines['left'].set_bounds(0,.5)
ax0.spines['left'].set_position(('axes',0))
ax0.spines['bottom'].set_position(('axes', -0.04))
ax0.spines['bottom'].set_bounds(-4, 4)
ax0.set_xticks([-4,0,4])
ax0.get_xaxis().set_label_coords(0.5,-0.18)

#Base
ax1 = plt.subplot2grid((3,10), (1,0), rowspan=1,colspan=10, sharex=ax0)
ax1.set_facecolor('None')
ax1.plot(x_d_20000,Mean_logprob_Base, '#0000FF', label="Stalk")
ax1.set_ylim((-0.1, 0.8))    
ax1.set_ylabel('dF2 Normal density')
ax1.set_xlabel('Z-score')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.get_xaxis().set_visible(False)
ax1.tick_params(axis="y", direction='in')

ax1.set_yticks([0,.5])
ax1.spines['left'].set_bounds(0,.5)
ax1.spines['left'].set_position(('axes', 0))




#%% Open data
#%%Fig2  #???????
#%%Fig2  #???????
#%% Open data
my_path = flydir + '/Data/Dorsal/Pa1/GitHub'
my_path = flydir + '/Fig1to4'

all_data = load_object(my_path + '/' + 'all_data.pickle')

diff_means  = all_data[:][0]
diff_errors = all_data[:][1]
diff_conf = all_data[:][2]
plot_time= all_data[:][3]
Base_Trig_means= all_data[:][4]
Base_Trig_errors= all_data[:][5]
Base_Trig_conf= all_data[:][6]
Stalk_Trig_means= all_data[:][7]
Stalk_Trig_errors= all_data[:][8]
Stalk_Trig_conf= all_data[:][9]
plot_timeCam= all_data[:][10]


all_other = load_object(my_path + '/' + 'all_other.pickle')  

x= all_other[:][0]
diff_mean= all_other[:][1]
diff_error= all_other[:][2]
base_mean= all_other[:][3]
base_error= all_other[:][4]
Stalk_mean= all_other[:][5]
Stalk_error= all_other[:][6]


#%% Figure
line_color = np.linspace(0,1,12)

GreyB1 = '#E2E2E2' #
Grey_dashline = '#969696'
GreyB2 = '#C4C4C4' #dark


plt.rc('font', size=7)
plt.rc('axes', titlesize=7)     # fontsize of the axes title
plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=5)    # fontsize of the tick labels
plt.rc('axes',linewidth=.25)
plt.rcParams['xtick.major.width'] = .25
plt.rcParams['ytick.major.width'] = .25
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2


fig = plt.figure(figsize=(7.25,6.5))
fig.set_facecolor('none')
fig.subplots_adjust(hspace=.05, wspace=0.05)


gs = fig.add_gridspec(3,12+12+5)#12+12+9)


valb = [0, 1, 2, 3] #[np.where(LWBA_sortedT==val[0])[0][0],np.where(LWBA_sortedT==val[1])[0][0],np.where(LWBA_sortedT==val[2])[0][0],np.where(LWBA_sortedT==val[3])[0][0]]

GreyLIM = [1,6]
InsideLim1 = 1+1
InsideLim3 = InsideLim1+3

LenghPlot = [4,12]
val_B = [0,1,2,3,4,5,6,7,8,9,10,11]
val = [2,11,10,1,4,5,0,7,6,3,8,9]

#L - R WBA
dataU = diff_means
dataErr = diff_errors
DATAM = dataU[val[0]]-np.mean(dataU[val[0]][0:1000])
Conf_UP = diff_conf[val[0]][0] -np.mean(dataU[val[0]][0:1000])
Conf_DOWN = diff_conf[val[0]][1] -np.mean(dataU[val[0]][0:1000])



Val_line = 1 
Val_line2 = .5

ColorM_normal = plt.cm.BrBG(line_color[valb])
tint_fact = 0.6
ColorM_lighter = plt.cm.BrBG(line_color[val_B]) + (1 - plt.cm.BrBG(line_color[val_B]))*0 * tint_fact


ax0 = plt.subplot(gs[0,1])
plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)

plt.plot(plot_time, DATAM, color = 'k', linewidth = Val_line)
plt.fill_between(plot_time, Conf_UP, Conf_DOWN, color = 'darkgray', edgecolor = 'none')

plt.ylabel('L-R WBA  (\N{DEGREE SIGN})')
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.get_xaxis().set_visible(False)
ax0.spines['left'].set_bounds(-10, 10)
ax0.set_yticks([-10,0,10])
ax0.set_ylim((-12, 18.5))
ax0.tick_params(axis="y", direction='in')
ax0.spines['left'].set_position(('axes', -0.1))


for i in range(11):
    ax1 = plt.subplot(gs[0,2+i], sharex=ax0, sharey=ax0)
    plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
    plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)

    DATAM = dataU[val[i+1]]-np.mean(dataU[val[i+1]][0:1000])
    Conf_UP = diff_conf[val[i+1]][0] -np.mean(dataU[val[i+1]][0:1000])
    Conf_DOWN = diff_conf[val[i+1]][1] -np.mean(dataU[val[i+1]][0:1000])

    plt.plot(plot_time,DATAM, color ='k', linewidth = Val_line) 
    plt.fill_between(plot_time, Conf_UP, Conf_DOWN, color = 'darkgray', edgecolor = 'none') 

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
  


#Base 
dataU = Base_Trig_means
dataErr = Base_Trig_errors
DATAM = (dataU[val[0]]-np.mean(dataU[val[0]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
Conf_UP = (Base_Trig_conf[val[0]][0] -np.mean(dataU[val[0]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
Conf_DOWN = (Base_Trig_conf[val[0]][1] -np.mean(dataU[val[0]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))

ax1 = plt.subplot(gs[1,1], sharex=ax0)

plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)

plt.plot(plot_timeCam,DATAM, color = 'k', linewidth = Val_line)
plt.fill_between(plot_timeCam, Conf_UP, Conf_DOWN, color = 'darkgray',  edgecolor = 'none')
plt.ylabel(u'ΔF/F (%) dF2 F')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.spines['left'].set_bounds(-10, 10)
ax1.set_yticks([-10,0,10])
ax1.set_ylim((-26, 26.5))
ax1.tick_params(axis="y", direction='in')
ax1.spines['left'].set_position(('axes', -0.1))



for i in range(11):
    ax2 = plt.subplot(gs[1,2+i], sharex=ax0, sharey=ax1)
    plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
    plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)

    DATAM = (dataU[val[i+1]]-np.mean(dataU[val[i+1]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
    Conf_UP = (Base_Trig_conf[val[i+1]][0] -np.mean(dataU[val[i+1]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
    Conf_DOWN = (Base_Trig_conf[val[i+1]][1] -np.mean(dataU[val[i+1]][0:11]) )/np.mean(abs(dataU[val[i+1]][0:11]))
    plt.plot(plot_timeCam,DATAM, color ='k', linewidth = Val_line)
    plt.fill_between(plot_timeCam, Conf_UP, Conf_DOWN, color ='darkgray', edgecolor = 'none')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)


#Stalk 
dataU = Stalk_Trig_means
dataErr = Stalk_Trig_errors
DATAM = (dataU[val[0]]-np.mean(dataU[val[0]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
Conf_UP = (Stalk_Trig_conf[val[0]][0] -np.mean(dataU[val[0]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
Conf_DOWN = (Stalk_Trig_conf[val[0]][1] -np.mean(dataU[val[0]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))

ax1 = plt.subplot(gs[2,1], sharex=ax0)
plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)


plt.plot(plot_timeCam,DATAM, color ='k', linewidth = Val_line)
plt.fill_between(plot_timeCam, Conf_UP, Conf_DOWN, color ='darkgray', edgecolor = 'none')
plt.ylabel(u'ΔF/F (%) dF1 F')
plt.xlabel('Time (s)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_bounds(-10, 10)
ax1.set_yticks([-10,0,10])
ax1.set_ylim((-50, 36))
ax1.spines['bottom'].set_bounds(2, 5)
ax1.set_xticks([2,5])
ax1.set_xticklabels([0,3])
ax1.tick_params(axis="y", direction='in')
ax1.spines['left'].set_position(('axes', -0.1))

ax1.spines['bottom'].set_position(('axes', -0.06))
ax1.tick_params(axis="x", direction='in')



for i in range(11):
    ax2 = plt.subplot(gs[2,2+i], sharex=ax0, sharey=ax1)
    plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
    plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)

    DATAM = (dataU[val[i+1]]-np.mean(dataU[val[i+1]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
    Conf_UP = (Stalk_Trig_conf[val[i+1]][0] -np.mean(dataU[val[i+1]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
    Conf_DOWN = (Stalk_Trig_conf[val[i+1]][1] -np.mean(dataU[val[i+1]][0:11]))/np.mean(abs(dataU[val[i+1]][0:11]))
    plt.plot(plot_timeCam,DATAM, color ='k', linewidth = Val_line)
    plt.fill_between(plot_timeCam, Conf_UP, Conf_DOWN, color ='darkgray' , edgecolor = 'none')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)



#######################################################

# Now, yaw-roll tuning curves
ax1b = fig.add_subplot(gs[0,14:12+6])
ax2b = fig.add_subplot(gs[1,14:12+6])
ax3b = fig.add_subplot(gs[2,14:12+6])

ax1b.set_facecolor('none') #GreyB2
ax2b.set_facecolor('none')
ax3b.set_facecolor('none')

val_Err = 1 
val_s = 30 

ax1b.errorbar(x,diff_mean-np.mean(diff_mean), yerr=[diff_mean-diff_error[:,0], diff_error[:,1]-diff_mean], fmt='o', markersize = 0, mfc='k', ecolor = 'k', elinewidth = val_Err) 
ax1b.scatter(x,diff_mean-np.mean(diff_mean), color = 'k', s = val_s,  zorder=10) 

ax1b.spines['bottom'].set_bounds(0, 360)
ax1b.set_xlim((-10,370))
ax1b.set_xticks([0, 90, 180, 270, 360])

ax1b.set_ylim((-.25,.3))
ax1b.set_yticks([-0.2,  0, 0.2])
ax1b.spines['left'].set_bounds(-.2,.2)

ax1b.tick_params(axis='both', which='major', pad=2)
ax1b.spines['top'].set_visible(False)
ax1b.spines['right'].set_visible(False)
ax1b.spines['bottom'].set_visible(False)

ax1b.yaxis.set_ticks_position('left')

ax1b.set_yticks([-0.5*max(abs(diff_mean)), 0,  0.5*max(abs(diff_mean))])
ax1b.spines['left'].set_bounds(-0.5*max(abs(diff_mean)), 0.5*max(abs(diff_mean)))

labelsax1b = ax1b.get_yticks()
llabelsax1bm = np.round(labelsax1b/max(abs(diff_mean)),decimals=1)
ax1b.set_yticklabels(llabelsax1bm)


#Base
ax2b.errorbar(x,base_mean-np.mean(base_mean), yerr=[base_mean-base_error[:,0], base_error[:,1]-base_mean], fmt='o',  markersize = 0,mfc='k', ecolor = 'k', elinewidth = val_Err)
ax2b.scatter(x,base_mean-np.mean(base_mean), color ='k',  s = val_s,  zorder=10)
ax2b.spines['bottom'].set_bounds(0, 360)
ax2b.set_xlim((-10,370))
ax2b.set_xticks([0, 90, 180, 270, 360])
ax2b.set_ylim((-.1, .1))
ax2b.set_yticks([-.05, 0,  0.05])
ax2b.spines['left'].set_bounds(-0.05, 0.05)


ax2b.tick_params(axis='both', which='major', pad=2)

ax2b.spines['top'].set_visible(False)
ax2b.spines['right'].set_visible(False)
ax2b.spines['bottom'].set_visible(False)
ax2b.set_xticks([])
ax2b.yaxis.set_ticks_position('left')


ax2b.set_yticks([-0.5*max(abs(base_mean)), 0,  0.5*max(abs(base_mean))])
ax2b.spines['left'].set_bounds(-0.5*max(abs(base_mean)), 0.5*max(abs(base_mean)))


labelsax2b = ax2b.get_yticks()
llabelsax2bm = np.round(labelsax2b/max(abs(base_mean)),decimals=1) #labelsax2b/max(abs(base_mean))
ax2b.set_yticklabels(llabelsax2bm)



#Stalk
ax3b.errorbar(x,Stalk_mean-np.mean(Stalk_mean), yerr=[Stalk_mean-Stalk_error[:,0], Stalk_error[:,1]-Stalk_mean], fmt='o', markersize = 0, mfc='k', ecolor = 'k', elinewidth = val_Err)
ax3b.scatter(x,Stalk_mean-np.mean(Stalk_mean), color ='k',  s = val_s,  zorder=10)
ax3b.spines['bottom'].set_bounds(0, 360)
ax3b.set_xlim((-10,370))
ax3b.set_xticks([0, 90, 180, 270, 360])
ax3b.set_ylim((-.09, .1))
ax3b.set_yticks([-.05, 0,  0.05])
ax3b.spines['left'].set_bounds(-0.05, 0.05)

ax3b.tick_params(axis='both', which='major', pad=2)

ax3b.spines['top'].set_visible(False)
ax3b.spines['right'].set_visible(False)
ax3b.spines['bottom'].set_visible(True)


ax1b.set_xticks([])

ax3b.xaxis.set_ticks_position('bottom')
ax3b.yaxis.set_ticks_position('left')

ax3b.set_yticks([-0.5*max(abs(Stalk_mean)), 0,  0.5*max(abs(Stalk_mean))])
ax3b.spines['left'].set_bounds(-0.5*max(abs(Stalk_mean)), 0.5*max(abs(Stalk_mean)))


labels3b2 = ax3b.get_yticks()
labels3b2m = np.round(labels3b2/max(abs(Stalk_mean)),decimals=1) # labels3b2/max(abs(Stalk_mean))
ax3b.set_yticklabels(labels3b2m)

labels = [item.get_text() for item in ax3b.get_xticklabels()]
labels[0] = 'RR'
labels[1] = 'YR'
labels[2] = 'RL'
labels[3] = 'YL'
labels[4] = 'RR'
ax3b.set_xticklabels(labels)

ax1b.tick_params(axis="y", direction='in')
ax2b.tick_params(axis="y", direction='in')
ax3b.tick_params(axis="y", direction='in')
ax3b.tick_params(axis="x", direction='in')

ax1b.spines['left'].set_position(('axes', -0.06))
ax2b.spines['left'].set_position(('axes', -0.06))
ax3b.spines['left'].set_position(('axes', -0.06))

ax3b.spines['bottom'].set_position(('axes', -0.06))    

############################################################

# Now, yaw-roll tuning curves

ax4 = fig.add_subplot(gs[0,12+7:12+12], projection = 'polar')#13+8:12+12+9], projection = 'polar')
ax5b =  fig.add_subplot(gs[1,12+7:12+12], projection = 'polar')
ax6b = fig.add_subplot(gs[2,12+7:12+12], projection = 'polar')

#L-R WBA
ax4.plot(np.deg2rad(x),stats.zscore(diff_mean), color = 'k', linewidth = 1)
ax4.set_yticklabels([])
ticks = np.arange(0,360, 90)
ax4.set_theta_zero_location("W")
ax4.grid(False)
ax4.set_thetagrids(ticks,('RR', 'YR', 'RL','YL'))


#Base
ax5b.plot(np.deg2rad(x),stats.zscore(base_mean), color = 'k', linewidth = 1)
ax5b.set_yticklabels([])
ticks = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
ax5b.set_xticks([])
ax5b.set_theta_zero_location("W")
ax5b.grid(False)

#Stalk
ax6b.plot(np.deg2rad(x),stats.zscore(Stalk_mean), color = 'k', linewidth = 1)
ax6b.set_yticklabels([])
ticks = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
ax6b.set_xticks([])
ax6b.set_theta_zero_location("W")
ax6b.grid(False)



#%% Figure 3   #???????
#%% open data 

raw_all = load_object(my_path + '/' + 'raw_all_Pa1.pickle')

plot_time_frames2 = raw_all[:][0]
plot_time2 = raw_all[:][1]
y_L  = raw_all[:][2]
y_Base = raw_all[:][3]
y_Stalk = raw_all[:][4]
wing_avgs = raw_all[:][5]
Basemusc_avgs = raw_all[:][6]
Stalkmusc_avgs = raw_all[:][7]



#%% Figure

val1 = 939
val2 = 1265


valP1 = 9
valP2 = 12 #14

PPat = 2000
PPat2 = 4000
PPat3 = 10000
PPat4 = 12000


PPatS = 22
PPatS2 = 44
PPatS3 = 110
PPatS4 = 132



Tint = valP1
TstartSacc = val1


Grey_dashline = '#969696'
GreyB2 = '#C4C4C4' #dark
GreyB1 = GreyB2 # '#E2E2E2' #

plt.rc('font', size=7)
plt.rc('axes', titlesize=7)     # fontsize of the axes title
plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=5)    # fontsize of the tick labels
plt.rc('axes',linewidth=.25)
plt.rcParams['xtick.major.width'] = .25
plt.rcParams['ytick.major.width'] = .25
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

Val_line = 1 
Val_line2 = .5
val_Err = 1 
val_s = 30 


fig = plt.figure(figsize=(4,6.5))#
fig.set_facecolor('none')
line_color = np.linspace(0,1,len(wing_avgs))
fig.subplots_adjust(hspace=.05, wspace=0.05)
gs = fig.add_gridspec(3,13)

#L-R

ax1 = fig.add_subplot(gs[0,0:2])
ax1.set_facecolor('none')
ax1.set_yticks([])
Inter = round(((1000*5)/11))
im = ax1.imshow(y_L, cmap="BrBG", aspect="auto", vmin=-15, vmax=15 ,  extent=[np.min(plot_time2),np.max(plot_time2), 0, len(y_L)], origin='lower')#, origin='lower', extent=[plot_time[0],plot_time[-1],0,len(y_Stalk)]) #BrBG 
ax1.margins(x=1, y=0.05)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel(' Angle (\N{DEGREE SIGN})')

ax1.spines['bottom'].set_visible(False)
ax1.get_xaxis().set_visible(False)

plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)


#Base

#estimates vmin and max 
valvmin2 =  -25 
valvmax2 = 25

ax2 = fig.add_subplot(gs[1,0:2])
ax2.set_facecolor('none')
ax2.set_yticks([])
im1 = ax2.imshow(y_Base,  cmap="BrBG", aspect="auto", vmin=valvmin2, vmax=valvmax2 , extent=[np.min(plot_time_frames2),np.max(plot_time_frames2), 0, len(y_Base)], origin='lower')#, origin='lower', extent=[plot_timeCam[0],plot_timeCam[-1],0,len(y_Stalk)]) #plasma #BrBG, vmin = -4, vmax = 4
ax2.set_ylabel(u'ΔF/F (%)')#Base
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.get_xaxis().set_visible(False)
plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)


#Stalk

#estimates vmin and max 
valvmin =  -25
valvmax = 25

ax3 = fig.add_subplot(gs[2,0:2])
ax3.set_facecolor('none')
ax3.set_yticks([])
im2 = ax3.imshow(y_Stalk, cmap="BrBG", aspect="auto",  vmin=valvmin, vmax=valvmax , extent=[np.min(plot_time_frames2),np.max(plot_time_frames2), 0, len(y_Stalk)], origin='lower')#, origin='lower'  , extent=[plot_timeCam[0],plot_timeCam[-1],0,len(y_Stalk)], vmin = -6, vmax = 6) #plasma
ax3.set_ylabel(u'ΔF/F (%)')#Stalk
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(True)
ax3.get_xaxis().set_visible(True)
ax3.spines['bottom'].set_position(('axes', -0.06))
ax3.set_xlabel('Time (s)')
plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)



ax3.spines['bottom'].set_bounds(2, 5)
ax3.set_xticks([2,5])
ax3.set_xticklabels([0,3])

ax3.get_xaxis().set_label_coords(0.55,-0.16)

ax3.tick_params(axis="x", direction='in')


################################

#WBA
ax2b = fig.add_subplot(gs[0,3:8])
for i in range(len(wing_avgs)):    
    ax2b.plot(plot_time2, wing_avgs[i] - np.mean(wing_avgs[i][0:1000],axis=0),color=plt.cm.BrBG(line_color[i]), zorder=10, linewidth = Val_line)
 
ax2b.yaxis.set_ticks_position('left')                                  
ax2b.spines['bottom'].set_visible(False)
ax2b.spines['top'].set_visible(False)
ax2b.spines['right'].set_visible(False)   
ax2b.get_xaxis().set_visible(False)

ax2b.spines['left'].set_bounds(-10, 10)
ax2b.set_yticks([-10,0,10])
ax2b.spines['left'].set_position(('axes', -0.06))
ax2b.set_facecolor(GreyB1)

plt.axvspan(GreyLIM[0], GreyLIM[1], facecolor = GreyB2, edgecolor = 'none')
plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)

ax2b.tick_params(axis="y", direction='in')


#Base
ax7 =  fig.add_subplot(gs[1,3:8], sharex=ax2b)
line_color = np.linspace(0,1,len(Basemusc_avgs))
for i in range(len(Basemusc_avgs)):
    ax7.plot(plot_time_frames2, (Basemusc_avgs[i] - np.mean(Basemusc_avgs[i][0:valP1]))/np.mean(Basemusc_avgs[i][0:valP1])*100,color=plt.cm.BrBG(line_color[i]), zorder=10, linewidth = Val_line) #/np.nanmean(Basemusc_avgs[i][0:11])

ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

ax7.spines['left'].set_bounds(-10, 10)
ax7.set_yticks([-10,0,10])
ax7.spines['left'].set_position(('axes', -0.06))
ax7.set_facecolor(GreyB1)
ax7.spines['bottom'].set_visible(False)
ax7.get_xaxis().set_visible(False)

ax7.tick_params(axis="y", direction='in')



plt.axvspan(GreyLIM[0], GreyLIM[1], facecolor = GreyB2, edgecolor = 'none')
plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)




#Stalk
ax1b = fig.add_subplot(gs[2,3:8], sharex=ax2b)
line_color = np.linspace(0,1,len(Stalkmusc_avgs))
for i in range(len(Stalkmusc_avgs)):
    ax1b.plot(plot_time_frames2, (Stalkmusc_avgs[i] - np.mean(Stalkmusc_avgs[i][0:valP1]))/np.mean(Stalkmusc_avgs[i][0:valP1])*100, color=plt.cm.BrBG(line_color[i]), zorder=10, linewidth = Val_line) #/np.nanmean(Stalkmusc_avgs[i][0:22])
ax1b.spines['top'].set_visible(False)
ax1b.spines['right'].set_visible(False)
ax1b.spines['bottom'].set_visible(True)
ax1b.get_xaxis().set_visible(True)

ax1b.spines['left'].set_bounds(-10, 10)
ax1b.set_yticks([-10,0,10])
ax1b.spines['left'].set_position(('axes', -0.06))
ax1b.set_facecolor(GreyB1)

plt.axvspan(GreyLIM[0], GreyLIM[1], facecolor = GreyB2, edgecolor = 'none')
plt.axvline(x = InsideLim1, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)
plt.axvline(x = InsideLim3, color = Grey_dashline, linestyle='dashed', linewidth = Val_line2)

ax1b.spines['bottom'].set_position(('axes', -0.06))


ax1b.spines['bottom'].set_bounds(2, 5)
ax1b.set_xticks([2,5])
ax1b.set_xticklabels([0,3])


ax1b.set_xlabel('Time (s)')

ax1b.tick_params(axis="y", direction='in')
ax1b.tick_params(axis="x", direction='in')
ax1b.get_xaxis().set_label_coords(0.54,-0.16)

msums = []
msums2 = []
wsums = []
for i in range(len(Stalkmusc_avgs)):
    Nstalk = Stalkmusc_avgs[i] - np.nanmean(Stalkmusc_avgs[i][0:PPatS2])
    musc_sum = np.mean(Nstalk[PPatS2:PPatS3])  
    msums.append(musc_sum)
    NBase = Basemusc_avgs[i] - np.nanmean(Basemusc_avgs[i][0:PPatS2])
    musc2_sum = np.mean(NBase[PPatS2:PPatS3])
    msums2.append(musc2_sum)
    wing_sum = np.mean(wing_avgs[i][PPat2:PPat3]- np.mean(wing_avgs[i][0:PPat],axis=0))
    wsums.append(wing_sum)    



Num_scat = list(range(len(msums)))# [1,2,3,4,5,6,7,8,9,10]

#WBA
ax4 = fig.add_subplot(gs[0,9:14])
ax4.scatter(Num_scat, stats.zscore(wsums[0:10],ddof=1),  color=plt.cm.BrBG(line_color[0:10]), zorder=10, s = val_s)

ax4.axvline(x=4.5, ymin = 0.01, ymax=0.99, color= Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax4.axhline(y=0, xmin=0.01, xmax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax4.spines['left'].set_bounds(0, 1)
ax4.set_yticks([0,1])
ax4.spines['top'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(True)
ax4.get_xaxis().set_visible(False)
ax4.set_xticklabels([])
ax4.set_xticks(Num_scat)
ax4.spines['bottom'].set_bounds(0, 9)
ax4.spines['left'].set_position(('axes', -0.06))
ax4.spines['bottom'].set_position(('axes', -0.06))
ax4.set_facecolor(GreyB2)

ax4.tick_params(axis="y", direction='in')

#Base
ax6 = fig.add_subplot(gs[1,9:14])
ax6.scatter(Num_scat, stats.zscore(msums2[0:10],ddof=1),  color=plt.cm.BrBG(line_color[0:10]), zorder=10, s = val_s)

ax6.axvline(x=4.5, ymin = 0.01, ymax=0.99, color= Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax6.axhline(y=0, xmin=0.01, xmax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax6.spines['left'].set_bounds(0, 1)
ax6.set_yticks([0,1])
ax6.spines['top'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(True)
ax6.spines['left'].set_position(('axes', -0.06))
ax6.spines['bottom'].set_position(('axes', -0.06))
ax6.set_facecolor(GreyB2)

ax6.get_xaxis().set_visible(False)
ax6.tick_params(axis="y", direction='in')



#Stalk
ax3b = fig.add_subplot(gs[2,9:14])
ax3b.scatter(Num_scat,stats.zscore(msums[0:10],ddof=1),  color=plt.cm.BrBG(line_color[0:10]), zorder=10, s = val_s)

ax3b.axvline(x=4.5, ymin = 0.01, ymax=0.99, color= Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax3b.axhline(y=0, xmin=0.01, xmax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax3b.spines['left'].set_bounds(0, 1)
ax3b.set_yticks([0,1])
ax3b.spines['top'].set_visible(False)
ax3b.spines['bottom'].set_visible(True)
ax3b.spines['right'].set_visible(False)
ax3b.spines['left'].set_visible(True)
ax3b.get_xaxis().set_visible(True)

ax3b.set_xticks(Num_scat)
ax3b.spines['bottom'].set_bounds(0, 9)
ax3b.spines['left'].set_position(('axes', -0.06))
ax3b.spines['bottom'].set_position(('axes', -0.06))
ax3b.set_facecolor(GreyB2)

ax3b.set_xlabel('Amplitude decile')
ax3b.set_xticklabels([])
ax3b.set_xticks(Num_scat)
ax3b.spines['bottom'].set_bounds(0, 9)

ax3b.tick_params(axis="y", direction='in')
ax3b.tick_params(axis="x", direction='in')



ax2.get_xaxis().set_label_coords(0.52,-0.18)
ax7.get_xaxis().set_label_coords(0.5,-0.18)
ax6.get_xaxis().set_label_coords(0.5,-0.18)





#%% Figure 4  #???????
#%% open data 

raw_all = load_object(my_path + '/' + 'raw_all.pickle')
avgs_all = load_object(my_path + '/' + 'avgs_all.pickle')

y_L  = raw_all[:][0]
y_Base = raw_all[:][1]
y_Stalk = raw_all[:][2]

plot_time2 = avgs_all[:][0]
wing_avgs= avgs_all[:][1]
plot_time_frames2= avgs_all[:][2]
Basemusc_avgs= avgs_all[:][3]
Stalkmusc_avgs= avgs_all[:][4]

#%% Figure
 
val1 = 939
val2 = 1265


valP1 = 9
valP2 = 12 #14


Tint = valP1
TstartSacc = val1
 
 
 
GreyB1 = '#E2E2E2' #
Grey_dashline = '#969696'
GreyB2 = '#C4C4C4' #dark


plt.rc('font', size=7)
plt.rc('axes', titlesize=7)     # fontsize of the axes title
plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=5)    # fontsize of the tick labels
plt.rc('axes',linewidth=.25)
plt.rcParams['xtick.major.width'] = .25
plt.rcParams['ytick.major.width'] = .25
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

Val_line = 1 
Val_line2 = .5
val_Err = 1 
val_s = 30 


#for powerpoint
#fig = plt.figure(figsize=(12,6))

fig = plt.figure(figsize=(6.5,4))
fig.set_facecolor('none')
line_color = np.linspace(0,1,len(wing_avgs))
fig.subplots_adjust(hspace=.05, wspace=0.05)

gs = fig.add_gridspec(3,13)


#L-R
ax1 = fig.add_subplot(gs[0,0:2])
ax1.set_facecolor('none')
ax1.set_yticks([])
Inter = round(((1000*5)/11))
im = ax1.imshow(y_L, cmap="BrBG", aspect="auto", vmin=-15, vmax=15 ,  extent=[np.min(plot_time2),np.max(plot_time2), 0, len(y_L)], origin='lower')#, origin='lower', extent=[plot_time[0],plot_time[-1],0,len(y_Stalk)]) #BrBG 
ax1.margins(x=1, y=0.05)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (\N{DEGREE SIGN})')

ax1.spines['bottom'].set_visible(False)
ax1.get_xaxis().set_visible(False)


#Base

#estimates vmin and max 
valvmin2 =  -11
valvmax2 = 11



ax2 = fig.add_subplot(gs[1,0:2])
ax2.set_facecolor('none')
ax2.set_yticks([])
im1 = ax2.imshow(y_Base,  cmap="BrBG", aspect="auto", vmin=valvmin2, vmax=valvmax2 , extent=[np.min(plot_time_frames2),np.max(plot_time_frames2), 0, len(y_Base)], origin='lower')#, origin='lower', extent=[plot_timeCam[0],plot_timeCam[-1],0,len(y_Stalk)]) #plasma #BrBG, vmin = -4, vmax = 4
ax2.set_ylabel(u'ΔF/F (%)')#Base
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.get_xaxis().set_visible(False)

#Stalk

#estimates vmin and max 
valvmin =  -7
valvmax = 7

ax3 = fig.add_subplot(gs[2,0:2])
ax3.set_facecolor('none')
ax3.set_yticks([])
im2 = ax3.imshow(y_Stalk, cmap="BrBG", aspect="auto", vmin=valvmin, vmax=valvmax , extent=[np.min(plot_time_frames2),np.max(plot_time_frames2), 0, len(y_Stalk)], origin='lower')#, origin='lower'  , extent=[plot_timeCam[0],plot_timeCam[-1],0,len(y_Stalk)], vmin = -6, vmax = 6) #plasma
ax3.set_ylabel(u'ΔF/F (%)')#Stalk
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(True)
ax3.tick_params(axis="x", direction='in')

ax3.get_xaxis().set_visible(True)
ax3.spines['bottom'].set_position(('axes', -0.06))
ax3.set_xticks([-.2,0,.2])
ax3.spines['bottom'].set_bounds(-.2, .2)
ax3.set_xlabel('Time (s)')





ax1.axvline(x=plot_time2[TstartSacc], color= Grey_dashline, linewidth = Val_line2)
ax2.axvline(x=plot_time_frames2[Tint], color= Grey_dashline, linewidth = Val_line2)
ax3.axvline(x=plot_time_frames2[Tint], color= Grey_dashline, linewidth = Val_line2)




##############################
#WBA
ax2b = fig.add_subplot(gs[0,3:8])
ax2b.axvspan(plot_time2[val1],plot_time2[val2],  facecolor = GreyB2, edgecolor = 'none')
for i in range(len(wing_avgs)):    
    ax2b.plot(plot_time2, wing_avgs[i] - np.mean(wing_avgs[i][0:1000],axis=0),color=plt.cm.BrBG(line_color[i]), zorder=10, linewidth = Val_line)
 
ax2b.spines['bottom'].set_bounds(0.25, .75)
ax2b.set_xlim((-0.45, 0.45))
ax2b.yaxis.set_ticks_position('left')                                  
ax2b.spines['bottom'].set_visible(False)
ax2b.spines['top'].set_visible(False)
ax2b.spines['right'].set_visible(False)   
ax2b.get_xaxis().set_visible(False)

ax2b.spines['left'].set_bounds(-10, 10)
ax2b.set_yticks([-10,0,10])
ax2b.spines['left'].set_position(('axes', -0.03))
ax2b.set_facecolor(GreyB1)
ax2b.tick_params(axis="y", direction='in')




#Base
ax7 =  fig.add_subplot(gs[1,3:8], sharex=ax2b)
line_color = np.linspace(0,1,len(Basemusc_avgs))
ax7.axvspan(plot_time_frames2[valP1], plot_time_frames2[valP2], facecolor = GreyB2, edgecolor = 'none')
for i in range(len(Basemusc_avgs)):
    ax7.plot(plot_time_frames2, (Basemusc_avgs[i] - np.mean(Basemusc_avgs[i][0:valP1]))/np.mean(Basemusc_avgs[i][0:valP1])*100,color=plt.cm.BrBG(line_color[i]), zorder=10, linewidth = Val_line) #/np.nanmean(Basemusc_avgs[i][0:11])

#ax7.set_ylabel('Base Zscore')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['bottom'].set_visible(False)
ax7.get_xaxis().set_visible(False)
ax7.spines['left'].set_bounds(0, 2)
ax7.set_yticks([0,2])
ax7.spines['left'].set_position(('axes', -0.03))
ax7.set_facecolor(GreyB1)


ax7.tick_params(axis="y", direction='in')




#Stalk
ax1b = fig.add_subplot(gs[2,3:8], sharex=ax2b)
line_color = np.linspace(0,1,len(Stalkmusc_avgs))
ax1b.axvspan(plot_time_frames2[valP1], plot_time_frames2[valP2], facecolor = GreyB2, edgecolor = 'none')
for i in range(len(Stalkmusc_avgs)):
    ax1b.plot(plot_time_frames2, (Stalkmusc_avgs[i] - np.mean(Stalkmusc_avgs[i][0:valP1]))/np.mean(Stalkmusc_avgs[i][0:valP1])*100, color=plt.cm.BrBG(line_color[i]), zorder=10, linewidth = Val_line) #/np.nanmean(Stalkmusc_avgs[i][0:22])
ax1b.spines['top'].set_visible(False)
ax1b.spines['right'].set_visible(False)

ax1b.spines['bottom'].set_visible(True)
ax1b.get_xaxis().set_visible(True)

ax1b.spines['left'].set_bounds(0, 2)
ax1b.set_yticks([0,2])
ax1b.spines['left'].set_position(('axes', -0.03))
ax1b.set_facecolor(GreyB1)


ax1b.spines['bottom'].set_position(('axes', -0.06))
ax1b.spines['bottom'].set_bounds(-0.1, 0.1)
ax1b.set_xticks([-0.1, 0 ,0.1])
ax1b.set_xlabel('Time (s)')

ax1b.tick_params(axis="y", direction='in')
ax1b.tick_params(axis="x", direction='in')


ax2b.axvline(x=plot_time2[TstartSacc],color = Grey_dashline,  linewidth = Val_line2)
ax1b.axvline(x=plot_time_frames2[Tint], color = Grey_dashline,  linewidth = Val_line2)
ax7.axvline(x=plot_time_frames2[Tint], color = Grey_dashline,  linewidth = Val_line2)





msums = []
msums2 = []
wsums = []
for i in range(len(Stalkmusc_avgs)):
    Nstalk = Stalkmusc_avgs[i] - np.nanmean(Stalkmusc_avgs[i][0:valP1])
    musc_sum = np.mean(Nstalk[valP1:valP2])
    msums.append(musc_sum)
    NBase = Basemusc_avgs[i] - np.nanmean(Basemusc_avgs[i][0:valP1])
    musc2_sum = np.mean(NBase[valP1:valP2])
    msums2.append(musc2_sum)
    wing_sum = np.mean(wing_avgs[i][val1:val2]- np.mean(wing_avgs[i][0:1000],axis=0))
    wsums.append(wing_sum)    


Num_scat = list(range(len(msums)))# [1,2,3,4,5,6,7,8,9,10]



#WBA
ax4 = fig.add_subplot(gs[0,9:14])
ax4.scatter(Num_scat, stats.zscore(wsums[0:10],ddof=1), s = val_s, color=plt.cm.BrBG(line_color[0:10]), zorder=10)

ax4.axvline(x=4.5, ymin = 0.01, ymax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax4.axhline(y=0, xmin=0.01, xmax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax4.spines['left'].set_bounds(0, 1)
ax4.set_yticks([0,1])
ax4.spines['top'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(True)
ax4.get_xaxis().set_visible(False)
ax4.set_xticklabels([])
ax4.set_xticks(Num_scat)
ax4.spines['bottom'].set_bounds(0, 9)
ax4.spines['left'].set_position(('axes', -0.04))
ax4.spines['bottom'].set_position(('axes', -0.04))
ax4.set_facecolor(GreyB2)

ax4.tick_params(axis="y", direction='in')

#Stalk
ax3b = fig.add_subplot(gs[2,9:14])
ax3b.scatter(Num_scat,stats.zscore(msums[0:10],ddof=1), s = val_s, color=plt.cm.BrBG(line_color[0:10]), zorder=10)

ax3b.axvline(x=4.5, ymin = 0.01, ymax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax3b.axhline(y=0, xmin=0.01, xmax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax3b.spines['left'].set_bounds(0, 1)
ax3b.set_yticks([0,1])
ax3b.spines['top'].set_visible(False)
ax3b.spines['bottom'].set_visible(True)
ax3b.spines['right'].set_visible(False)
ax3b.spines['left'].set_visible(True)
ax3b.get_xaxis().set_visible(True)
ax3b.set_xticklabels([])
ax3b.set_xticks(Num_scat)
ax3b.spines['bottom'].set_bounds(0, 9)
ax3b.spines['left'].set_position(('axes', -0.04))
ax3b.spines['bottom'].set_position(('axes', -0.04))
ax3b.set_facecolor(GreyB2)
ax3b.tick_params(axis="y", direction='in')
ax3b.tick_params(axis="x", direction='in')
ax3b.set_xlabel('Amplitude decile')


#Base
ax6 = fig.add_subplot(gs[1,9:14])
ax6.scatter(Num_scat, stats.zscore(msums2[0:10],ddof=1), s = val_s, color=plt.cm.BrBG(line_color[0:10]), zorder=10)

ax6.axvline(x=4.5, ymin = 0.01, ymax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax6.axhline(y=0, xmin=0.01, xmax=0.99, color=Grey_dashline, linestyle='dashed', linewidth = Val_line2)
ax6.spines['left'].set_bounds(0, 1)
ax6.set_yticks([0,1])
ax6.spines['top'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(True)
ax6.get_xaxis().set_visible(False)
ax6.set_xlabel('Amplitude decile')
ax6.set_xticklabels([])
ax6.set_xticks(Num_scat)
ax6.spines['bottom'].set_bounds(0, 9)
ax6.spines['left'].set_position(('axes', -0.04))
ax6.spines['bottom'].set_position(('axes', -0.04))
ax6.set_facecolor(GreyB2)
ax6.tick_params(axis="y", direction='in')




ax2.get_xaxis().set_label_coords(0.52,-0.18)
ax7.get_xaxis().set_label_coords(0.5,-0.18)
ax6.get_xaxis().set_label_coords(0.5,-0.18)






