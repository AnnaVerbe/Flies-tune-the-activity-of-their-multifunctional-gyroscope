# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:47:45 2017

@author: bradleydickerson

trying to find different deciles of haltere muscle activity and resulting L-R WBA during spontaneous saccades
"""
#load the data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


fly1_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_pos_sacc.csv', delimiter = ',')
fly1_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_b1.csv').as_matrix()
fly1_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_b2.csv').as_matrix()
fly1_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_iii1.csv').as_matrix()
fly1_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_iii2.csv').as_matrix()
fly1_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_i1.csv').as_matrix()


fly2_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_pos_sacc.csv', delimiter = ',')
fly2_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_b1.csv').as_matrix()
fly2_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_b2.csv').as_matrix()
fly2_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_iii1.csv').as_matrix()
fly2_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_iii2.csv').as_matrix()
fly2_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_i1.csv').as_matrix()

fly3_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_pos_sacc.csv', delimiter = ',')
fly3_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_b1.csv').as_matrix()
fly3_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_b2.csv').as_matrix()
fly3_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_iii1.csv').as_matrix()
fly3_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_iii2.csv').as_matrix()
fly3_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_i1.csv').as_matrix()

fly4_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_pos_sacc.csv', delimiter = ',')
fly4_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_b1.csv').as_matrix()
fly4_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_b2.csv').as_matrix()
fly4_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_iii1.csv').as_matrix()
fly4_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_iii2.csv').as_matrix()
fly4_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_i1.csv').as_matrix()

fly5_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_pos_sacc.csv', delimiter = ',')
fly5_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_b1.csv').as_matrix()
fly5_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_b2.csv').as_matrix()
fly5_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_iii1.csv').as_matrix()
fly5_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_iii2.csv').as_matrix()
fly5_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_i1.csv').as_matrix()

fly6_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_pos_sacc.csv', delimiter = ',')
fly6_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_b1.csv').as_matrix()
fly6_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_b2.csv').as_matrix()
fly6_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_iii1.csv').as_matrix()
fly6_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_iii2.csv').as_matrix()
fly6_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_i1.csv').as_matrix()

fly7_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_pos_sacc.csv', delimiter = ',')
fly7_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_b1.csv').as_matrix()
fly7_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_b2.csv').as_matrix()
fly7_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_iii1.csv').as_matrix()
fly7_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_iii2.csv').as_matrix()
fly7_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_i1.csv').as_matrix()

fly8_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_neg_sacc.csv', delimiter = ',')
fly8_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_b1neg.csv').as_matrix()
fly8_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_b2neg.csv').as_matrix()
fly8_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_iii1neg.csv').as_matrix()
fly8_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_iii2neg.csv').as_matrix()
fly8_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 2/triggered_fluor_i1neg.csv').as_matrix()


fly9_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_neg_sacc.csv', delimiter = ',')
fly9_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_b1neg.csv').as_matrix()
fly9_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_b2neg.csv').as_matrix()
fly9_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_iii1neg.csv').as_matrix()
fly9_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_iii2neg.csv').as_matrix()
fly9_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 4/triggered_fluor_i1neg.csv').as_matrix()

fly10_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_neg_sacc.csv', delimiter = ',')
fly10_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_b1neg.csv').as_matrix()
fly10_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_b2neg.csv').as_matrix()
fly10_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_iii1neg.csv').as_matrix()
fly10_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_iii2neg.csv').as_matrix()
fly10_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 5/triggered_fluor_i1neg.csv').as_matrix()

fly11_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_neg_sacc.csv', delimiter = ',')
fly11_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_b1neg.csv').as_matrix()
fly11_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_b2neg.csv').as_matrix()
fly11_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_iii1neg.csv').as_matrix()
fly11_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_iii2neg.csv').as_matrix()
fly11_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 6/triggered_fluor_i1neg.csv').as_matrix()

fly12_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_neg_sacc.csv', delimiter = ',')
fly12_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_b1neg.csv').as_matrix()
fly12_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_b2neg.csv').as_matrix()
fly12_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_iii1neg.csv').as_matrix()
fly12_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_iii2neg.csv').as_matrix()
fly12_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 7/triggered_fluor_i1neg.csv').as_matrix()

fly13_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_neg_sacc.csv', delimiter = ',')
fly13_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_b1neg.csv').as_matrix()
fly13_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_b2neg.csv').as_matrix()
fly13_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_iii1neg.csv').as_matrix()
fly13_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_iii2neg.csv').as_matrix()
fly13_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 9/triggered_fluor_i1neg.csv').as_matrix()

fly14_pos = np.genfromtxt('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_neg_sacc.csv', delimiter = ',')
fly14_b1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_b1neg.csv').as_matrix()
fly14_b2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_b2neg.csv').as_matrix()
fly14_iii1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_iii1neg.csv').as_matrix()
fly14_iii2 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_iii2neg.csv').as_matrix()
fly14_i1 = pd.read_csv('/Volumes/My Book/postdoc data/closed-loop muscle imaging/fly 10/triggered_fluor_i1neg.csv').as_matrix()

all_sacc = []
all_sacc = np.append(fly1_pos,fly2_pos, axis = 0)
all_sacc = np.append(all_sacc, fly3_pos, axis = 0)
all_sacc = np.append(all_sacc, fly4_pos, axis = 0)
all_sacc = np.append(all_sacc, fly5_pos, axis = 0)
all_sacc = np.append(all_sacc, fly6_pos, axis = 0)
all_sacc = np.append(all_sacc, fly7_pos, axis = 0)
all_sacc = np.append(all_sacc, fly8_pos, axis = 0)
all_sacc = np.append(all_sacc, fly9_pos, axis = 0)
all_sacc = np.append(all_sacc, fly10_pos, axis = 0)
all_sacc = np.append(all_sacc, fly11_pos, axis = 0)
all_sacc = np.append(all_sacc, fly12_pos, axis = 0)
all_sacc = np.append(all_sacc, fly13_pos, axis = 0)
all_sacc = np.append(all_sacc, fly14_pos, axis = 0)

all_b1 = []
all_b1 = np.append(fly1_b1,fly2_b1, axis = 1)
all_b1 = np.append(all_b1, fly3_b1, axis = 1)
all_b1 = np.append(all_b1, fly4_b1, axis = 1)
all_b1 = np.append(all_b1, fly5_b1, axis = 1)
all_b1 = np.append(all_b1, fly6_b1, axis = 1)
all_b1 = np.append(all_b1, fly7_b1, axis = 1)
all_b1 = np.append(all_b1, fly8_b1, axis = 1)
all_b1 = np.append(all_b1, fly9_b1, axis = 1)
all_b1 = np.append(all_b1, fly10_b1, axis = 1)
all_b1 = np.append(all_b1, fly11_b1, axis = 1)
all_b1 = np.append(all_b1, fly12_b1, axis = 1)
all_b1 = np.append(all_b1, fly13_b1, axis = 1)
all_b1 = np.append(all_b1, fly14_b1, axis = 1)
all_b1 = np.transpose(all_b1)

all_b2 = []
all_b2 = np.append(fly1_b2,fly2_b2, axis = 1)
all_b2 = np.append(all_b2, fly3_b2, axis = 1)
all_b2 = np.append(all_b2, fly4_b2, axis = 1)
all_b2 = np.append(all_b2, fly5_b2, axis = 1)
all_b2 = np.append(all_b2, fly6_b2, axis = 1)
all_b2 = np.append(all_b2, fly7_b2, axis = 1)
all_b2 = np.append(all_b2, fly8_b2, axis = 1)
all_b2 = np.append(all_b2, fly9_b2, axis = 1)
all_b2 = np.append(all_b2, fly10_b2, axis = 1)
all_b2 = np.append(all_b2, fly11_b2, axis = 1)
all_b2 = np.append(all_b2, fly12_b2, axis = 1)
all_b2 = np.append(all_b2, fly13_b2, axis = 1)
all_b2 = np.append(all_b2, fly14_b2, axis = 1)
all_b2 = np.transpose(all_b2)

all_iii1 = []
all_iii1 = np.append(fly1_iii1,fly2_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly3_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly4_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly5_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly6_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly7_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly8_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly9_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly10_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly11_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly12_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly13_iii1, axis = 1)
all_iii1 = np.append(all_iii1, fly14_iii1, axis = 1)
all_iii1 = np.transpose(all_iii1)

all_iii2 = []
all_iii2 = np.append(fly1_iii2,fly2_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly3_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly4_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly5_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly6_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly7_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly8_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly9_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly10_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly11_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly12_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly13_iii2, axis = 1)
all_iii2 = np.append(all_iii2, fly14_iii2, axis = 1)
all_iii2 = np.transpose(all_iii2)

all_i1 = []
all_i1 = np.append(fly1_i1,fly2_i1, axis = 1)
all_i1 = np.append(all_i1, fly3_i1, axis = 1)
all_i1 = np.append(all_i1, fly4_i1, axis = 1)
all_i1 = np.append(all_i1, fly5_i1, axis = 1)
all_i1 = np.append(all_i1, fly6_i1, axis = 1)
all_i1 = np.append(all_i1, fly7_i1, axis = 1)
all_i1 = np.append(all_i1, fly8_i1, axis = 1)
all_i1 = np.append(all_i1, fly9_i1, axis = 1)
all_i1 = np.append(all_i1, fly10_i1, axis = 1)
all_i1 = np.append(all_i1, fly11_i1, axis = 1)
all_i1 = np.append(all_i1, fly12_i1, axis = 1)
all_i1 = np.append(all_i1, fly13_i1, axis = 1)
all_i1 = np.append(all_i1, fly14_i1, axis = 1)
all_i1 = np.transpose(all_i1)


mean_b1 = []
mean_b2 = []
mean_iii1 = []
mean_iii2 = []
mean_i1 = []
mean_sacc = []

#calculate delta F/F and convert to percentages
for fly in range(len(all_sacc)):
    indiv_mean = (all_b1[fly]-np.mean(all_b1[fly][0:15]))/np.mean(all_b1[fly][0:15])
    mean_b1.append(indiv_mean)
    indiv_mean = (all_b2[fly]-np.mean(all_b2[fly][0:15]))/np.mean(all_b2[fly][0:15])
    mean_b2.append(indiv_mean)
    indiv_mean = (all_iii1[fly]-np.mean(all_iii1[fly][0:15]))/np.mean(all_iii1[fly][0:15])
    mean_iii1.append(indiv_mean)
    indiv_mean = (all_iii2[fly]-np.mean(all_iii2[fly][0:15]))/np.mean(all_iii2[fly][0:15])
    mean_iii2.append(indiv_mean)
    indiv_mean = (all_i1[fly]-np.mean(all_i1[fly][0:15]))/np.mean(all_i1[fly][0:15])
    mean_i1.append(indiv_mean)
    indiv_sacc = (all_sacc[fly]-np.mean(all_sacc[fly][0:1000]))
    mean_sacc.append(indiv_sacc)
#%% sort the data, and then find the deciles based on saccade magnitude
deciles = np.arange(10,100,10)

wba_sorted = []
#musc_sorted = np.argsort(np.sum(mean_iii1, axis = 1))
for i in range(len(mean_sacc)):
    wba = np.mean(mean_sacc[i][924:1201]-np.mean(mean_sacc[i][0:1000]))
    wba_sorted.append(wba)


wba_sorted = np.argsort(wba_sorted)



dec_inds = []

for i in range(len(deciles)):
    dec = round( (len(wba_sorted) - 1) * (deciles[i] / 100.) )
    dec_inds.append(dec)
    
decile_indices = []
tst =[]

for i in range(len(dec_inds)):
    decs = int(dec_inds[i])
    decile_indices.append(decs)
    #tst.append(decs)

#%% now find the saccade-triggered averages
b1musc_avgs = []
b2musc_avgs = []
iii1musc_avgs = []
iii2musc_avgs = []
i1musc_avgs = []
wing_avgs = []

plot_time_frames = np.arange(len(fly1_b1))
plot_time_frames = plot_time_frames * 0.033

plot_time = np.arange(len(fly1_pos[0]))
fs_axon = 1.0/2000.0
plot_time = plot_time * fs_axon

for i in range(0,2620,262):
    musc = np.mean(all_iii2[wba_sorted[i:i+262]], axis = 0)
    iii2musc_avgs.append(musc)
    musc = np.mean(all_i1[wba_sorted[i:i+262]], axis = 0)
    i1musc_avgs.append(musc)    
    musc = np.mean(all_iii1[wba_sorted[i:i+262]], axis = 0)
    iii1musc_avgs.append(musc)
    musc = np.mean(all_b1[wba_sorted[i:i+262]], axis = 0)
    b1musc_avgs.append(musc)
    musc = np.mean(all_b2[wba_sorted[i:i+262]], axis = 0)
    b2musc_avgs.append(musc)
    wing = np.mean(all_sacc[wba_sorted[i:i+262]], axis = 0)
    wing_avgs.append(wing)



plt.rc('font', size=8)
plt.rc('axes',linewidth=.5)
plt.rcParams['xtick.major.width'] = .5
plt.rcParams['ytick.major.width'] = .5
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

fig = plt.figure(figsize=(6,11))
fig.set_facecolor('w')
ax1 = fig.add_subplot(6, 2, 7)

line_color = np.linspace(0,1,len(iii1musc_avgs))
ax1.axvspan(.462, .65, facecolor = 'gray', edgecolor = 'none', alpha = 0.3)


for i in range(len(iii1musc_avgs)):
    ax1.plot(plot_time_frames, (iii2musc_avgs[i] - np.mean(iii2musc_avgs[i][0:15]))/np.mean(iii2musc_avgs[i][0:15]),color=plt.cm.BrBG(line_color[i]))
    
    
ax1.set_xlim((-0.05, 1.0))
ax1.set_ylabel('Muscle'r'$\Delta$' 'F/F (%)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.spines['left'].set_bounds(-0.01, 0.01)
ax1.yaxis.set_ticks_position('left')  
ax1.set_ylim((-.05, .05))
ax1.set_yticks([-.01, 0, .01])

labels = [item.get_text() for item in ax1.get_yticklabels()]
labels[0] = '-1'
labels[1] = '0'
labels[2] = '1'

ax1.set_yticklabels(labels)  

ax2 = fig.add_subplot(6, 2, 1)
ax2.axvspan(.462, .65, facecolor = 'gray', edgecolor = 'none', alpha = 0.3)

for i in range(len(wing_avgs)):    
    ax2.plot(plot_time, wing_avgs[i] - np.mean(wing_avgs[i][0:1000],axis=0),color=plt.cm.BrBG(line_color[i]))
    
#ax2.spines['bottom'].set_bounds(0.25, .75)
ax2.set_xlim((-0.05, 1.0))
ax2.set_ylabel('L-R WBA (arb. units)')
ax2.set_xticks([])
ax2.yaxis.set_ticks_position('left')                                  
#ax2.set_xlabel('Time (s)')
ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_bounds(-.5, .5)
ax2.set_ylim((-3., 3.))
ax2.set_yticks([-.5, 0, .5])        

    
msums = []
wsums = []
for i in range(len(iii1musc_avgs)):
#    musc_sum = np.mean((iii1musc_avgs[i][14:19] - np.mean(iii1musc_avgs[i][0:15]))/np.mean(iii1musc_avgs[i][0:15]))#np.sum(iii1musc_avgs[i][12:18] - np.mean(iii1musc_avgs[i][0:15]))    
#    msums.append(musc_sum)
    musc_sum = max((iii2musc_avgs[i][14:17] - np.mean(iii2musc_avgs[i][0:15]))/np.mean(iii2musc_avgs[i][0:15]), key=abs)#np.sum(iii1musc_avgs[i][12:18] - np.mean(iii1musc_avgs[i][0:15]))    
    #musc_sum = max(iii2musc_avgs[i])   
    msums.append(musc_sum)
    wing_sum = max(wing_avgs[i][924:1122]- np.mean(wing_avgs[i][0:1000],axis=0), key=abs)#np.sum(wing_avgs[i][800:1200]- np.mean(wing_avgs[i][0:1000],axis=0))
    wsums.append(wing_sum)    
    #wing_sum = np.argsort(np.abs(wing_avgs[i]))    
    #wsums.append(wing_avgs[i][wing_sum])

ax3 = fig.add_subplot(6, 2, 8)
ax3.plot(stats.zscore(msums,ddof=1),'--ko')
#ax3.plot(msums,'--ko')

#ax.spines['bottom'].set_bounds(0, 1)
ax3.set_xlim((-1, 10))
ax3.set_ylabel('z-score')
#ax.set_xticks([0, 1])                                
#ax.set_xlabel('Time (s)')
ax3.spines['top'].set_visible(False)
#ax3.spines['right'].set_visible(False)
#ax3.spines['bottom'].set_visible(False)

ax3.spines['right'].set_position('center')
ax3.spines['bottom'].set_position('center')
ax3.spines['left'].set_bounds(-1, 1)
ax3.get_xaxis().set_visible(False)
ax3.yaxis.set_ticks_position('left')  
ax3.spines['right'].set_bounds(-1, 1)
ax3.set_ylim((-3, 3))
ax3.set_yticks([-1, 0,1]) 
ax3.yaxis.set_ticks_position('left') 



ax4 = fig.add_subplot(6, 2, 2)
ax4.plot(stats.zscore(wsums,ddof=1),'--ko')
#ax4.spines['bottom'].set_bounds(0, 9)
ax4.set_xlim((-1, 10))
ax4.set_ylabel('z-score')
ax4.set_xticks([])
ax4.yaxis.set_ticks_position('left')                            
#ax4.set_xlabel('Amplitude decile')
ax4.spines['top'].set_visible(False)
#ax4.spines['right'].set_visible(False)
#ax4.spines['bottom'].set_visible(False)

ax4.spines['right'].set_position('center')
ax4.spines['bottom'].set_position('center')
ax4.spines['right'].set_bounds(-1, 1)
ax4.spines['left'].set_bounds(-1, 1)
ax4.set_ylim((-3.5, 3.5))
ax4.set_yticks([-1, 0, 1]) 
ax4.yaxis.set_ticks_position('left')
 




ax7 = fig.add_subplot(6, 2, 11)

ax7.set_xlim((-0.05, 1.0))

ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['left'].set_visible(False)
ax7.spines['bottom'].set_bounds(0.25, .75)
ax7.set_xticks([0.25, 0.5,.75])                                
ax7.set_xlabel('Time (s)')
ax7.yaxis.set_ticks_position('left')
ax7.xaxis.set_ticks_position('bottom')  
ax7.spines['left'].set_bounds(-0.01, 0.01)
ax7.set_ylim((-.05, .05))
ax7.set_yticks([])


   

msums = []

for i in range(len(iii2musc_avgs)):
    musc_sum = np.mean((iii2musc_avgs[i][14:19] - np.mean(iii2musc_avgs[i][0:15]))/np.mean(iii2musc_avgs[i][0:15]))#np.sum(i1musc_avgs[i][12:18] - np.mean(i1musc_avgs[i][0:15]))    
    msums.append(musc_sum)


ax8 = fig.add_subplot(6, 2, 12)

ax8.spines['bottom'].set_bounds(0, 9)
ax8.set_xlim((-1, 10))
#ax8.set_ylabel('z-score')
#ax.set_xticks([0, 1])                                
#ax.set_xlabel('Time (s)')
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
ax8.spines['left'].set_visible(False)
#ax8.spines['bottom'].set_visible(False)
ax8.set_xticks([0,1,2,3,4,5,6,7,8,9])                                
ax8.set_xlabel('Amplitude decile')
ax8.spines['left'].set_bounds(-.5, .5)
ax8.set_ylim((-3.5, 3.5))
ax8.set_yticks([]) 
ax8.yaxis.set_ticks_position('left') 
ax8.xaxis.set_ticks_position('bottom') 

ax9 = fig.add_subplot(6, 2, 3)
ax9.axvspan(.462, .65, facecolor = 'gray', edgecolor = 'none', alpha = 0.3)


for i in range(len(b1musc_avgs)):
    ax9.plot(plot_time_frames, (b1musc_avgs[i] - np.mean(b1musc_avgs[i][0:15]))/np.mean(b1musc_avgs[i][0:15]),color=plt.cm.BrBG(line_color[i]))

ax9.set_xlim((-0.05, 1.0))
ax9.set_ylabel('Muscle'r'$\Delta$' 'F/F (%)')
ax9.spines['top'].set_visible(False)
ax9.spines['right'].set_visible(False)
ax9.spines['bottom'].set_visible(False)
ax9.get_xaxis().set_visible(False)
ax9.spines['left'].set_bounds(-0.01, 0.01)
ax9.set_ylim((-.05, .05))
ax9.set_yticks([-.01, 0, .01])
ax9.yaxis.set_ticks_position('left')
labels = [item.get_text() for item in ax9.get_yticklabels()]
labels[0] = '-1'
labels[1] = '0'
labels[2] = '1'

ax9.set_yticklabels(labels)            

    
msums = []

#for i in range(len(b1musc_avgs)):
#    musc_sum = np.mean((b1musc_avgs[i][14:19] - np.mean(b1musc_avgs[i][0:15]))/np.mean(b1musc_avgs[i][0:15]))#np.sum(b1musc_avgs[i][12:18] - np.mean(b1musc_avgs[i][0:15]))   
#    msums.append(musc_sum)

for i in range(len(b1musc_avgs)):
    musc_sum = max((b1musc_avgs[i][14:17] - np.mean(b1musc_avgs[i][0:15]))/np.mean(b1musc_avgs[i][0:15]), key=abs)#np.sum(b1musc_avgs[i][12:18] - np.mean(b1musc_avgs[i][0:15]))   
    msums.append(musc_sum)

ax10 = fig.add_subplot(6, 2, 4)
ax10.plot(stats.zscore(msums,ddof=1),'--ko')
#ax.spines['bottom'].set_bounds(0, 1)
ax10.set_xlim((-1, 10))
ax10.set_ylabel('z-score')
#ax.set_xticks([0, 1])                                
#ax.set_xlabel('Time (s)')
ax10.spines['top'].set_visible(False)
#ax10.spines['right'].set_visible(False)
#ax10.spines['bottom'].set_visible(False)
ax10.spines['right'].set_position('center')
ax10.spines['bottom'].set_position('center')
ax10.get_xaxis().set_visible(False)
ax10.spines['left'].set_bounds(-1, 1)
ax10.spines['right'].set_bounds(-1, 1)
ax10.set_ylim((-3.5, 3.5))
ax10.set_yticks([-1, 0, 1]) 
ax10.yaxis.set_ticks_position('left')  


  

plt.savefig('/Volumes/My Book/postdoc data/closed-loop muscle imaging/' + '240422_allsacc_decsumm.pdf', format = 'pdf', dpi = 600)