# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:28:12 2019
2020/01/13 Plot 3D crystal strucutre and then cut it into a retangular shape
            Quick method to find matrix elements using Pandat
2020/01/15 Adjust the method to find neighbors quickly with the help of MK
2020/01/15 The center's brightness could be weaker than its negbours 
            Change the code for Al-Mg        
2020/01/17 Add a cycle for calculating Al-Al and Mg-Mg sequentially
2020/01/24 add new plotting style 
                from matplotlib.colors import BoundaryNorm
                from matplotlib.ticker import MaxNLocator
2020/02/12 3D scan RO  
2020/02/13 output figures, clear files  
2020/02/19 new generated file's name is given by add '00' to make the read file sequence is according to 1 2 3 ...11        
2020/03/16 generating 50, 54, 4, 2 images
2020/03/23 generating 40, 70, 4, 2 images
2020/03/24 generating 0, 180, 4, 2 images
2020/03/24 generating 0, 190, 4, 2 images
2020/04/01 recording running time
2020/04/03 send to Leigh for optimizing the code
2020/04/07 Optimize code according to Leigh's suggestions
2020/04/14 Reduced from 1.5h to 35 mins
2020/04/16 To CuAu
2020/05/23 To FeAl
2020/08/03 ZSDMs Curve
2020/09/22 ZSDMs Curve for Fe and Al
2020/10/08 generating training data
2020/10/23 v14 pos
2020/11/26 0.5nm
2020/12/08 12, 12, 60 continue comment
2020/12/08 parallel
2020/12/28 CoCrNi
2021/01/04 add read pos and disable return
2021/01/05 to FeAl with PFIB data and add data.columns for the .csv file from AP suite
@author: yue.li
"""

import numpy as np
# import matplotlib.pyplot as plt
from scipy import spatial
import pandas as pd 
# import random as rd
# import math
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import MaxNLocator
# import shutil,sys
import os
import datetime
from itertools import product
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fast_histogram import histogram2d as fast_histogram2d 
# import multiprocessing
from joblib import Parallel, delayed
#from tqdm import tqdm
from os import listdir
import re

# num_cores = multiprocessing.cpu_count()


def ZSDM(count):
    # zSDM_simu_0_pre = np.zeros((int(0.69/0.015*2+1), 2))
    # zSDM_simu_1_pre = np.zeros((int(0.69/0.015*2+1), 2))
    # count = 1
    # temp_starttime = datetime.datetime.now()
    data_voxel_sphere = element_chosen[index_voxel_sphere[count,],:]
    if len(data_voxel_sphere) == 0:
        # print ('blank image')
        x_zSDM = np.arange(-0.69, 0.7, (0.69*2+0.015)/93).reshape((93, 1))
        y_zSDM = np.zeros((93, 1))
        zSDM_simu = np.concatenate((x_zSDM, y_zSDM), axis=1)
        return zSDM_simu 
        # fig2D = plt.figure(figsize=(4,4))
        # ax2D = fig2D.add_subplot(111)
        # plt.axis('off')
        # if element == 'Fe':
        #     if count < 10: 
        #         fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_00000%s.png'%(count),dpi=SDM_bins)
        #     elif count<100:
        #         fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_0000%s.png'%(count),dpi=SDM_bins)                                    
        #     elif count<1000:
        #         fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_000%s.png'%(count),dpi=SDM_bins)                                                                        
        #     elif count<10000:
        #         fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_00%s.png'%(count),dpi=SDM_bins)  
        #     elif count<100000:
        #         fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_0%s.png'%(count),dpi=SDM_bins)  
        # else:
        #     if count < 10: 
        #         fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_00000%s.png'%(count),dpi=SDM_bins)
        #     elif count<100:
        #         fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_0000%s.png'%(count),dpi=SDM_bins)                                    
        #     elif count<1000:
        #         fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_000%s.png'%(count),dpi=SDM_bins)                                                                        
        #     elif count<10000:
        #         fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_00%s.png'%(count),dpi=SDM_bins)  
        #     elif count<100000:
        #         fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_0%s.png'%(count),dpi=SDM_bins) 
        # plt.close()   
        # return
    #Looking for neighbor quickly using tree.query_ball_point
    SDM = np.zeros([SDM_bins,SDM_bins])
    x_tot = [];
    y_tot = [];
    num_in_SDM = 0;
    max_cand =0
    cand = tree.query_ball_point(data_voxel_sphere, 1.5,return_sorted=False, n_jobs = 1)
    for list in cand:
        num_in_SDM += len(list);
        if (len(list) > max_cand):
            max_cand = len(list);
    x_tot = np.zeros([num_in_SDM,], dtype = np.float32)
    y_tot = np.zeros([num_in_SDM,], dtype = np.float32)
    x = np.zeros([max_cand,], dtype = np.float32)   
    y = np.zeros([max_cand,], dtype = np.float32)   
    
    start = 0;
    i = 0;
    for list in cand:
        length = len(list)
        x_tot[start:(start+length)] = np.ndarray.__sub__(element_chosen[list,0],data_voxel_sphere[i,0]);
        y_tot[start:(start+length)] = np.ndarray.__sub__(element_chosen[list,2],data_voxel_sphere[i,2]);
        i += 1
        start = start+length;
    notzero = (x_tot!=0)*(y_tot!=0);
    SDM = fast_histogram2d(y_tot[notzero],x_tot[notzero], range = [[-1.5,1.5],[-1.5,1.5]],  bins=SDM_bins)
    #Leigh: https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/ <-- this offers some clues, I used fast histogram mentioned there.
    #%%z-SDMs curve
    y_zSDM = SDM.sum(axis=1).reshape((200, 1))
    # sum_y_zSDM = sum(y_zSDM)
    # y_fre = y_zSDM/sum_y_zSDM*100
    # y_fre = np.array(y_fre).reshape((len(y_fre), 1))
            
    # # scaler = StandardScaler()
    # scaler = MinMaxScaler() 
    # scaler.fit(y_fre)
    # y_fre_scale = scaler.transform(y_fre)          #normalization
    
    x_zSDM = np.arange(-1.5, 1.5, 3/200).reshape((200, 1))
    # zSDM_simu = np.zeros((200, 1))
    zSDM_simu = np.concatenate((x_zSDM, y_zSDM), axis=1)
    zSDM_simu_index = np.where((zSDM_simu[:,0]>=-0.7) & (zSDM_simu[:,0]<0.7))
    zSDM_simu_index_array = np.array(zSDM_simu_index).reshape(-1, 1)
    zSDM_simu_part = zSDM_simu[zSDM_simu_index_array[0, 0]:zSDM_simu_index_array[-1, 0]+1, ]
    
    # #Plot
    # fig2D = plt.figure(figsize=(4,4))                
    # ax2D = fig2D.add_subplot(111)   
    # plt.plot(zSDM_simu_part[: ,0], zSDM_simu_part[: ,1])
    # # ax2D.set_aspect(1)
    # plt.xlim((-0.7,0.7))
    # # plt.ylim((0.0,1.0))
    # # ax2D.set_xticks([])
    # # ax2D.set_yticks([])
    # # plt.axis('off')
    # # plt.subplots_adjust(0,0,1,1) 
    # plt.close() 
    # if element_num == 1:
    #     zSDM_simu_0_pre[:, count*2:(count*2+2)] = zSDM_simu_part   # know where to put zSDM
    #     # if count < 10: 
    #     #     fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_00000%s.png'%(count),dpi=SDM_bins)
    #     # elif count<100:
    #     #     fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_0000%s.png'%(count),dpi=SDM_bins)                                    
    #     # elif count<1000:
    #     #     fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_000%s.png'%(count),dpi=SDM_bins)                                                                        
    #     # elif count<10000:
    #     #     fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_00%s.png'%(count),dpi=SDM_bins)  
    #     # elif count<100000:
    #     #     fig2D.savefig('Results_ZSDMs\Fe_Fe\Fe_Fe_0%s.png'%(count),dpi=SDM_bins)  
    # else:
    #     zSDM_simu_1_pre[:, count*2:(count*2+2)] = zSDM_simu_part
    #     # if count < 10: 
    #     #     fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_00000%s.png'%(count),dpi=SDM_bins)
    #     # elif count<100:
    #     #     fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_0000%s.png'%(count),dpi=SDM_bins)                                    
    #     # elif count<1000:
    #     #     fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_000%s.png'%(count),dpi=SDM_bins)                                                                        
    #     # elif count<10000:
    #     #     fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_00%s.png'%(count),dpi=SDM_bins)  
    #     # elif count<100000:
    #     #     fig2D.savefig('Results_ZSDMs\Al_Al\Al_Al_0%s.png'%(count),dpi=SDM_bins)   
    # # temp_endtime = datetime.datetime.now()
    # # print ('The running time of SDM %s/%s= '%(count, len(index_voxel_sphere)), temp_endtime-temp_starttime)   
    return zSDM_simu_part

def atom_filter(x, Atom_range):
    Atom_total = pd.DataFrame()
    for i in range(len(Atom_range)):
        Atom = x[x['Da'].between(Atom_range['lower'][i], Atom_range['upper'][i], inclusive=True)]
        Atom_total = Atom_total.append(Atom)
        # Count_Atom= len(Atom_total['Da'])   
    return Atom_total[['x','y','z']]

def read_rrng(f):
    rf = open(f,'r').readlines()
    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')
    ions = []
    rrngs = []
    for line in rf:
        m = patterns.search(line)
        if m:
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            else:
                rrngs.append(m.groups()[2:])
    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True) 
    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)
    return ions, rrngs

def readpos(file_name):
    f = open(file_name, 'rb')
    dt_type = np.dtype({'names':['x', 'y', 'z', 'm'], 
                  'formats':['>f4', '>f4', '>f4', '>f4']})
    pos = np.fromfile(f, dt_type, -1)
    f.close()
    return pos
#%%
if __name__ == "__main__":
    #%% record start time
    starttime = datetime.datetime.now()
    #%% Build output file
    # try:
    #     shutil.rmtree('Results_ZSDMs\Fe_Fe')  
    #     shutil.rmtree('Results_ZSDMs\Al_Al')  
    # except:
    #     print("file 1 does not exist")
    # os.mkdir('Results_ZSDMs\Fe_Fe') 
    # os.mkdir('Results_ZSDMs\Al_Al') 
    #%% Input data
    folder = 'data'
    
    # for filename in tqdm(os.listdir(folder)):    
    #     print(filename)
    #     pos = readpos(folder+'/'+filename)
    #     dpos = pd.DataFrame({'x':pos['x'],
    #                                 'y': pos['y'],
    #                                 'z': pos['z'],
    #                                 'Da': pos['m']})
    #     dpos.to_csv(folder+'/'+'{}.csv'.format(filename), index=False)  

    data_name = [file for file in listdir(folder) if file.endswith('.csv')]
    data_name = data_name[0]
    data = pd.read_csv(folder+'/'+data_name)
    data.columns =['x', 'y', 'z', 'Da']

    
    #Finding element 1 Fe and element 2 Al    
    rrange_file = 'recon_v01.RRNG'
    ions, rrngs = read_rrng(rrange_file)
    element_1_range = rrngs[rrngs['comp']=='Fe:1']
    element_2_range = rrngs[rrngs['comp']=='Al:1']
    # element_3_range = rrngs[rrngs['comp']=='Ni:1']
    
    element_1 = atom_filter(data, element_1_range)
    element_2 = atom_filter(data, element_2_range)
    # element_3 = atom_filter(data, element_3_range)
    
    # element_1 = element_1
    # get max, min of each column    
    # df = pd.DataFrame(data,columns=['a','b','c','d'])  #'a', 'b', 'c', 'd' for x,y,z,m. 
    data_min = data.min()
    data_min['c'], data_min['b'], data_min['a'] = -70, -10, -8  #nm
    data_max = data.max()
    data_max['c'], data_max['b'], data_max['a'] = 0, 10, 8   #nm
   
    # row_number = data.shape[0]
    # detect_eff = 0.52
    #%%Finding element 1 Fe and element 2 Al
    # element_1 = df.loc[((df['d'] >= 27.832) & (df['d'] <= 28.115)) |
    #                    ((df['d']>=26.929) & (df['d'] <=27.038)) |
    #                    ((df['d']>=28.436) & (df['d'] <=28.542)) |
    #                    ((df['d']>=28.919) & (df['d'] <=29.039)) , ['a','b','c']]  
    # element_2 = df.loc[((df['d'] >= 13.444) & (df['d'] <=13.558)) | 
    #                     ((df['d'] >= 8.99) & (df['d'] <=9.02))  , ['a','b','c']]  
    #Note to use & or | to replace and/or
    element_num=0
    
    #%% scanning parameters 
    voxel = 2   #nm
    stride = 0.5   #nm
    #%%Building sphere heart array
    data_Z_list = list(np.arange(int(data_min['c']), int(data_max['c']), stride))
    data_Y_list = list(np.arange(int(data_min['b']), int(data_max['b']), stride))
    data_X_list = list(np.arange(int(data_min['a']), int(data_max['a']), stride))
    data_sphere_points = np.zeros((1,3))
    for data_Z, data_Y, data_X in product(data_Z_list, data_Y_list, data_X_list):
        if data_Z+voxel > data_max['c'] or data_Y+voxel > data_max['b'] or data_X+voxel > data_max['a']:
            continue
        else:
            temp = np.array([data_X+voxel/2, data_Y+voxel/2, data_Z+voxel/2]).reshape((1,3)) 
            data_sphere_points = np.concatenate((data_sphere_points, temp), axis=0)
    data_sphere_points = data_sphere_points[1:]
    
    for element_chosen in (element_1.values, element_2.values) :  #
        element_num = element_num+1
        print ('The chosen element is ', element_num)
    
        #%%create sphere voxels using tree.query_ball_point
        print('Please wait for a moment')
        tree = []
        temp_starttime = datetime.datetime.now()
        tree = spatial.cKDTree(element_chosen)
        temp_endtime = datetime.datetime.now()
        print ('The running time of spatial.cKDTree = ', temp_endtime-temp_starttime)
        #Yue: approximately 1 min on my computer
        #%%
        index_voxel_sphere = tree.query_ball_point(data_sphere_points, voxel/2, n_jobs = 1) 
        #-1 for all processors
        #%% generating SDMs 
        SDM_bins = 200   #define pixel density
        zSDM_output = np.zeros((int(0.69/0.015*2+1), len(index_voxel_sphere)*2))
        myList = range(0,len(index_voxel_sphere))
        zSDM_output = Parallel(n_jobs=1, verbose=2)(delayed(ZSDM)(i) for i in myList)
  
        # zSDM_simu_0_pre, zSDM_simu_1_pre = zip(*zSDM_output)
        if element_num == 1:
            zSDM_simu_1 = zSDM_output   # know where to put zSDM 
        if element_num == 2:
            zSDM_simu_2 = zSDM_output
        # if element_num == 3:
        #     zSDM_simu_3 = zSDM_output
    #%%save
    # zSDM_simu_0 = zSDM_simu_0[:,1:]
    # zSDM_simu_1 = zSDM_simu_1[:,1:]
    # np.save ("zSDM_exp_test_Fe_%s_%s_10_10_2_0.5.npy"%(int(data_min['c']), int(data_max['c'])), zSDM_simu_1)
    # np.save ("zSDM_exp_test_Al_%s_%s_10_10_2_0.5.npy"%(int(data_min['c']), int(data_max['c'])), zSDM_simu_2)
    # np.save ("zSDM_exp_test_Ni_%s_%s_5_5_2_2.npy"%(int(data_min['c']), int(data_max['c'])), zSDM_simu_3)
    #%% recording endtime
    endtime = datetime.datetime.now()
    print ('Total running time = ', endtime-starttime)
