# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 01:38:42 2020

@author: marga
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math


with open('fastmap-wordlist.txt','r') as f:
    item = f.readlines()
item = [i.strip() for i in item]

dismap = np.loadtxt('fastmap-data.txt')

def fastmapalg(dismap,num_coor):

    fastmap = []
    k = 1
    
    while k <= num_coor:  
    
        dis = dismap[:,2]
        dismax = np.amax(dis)
        indexmax = np.where(dis == dismax)[0][0]
        
        oa, ob = dismap[indexmax , 0:2]
        oa = int(oa)
        ob = int(ob)
        
        coordinate = []
        
        for i in range(1 , (len(item)+1)):
            if i == oa:
                xi = 0
                coordinate.append(xi)
                
                continue
            if i == ob:
                xi = dismax
                coordinate.append(xi)
                continue
            
            # get row number
            ai = np.where(((dismap[:,0] == oa) * (dismap[:,1] == i)) | (dismap[:,1] == oa) * (dismap[:,0] == i))[0][0]
            bi = np.where(((dismap[:,0] == ob) * (dismap[:,1] == i)) | (dismap[:,1] == ob) * (dismap[:,0] == i))[0][0]
            
            # get xi
            dai = dis[ai]
            dbi = dis[bi]
            xi = (dai**2 + dismax**2 - dbi**2) / (2 * dismax)
            coordinate.append(xi)
      
        for row in dismap:
            a = int(row[0])
            b = int(row[1])
            dold = row[2]
            xa = coordinate[a-1]
            xb = coordinate[b-1]
            row[2] = math.sqrt(dold**2 - (xa - xb)**2)
            
        fastmap.append(coordinate)
        k = k+1
        
    return np.array(fastmap).T
        
        
        
result = fastmapalg(dismap, 2)




    






