# Zepei Zhao (5*2)
# Mingjun Liu (1*9)

import numpy as np
import math

with open("hmm-data.txt") as f:
    grid = []
    tLoc = []
    nDist = []
    f.readline()
    f.readline()

    for i in range(10):
        grid.append([int(j) for j in f.readline().split()])
    for i in range(4):
        f.readline()
    for i in range(4):
        tLoc.append([int(j) for j in f.readline().split()[2:]])
    for i in range(4):
        f.readline()
    for i in range(11):
        nDist.append([float(j) for j in f.readline().split()])
print('nDist',nDist)

valid = []
for i in range(len(grid)):
    for j,ele in enumerate(grid[i]):
        if ele != 0:
            valid.append([i,j])
print('valid',valid)

def dist_tower(valid,tLoc):
    dist_t = []
    for i,j in enumerate(valid):
        tmp = []
        for h,d in enumerate(tLoc):
            edu_dist = math.sqrt((j[0]-d[0])**2 + (j[1]-d[1])**2)
            tmp.append([edu_dist*0.7,edu_dist*1.3])
        dist_t.append(tmp)
    return dist_t

def neighbor(cell,size):
    x =cell[0]
    y = cell[1]
    neighbor = []
    if x + 1 < size:
        neighbor.append((x + 1,y))
    if y + 1 < size:
        neighbor.append((x,y + 1))
    if x - 1 > 0:
        neighbor.append((x - 1,y))
    if y - 1 > 0:
        neighbor.append((x,y - 1))
    return neighbor

def prob_cells(valid,nDist,dist_t):
    prob = []
    for i in range(len(valid)):
        count = 0
        for j in range(len(nDist)):
            if dist_t[i][j][0] <= nDist[j] and dist_t[i][j][1] >= nDist[j]:
                count += 1
        if count == len(nDist):
            prob.append(valid[i])
    return prob

dist_t = dist_tower(valid,tLoc)
print('dist_t',dist_t)
# print('len_ndsit',len(nDist))

prob_state = {}
state = {}
for i in range(len(nDist)):
    prob_state[i] = prob_cells(valid,nDist[i],dist_t)
    for j in prob_state[i]:
        if tuple(j) not in state:
            state[tuple(j)] = []
        state[tuple(j)].append(i)
print('prob_state',prob_state)
print('state',state)

neighbor_ = {}
for i in state:
    neighbor_[i] = neighbor(i,10)
print('neighbor',neighbor_)


def trans_p(state,neighbor_):
    total_ = {}
    trans_neighbor = {}
    trans_p = {}
    for i in state:
        trans_neighbor[i] = {}
        trans_p[i] = {}
        total_[i] = 0
        tmp = state[i]
        n = neighbor_[i]
        for j in tmp:
            for h in n:
                if (h in state) and (j+1 in state[h]):
                    if h not in trans_neighbor[i]:
                        trans_neighbor[i][h] = 0
                    trans_neighbor[i][h] += 1
                    total_[i] += 1
        for nei in trans_neighbor[i]:
            trans_p[i][nei] = trans_neighbor[i][nei]/total_[i]
    return trans_p
trans_p = trans_p(state,neighbor_)
print('trans_p',trans_p)

def viterbi(valid,tLoc,nDist,dist_t,prob_state,neighbor,trans_p):
    step = 0
    s = []
    path = {}
    path[step] = {}
    for i in prob_state[step]:
        i = tuple(i)
        path[step][i] = {}
        path[step][i]['previous'] = None
        path[step][i]['probability'] = 1.0/len(prob_state[step])
    for step in range(1,len(nDist)):
        path[step] = {}
        for i in path[step-1]:
            if i in trans_p:
                for n in trans_p[i]:
                    if list(n) in prob_state[step]:
                        if n not in path[step]:
                            path[step][n] = {}
                            path[step][n]['previous'] = i
                            curr = path[step-1][i]['probability']*trans_p[i][n]
                            path[step][n]['probability'] = curr
                        else:
                            curr = path[step-1][i]['probability']*trans_p[i][n]
                            if curr > path[step][n]['probability']:
                                path[step][n]['previous'] = i
                                path[step][n]['probability'] = curr
    max_ = -1
    result = []
    for h in path[10]:
        if max_ < path[step][h]['probability']:
            max_ = path[step][h]['probability']
            cell = h
    result.append(cell)
    for s in range(10,0,-1):
        previous = path[s][cell]['previous']
        result.append(previous)
        cell = previous
    return result

result = viterbi(valid,tLoc,nDist,dist_t,prob_state,neighbor_,trans_p)
print('path',result[::-1])




