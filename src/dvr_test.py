#!/usr/bin/env python
import numpy as np
from numpy import linalg as LA
import math
from copy import copy
from copy import deepcopy
from scipy.linalg import eigh
import scipy.linalg as sp
import itertools
from tqdm import tqdm
from numba import jit
# -
import junit as ju
# import para
import env_vars as ev
import pes
import get_shift.all_py as gs
import m_tool as mt


#def sinc_dvr_1d(m , N, a, b, x_arr, e_arr, wfs, T_mtx, pot):
def sinc_dvr_1d(pot, get_h= False):
    a = ev.a
    b = ev.b
    # a = ev.b
    # b = ev.a
    m = ev.m
    N = ev.n_sinc_dvr
    delta_x = (b-a)/N
    # print(pot(0.0), a,b,m, N)
    x_arr = np.array([ i*delta_x+a for  i in range(1,N )])
    #print(x)
    #print(delta_x, len(x))
    
    # make H_dvr
    h_dvr = np.zeros((N-1, N-1))
    t_dvr = np.zeros((N-1, N-1))
    for i in range(1,N):
        for j in range(1, i):
            t_dvr[i-1,j-1] = 1.0/(2*m)* pow(-1, i-j) / pow(b-a, 2)                            *math.pi*math.pi/2.0                            *(                              1/pow(math.sin(math.pi*(i-j)/(2*N)), 2)                             -1/pow(math.sin(math.pi*(i+j)/(2*N)), 2)                             )
            t_dvr[j-1, i-1] = t_dvr[i-1, j-1]
            h_dvr[i-1, j-1] = t_dvr[i-1, j-1]
            h_dvr[j-1, i-1] = t_dvr[j-1, i-1]
            
    for i  in range(1,N):
        t_dvr[i-1, i-1] = 1.0/(2*m)*1.0/pow(b-a,2) *math.pi*math.pi/2.0*(                          (2.0*N*N +1.0)/3.0-1/pow(math.sin(math.pi*i/N),2)                          )
        h_dvr[i-1, i-1] = t_dvr[i-1, i-1] + pot(x_arr[i-1])
    
    
    values, vectors = eig(h_dvr)
    #set_sign_(vectors)
    #set_sign_v2_(vectors)
    if get_h:
        return values, x_arr, h_dvr, vectors
    return values, vectors, x_arr
    
def sinc_dvr_1d_v2(pot):
    a = ev.a
    b = ev.b
    m = ev.m
    N = ev.n_sinc_dvr
    delta_x = (b-a)/N
    x_arr = np.array([ i*delta_x+a for  i in range(1,N )])
    #print(x)
    #print(delta_x, len(x))
    
    # make H_dvr
    h_dvr = np.zeros((N-1, N-1))
    t_dvr = np.zeros((N-1, N-1))
    for i in range(1,N):
        for j in range(1, i):
            t_dvr[i-1,j-1] = 1.0/(2*m)* pow(delta_x, 2) *pow(-1, (i-j))*2                            / pow((i-j), 2)
            t_dvr[j-1, i-1] = t_dvr[i-1, j-1]
            h_dvr[i-1, j-1] = t_dvr[i-1, j-1]
            h_dvr[j-1, i-1] = t_dvr[j-1, i-1]
            
    for i  in range(1,N):
        t_dvr[i-1, i-1] = 1.0/(2*m*pow(delta_x, 2))*pow(math.pi, 2)/3.0
        h_dvr[i-1, i-1] = t_dvr[i-1, i-1] + pot(x_arr[i-1])
    
    values, vectors = eig(h_dvr)
    return values, vectors, x_arr
    
def eig(mtx):
    eigenValues, eigenVectors = LA.eig(mtx)
    idx = eigenValues.argsort()   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors

def pot_1d(x):
    res = x*x*.5
    return res

def pot_1d_x(x):
    res = x*x*.5
    return res

def pot_1d_y(y):
    res = y*y*.5*1.01
    return res

def pot_1d_z(z):
    res = z*z*.5*1.02
    return res

def pot_3d_harm(a):
    res = [pot_1d_x(a[0]) + pot_1d_y(a[1]) + pot_1d_z(a[2])]
    return res


def mel(left, o, right):
    res = 0.0
    for i in range(len(o)):
        res += (left[i]*o[i]*right[i])
    return res

def set_sign_v2_(mtx, axis = 0):
    # unify the sign, assume c of 0 state are always positive
    if axis == 0:
        for i in range(mtx.shape[0]):
            if mtx[i, 0] < 0:
                mtx[i, :] *= -1.0
    elif axis == 1:
        for i in range(mtx.shape[1]):
            if mtx[0, i] < 0:
                mtx[:, i] *= -1.0

def set_sign_(mtx, axis = 0):
    if axis == 0:
        for i in range(mtx.shape[0]):
            for j in range(mtx.shape[1]):
                if abs(mtx[j, i]) > 0.01:
                    #print(f"check change for {mtx[j, i]} at idx {j} state {i}")
                    if mtx[j,i] * pow(-1, i) < 0:
                        #print(f"need change for {mtx[j, i]} at idx {j} state {i}")
                        mtx[:, i]*= -1.0
                        #print(f"after change is {mtx[j, i]}")
                        break
                    else:
                        break
    if axis == 1:
        for i in range(mtx.shape[1]):
            for j in range(mtx.shape[0]):
                if abs(mtx[i, j]) > 0.01:
                    print(f"check change for {mtx[j, i]} at idx {j} state {i}")
                    if mtx[i,j] * pow(-1, i) < 0:
                        #print(f"need change for {mtx[j, i]} at idx {j} state {i}")
                        mtx[i, :]*= -1.0
                        print(f"after change is {mtx[j, i]}")
                        break
                    else:
                        break
    return mtx
    #if axis == 0:
    #    for i in range(mtx.shape[0]):
    #        if mtx[i, 0] < 0:
    #            mtx[i, :]*= -1
    #elif axis ==1:
    #    for i in range(mtx.shape[1]):
    #        if mtx[0, i] < 0:
    #            mtx[:, i]*= -1
    #elif axis ==2:
    #    if np.max(mtx[:,0])<0.000001:
    #        mtx*=-1

def sinc_podvr_1d(pot):
    n_po = ev.n_po
    levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(pot)
    #set_sign_(wfs_dvr, 2)
    X = np.zeros((n_po, n_po))
    for i in range(n_po):
        for j in range(i+1):
            X[i,j] = mel(wfs_dvr[:, i], sinc_grids, wfs_dvr[:, j])
            X[j,i] = X[i,j]
        
    
    grids_x, wfs_x = eig(X)
    #print(wfs_x)
    #print(wfs_x.shape)
    set_sign_v2_(wfs_x, 1)
    #set_sign_(wfs_x, 1)
    #print(wfs_x)
    h_podvr = np.matmul(np.matmul(wfs_x.transpose(), np.diag(levels_dvr[:n_po])), wfs_x)
    #print("after multiply:")
    #print(wfs_x)
    wfs_e = copy(wfs_x.transpose())
    
    return levels_dvr[:n_po], grids_x, h_podvr, wfs_e
    
def idx_2_to_1(i1,i2, N):
    res = i1*N + i2
    return res

def idx_3_to_1(i1,i2,i3, N):
    res = i1*N*N + i2 *N + i3
    return res

def get_sinc_grids():
    a = ev.a
    b = ev.b
    m = ev.m
    N = ev.n_sinc_dvr
    delta_x = (b-a)/N
    x_arr = np.array([ i*delta_x+a for  i in range(1,N )])
    return x_arr

def delta(i,j):
    if i == j:
        return 1.0
    else:
        return 0

def d2_podvr(pot1, pot2):
    n_po = ev.n_po
    levels_dvr1, grids_x1, h_1d1, wfs_podvr1 = sinc_podvr_1d(pot1)
    levels_dvr2, grids_x2, h_1d2, wfs_podvr2 = sinc_podvr_1d(pot2)
    h = np.zeros((n_po*n_po, n_po*n_po))
    # i,j : left and right, 1,2: 1st and 2nd freedom
    for i1 in range(n_po):
        for i2 in range(n_po):
            for j1 in range(n_po):
                for j2 in range(n_po):
                    i = idx_2_to_1(i1, i2, n_po)
                    j = idx_2_to_1(j1, j2, n_po)
                    h[i,j] = h_1d1[i1,j1]* delta(i2,j2)+ h_1d2[i2,j2] * delta(i1,j1)
    #for i in range(n_po*n_po):
    #    for j in range(i):
    #        h[i,j] = h[j, i]
    ##print(h)
    levels, wfs = eig(h)
    set_sign_v2_(wfs)
    return levels_dvr1, levels, wfs

# @jit
def d3_sincdvr(pot1, pot2, pot3, v, v_trj, v_ref):
    levels_dvr1, grids_x1, h_1d1, wfs_podvr1 = sinc_dvr_1d(pot1, get_h = True)
    levels_dvr2, grids_x2, h_1d2, wfs_podvr2 = sinc_dvr_1d(pot2, get_h = True)
    levels_dvr3, grids_x3, h_1d3, wfs_podvr3 = sinc_dvr_1d(pot3, get_h = True)

    # print(h_1d1[49, 49])
    n_po = ev.n_sinc_dvr-1
    h = np.zeros((n_po*n_po*n_po, n_po*n_po*n_po))

    delta_v = np.zeros(n_po**3)
    # for i1, i2, i3, j1, j2, j3 in tqdm(itertools.product(range(n_po), repeat=6)):
    for i1 in range(n_po):
        for i2 in range(n_po):
            for i3 in range(n_po):
                for j1 in range(n_po):
                    for j2 in range(n_po):
                        for j3 in range(n_po):
                            i = idx_3_to_1(i1, i2, i3, n_po)
                            j = idx_3_to_1(j1, j2, j3, n_po)
                            h[i,j] = h_1d1[i1,j1]* delta(i2,j2) * delta(i3,j3)\
                                     + h_1d2[i2,j2] * delta(i1,j1) * delta(i3, j3)\
                                     + h_1d3[i3, j3]* delta(i1,j1) * delta(i2,j2)
                            if i == j:
                                h[i, j] +=(\
                                            v([grids_x1[i1], grids_x2[i2], grids_x3[i3]])[0]\
                                             - pot1(grids_x1[i1])\
                                             - pot2(grids_x2[i2])\
                                             - pot3(grids_x3[i3])\
                                             + delta_v[i])

    levels, wfs = eig(h)
    set_sign_v2_(wfs)
    return levels, wfs

# @jit
def d3_podvr_surf(pot1, pot2, pot3, v, v_trj, v_ref):
    # v: v_ref_abin
    # v_ref: v_ref_em
    # v_trj: v_em in MD trajectory
    # delta_v = v_trj -v_ref
    n_po = ev.n_po
    # delta_v = np.zeros(n_po**3)
    # print(pot1(0.0))
    levels_dvr1, grids_x1, h_1d1, wfs_podvr1 = sinc_podvr_1d(pot1)
    levels_dvr2, grids_x2, h_1d2, wfs_podvr2 = sinc_podvr_1d(pot2)
    levels_dvr3, grids_x3, h_1d3, wfs_podvr3 = sinc_podvr_1d(pot3)
    h = np.zeros((n_po*n_po*n_po, n_po*n_po*n_po))

    # print(grids_x1)
    # print(grids_x2)
    # print(grids_x3)

    # i,j : left and right, 1,2,3: 1st and 2nd freedom
    for i1 in range(n_po):
        for i2 in range(n_po):
            for i3 in range(n_po):
                for j1 in range(n_po):
                    for j2 in range(n_po):
                        for j3 in range(n_po):
                            i = idx_3_to_1(i1, i2, i3, n_po)
                            j = idx_3_to_1(j1, j2, j3, n_po)
                            h[i,j] = h_1d1[i1,j1]* delta(i2,j2) * delta(i3,j3)\
                                     + h_1d2[i2,j2] * delta(i1,j1) * delta(i3, j3)\
                                     + h_1d3[i3, j3]* delta(i1,j1) * delta(i2,j2)
                            if i == j:
                                posi = [grids_x1[i1], grids_x2[i2], grids_x3[i3]]
                                h[i, j] +=(\
                                            v(posi)\
                                             - pot1(grids_x1[i1])\
                                             - pot2(grids_x2[i2])\
                                             - pot3(grids_x3[i3])\
                                             + v_trj(posi)- v_ref(posi)
                                             )

    levels, wfs = eig(h)
    set_sign_v2_(wfs)
    return levels, wfs

def d3_podvr(pot1, pot2, pot3, v, v_trj, v_ref):
    # v: v_ref_abin
    # v_ref: v_ref_em
    # v_trj: v_em in MD trajectory
    # delta_v = v_trj -v_ref
    n_po = ev.n_po
    delta_v = np.zeros(n_po**3)
    # print(pot1(0.0))
    levels_dvr1, grids_x1, h_1d1, wfs_podvr1 = sinc_podvr_1d(pot1)
    levels_dvr2, grids_x2, h_1d2, wfs_podvr2 = sinc_podvr_1d(pot2)
    levels_dvr3, grids_x3, h_1d3, wfs_podvr3 = sinc_podvr_1d(pot3)
    h = np.zeros((n_po*n_po*n_po, n_po*n_po*n_po))

    # print(grids_x1)
    # print(grids_x2)
    # print(grids_x3)

    # i,j : left and right, 1,2,3: 1st and 2nd freedom
    for i1 in tqdm(range(n_po)):
        for i2 in range(n_po):
            for i3 in range(n_po):
                for j1 in range(n_po):
                    for j2 in range(n_po):
                        for j3 in range(n_po):
                            i = idx_3_to_1(i1, i2, i3, n_po)
                            j = idx_3_to_1(j1, j2, j3, n_po)
                            h[i,j] = h_1d1[i1,j1]* delta(i2,j2) * delta(i3,j3)\
                                     + h_1d2[i2,j2] * delta(i1,j1) * delta(i3, j3)\
                                     + h_1d3[i3, j3]* delta(i1,j1) * delta(i2,j2)
                            if i == j:
                                h[i, j] +=(\
                                            v([grids_x1[i1], grids_x2[i2], grids_x3[i3]])[0]\
                                             - pot1(grids_x1[i1])\
                                             - pot2(grids_x2[i2])\
                                             - pot3(grids_x3[i3])\
                                             + delta_v[i])

    levels, wfs = eig(h)
    set_sign_v2_(wfs)
    return levels, wfs



def d3_podvr_v2(v_grids, v_ref):
    M = deepcopy(para.hPodvr)
    delta_v = v_grids - v_ref
    # print(M[0, :5])
    for i, j, k  in itertools.product(range(5), range(5), range(5)):
        I = idx_3_to_1(i, j, k, ev.n_po)
        M[I, I] += delta_v[I]

    levels, wfs = eig(M)
    set_sign_v2_(wfs)
    return levels, wfs

def d2_test():
    levels_dvr, levels, wfs = d2_podvr(pot_1d)
    print(levels_dvr[:5])
    print(levels[:5])
    print(wfs[:5, :5])
    
def podvr_test():
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pot_1d)
    print(levels_dvr)
    print(wfs)
    
#podvr_test()
#d2_test()

def mk_1d_dat_harm():
# podvr
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pot_1d_x)
    np.savetxt("cc_podvr.log", wfs)
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pot_1d_y)
    np.savetxt("ch_podvr.log", wfs)
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pot_1d_z)
    np.savetxt("cn_podvr.log", wfs)
    print(levels_dvr[:5]- levels_dvr[0])

# dvr
    levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(pot_1d_x)
    np.savetxt("cc_dvr.log", wfs_dvr)
    levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(pot_1d_y)
    np.savetxt("ch_dvr.log", wfs_dvr)
    levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(pot_1d_z)
    np.savetxt("cn_dvr.log", wfs_dvr)
    trans = levels_dvr[:5] -levels_dvr[0]
    print(trans/trans[1])

    levels_dvr, levels, wfs= d2_podvr(pot_1d_x, pot_1d_y)
    
    print(levels_dvr)
    print(levels)
    #wfs[:, 0] = -1.0 * np.abs(wfs[:,0])
    np.savetxt("d2.log", wfs)

def mk_1d_dat():
# podvr
    print("mk_1d_dat")
    # levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(ev.v_cc)
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pes.spl_cc)
    np.savetxt("cc_podvr.log", wfs)
    print((levels_dvr[:5]- levels_dvr[0]) * ju.getUnitFactor("a.u.", "1/cm"))
    print(grids_x* ju.getUnitFactor("a.u.", "Angstrom"))
    # levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(ev.v_ch)
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pes.spl_ch3)
    np.savetxt("ch_podvr.log", wfs)
    print((levels_dvr[:5]- levels_dvr[0]) * ju.getUnitFactor("a.u.", "1/cm"))
    print(grids_x* ju.getUnitFactor("a.u.", "Angstrom"))
    # levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(ev.v_cn)
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pes.spl_cn)
    np.savetxt("cn_podvr.log", wfs)
    print((levels_dvr[:5]- levels_dvr[0]) * ju.getUnitFactor("a.u.", "1/cm"))
    print(grids_x* ju.getUnitFactor("a.u.", "Angstrom"))

# dvr
    # levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(ev.v_cc)
    # np.savetxt("cc_dvr.log", wfs_dvr)
    # levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(ev.v_ch)
    # np.savetxt("ch_dvr.log", wfs_dvr)
    # levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(ev.v_cn)
    # np.savetxt("cn_dvr.log", wfs_dvr)

    # levels_dvr, levels, wfs= d2_podvr(pot_1d_x, pot_1d_y)
    # print(levels_dvr)
    # print(levels)
    # #wfs[:, 0] = -1.0 * np.abs(wfs[:,0])
    # np.savetxt("d2.log", wfs)

def mk_3d_sincdvr_dat():
    print("mk_3d_sincdvr_dat")
    # levels, wfs= d3_podvr(ev.v_cc, ev.v_ch, ev.v_cn, ev.v_d3, v_grids, ev.v_ref)
    levels, wfs = d3_sincdvr(pes.spl_cc, pes.spl_ch3, pes.spl_cn, \
                           pes.surf_d3_rbf_au, pes.surf_zero, pes.surf_zero)
    trans = levels - levels[0]
    trans *= ev.au2cm

    print("use V_ab_init:")
    print(trans[:8])
    # np.savetxt("d3.log", wfs)

def mk_3d_dat(v_grids = 0):
    print("mk_3d_dat")
    # levels, wfs= d3_podvr(ev.v_cc, ev.v_ch, ev.v_cn, ev.v_d3, v_grids, ev.v_ref)
    levels, wfs = d3_podvr_surf(pes.spl_cc, pes.spl_ch3, pes.spl_cn, \
                           pes.surf_d3_rbf_au, pes.surf_d3_trj_em_g15, pes.surf_d3_ref_em_g15)
    trans = levels - levels[0]
    trans *= ev.au2cm

    print("use V_ab_init:")
    print(trans[:8])
    np.savetxt("d3.log", wfs)
    return 0

def get_eig_dat(surf_trj_em = pes.surf_d3_trj_em_g15, surf_ref_em = pes.surf_d3_ref_em_g15,\
                surf_ref_abin = pes.surf_d3_rbf_au, \
                spl0 = pes.spl_cc, spl1 = pes.spl_ch3, spl2 = pes.spl_cn):
    # levels, wfs = d3_podvr_surf(pes.spl_cc, pes.spl_ch3, pes.spl_cn, \
                           # pes.surf_d3_rbf_au, surf_trj_em, surf_ref_em)
    levels, wfs = d3_podvr_surf(spl0, spl1, spl2, \
                           surf_ref_abin, surf_trj_em, surf_ref_em)
    # trans = levels - levels[0]
    # trans *= ev.au2cm
    return levels, wfs

def get_grids():
    levels_dvr, g0, h_podvr, wfs = sinc_podvr_1d(pes.spl_cc)
    levels_dvr, g1, h_podvr, wfs = sinc_podvr_1d(pes.spl_ch3)
    levels_dvr, g2, h_podvr, wfs = sinc_podvr_1d(pes.spl_cn)
    return g0, g1, g2

def get_trans(levels):
    return (levels - levels[0]) * ju.getUnitFactor("a.u.","1/cm")

def get_all_dat(pot_file ="/home/jtyang/anSolution/md/nvt7ns/resxcluster/energy_test/3d/e_stru1.csv",\
    fout= "tmp.dat", aim_trans = 4):
    # levels_ref, wfs_ref = get_eig_dat( pes.surf_zero, pes.surf_zero )

    # gs.wfs0 = wfs_ref
    # gs.e0 = levels_ref
    # # gs.level_cut = 12
    # gs.level_cut = len(gs.e0)

    surf_trj_em = pes.surf_d3_trj_em_g15
    surf_ref_em = pes.surf_d3_ref_em_g15
    surf_ref_abin = pes.surf_d3_rbf_au

    # eig
    surf_d3 = pes.mk_surf_3d_by_file(pot_file)
    spl0, spl1, spl2 = mt.surf_to_3_spl(surf_ref_abin)
    # levels_eig, wfs_eig = get_eig_dat( surf_trj_em = surf_trj_em, surf_ref_em = surf_ref_em, \
                                        # surf_ref_abin = pes.surf_zero, spl0 = spl0, spl1 = spl1, spl2 = spl2)
    levels_eig, wfs_eig = get_eig_dat( surf_trj_em = surf_trj_em, surf_ref_em = surf_ref_em, \
                                        surf_ref_abin = surf_ref_abin, spl0 = spl0, spl1 = spl1, spl2 = spl2)

    # rspt
    # g0, g1, g2 = get_grids()
    # levels_rspt = gs.get_3d_levels_v3(surf_trj_em, surf_ref_em, set_order= "all", g0 = g0, g1= g1, g2= g2)
    levels_rspt = []
    
    levels_rspt.append(levels_eig)

    for levels in levels_rspt:
        print(get_trans(levels)[:10])

def mk_dat():
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pot_1d_x)
    np.savetxt("cc_podvr.log", wfs)
    levels_dvr, grids_x, h_podvr, wfs = sinc_podvr_1d(pot_1d_y)
    np.savetxt("ch_podvr.log", wfs)
    np.savetxt("cn_podvr.log", wfs)
    levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(pot_1d_x)
    #set_sign_(wfs_dvr, 2)
    np.savetxt("cc_dvr.log", wfs_dvr)
    levels_dvr, wfs_dvr, sinc_grids = sinc_dvr_1d(pot_1d_y)
    np.savetxt("ch_dvr.log", wfs_dvr)
    np.savetxt("cn_dvr.log", wfs_dvr)
    levels_dvr, levels, wfs= d2_podvr(pot_1d_x, pot_1d_y)
    
    print(levels_dvr)
    print(levels)
    #wfs[:, 0] = -1.0 * np.abs(wfs[:,0])
    np.savetxt("d2.log", wfs)

def d3_test():
    # levels, wfs= d3_podvr(ev.v_cc, ev.v_ch, ev.v_cn, ev.v_d3)
    # trans = levels - levels[0]
    # trans *= ev.au2cm

    # print("use V_ab_init:")
    # print(trans[:8])
    # np.savetxt("d3.log", wfs)


    # a = np.zeros(125)
    # levels, wfs= d3_podvr_v2(a, a)
    # trans = levels - levels[0]
    # trans *= ev.au2cm

    # print("use H_podvr:")
    # print(trans[:8])
    # np.savetxt("d3.log", wfs)
    pass

def main():
#d2_test()
#podvr_test()
    # mk_dat()
    # d3_test()
    # ev.n_po = 10
    # mk_1d_dat()
    # n_po_list = [ 5, 7, 10, 20]
    # for n_po in n_po_list:
        # ev.n_po = n_po
        # # mk_1d_dat()
        # mk_3d_dat()
    # ev.n_po = 5
    # mk_3d_dat()
    # ev.n_sinc_dvr = 31
    # mk_3d_sincdvr_dat()
    # mk_1d_dat_harm()
    # n_po_list = [ 5, 7, 10, 20]
    n_po_list = [ 5, 7, 10, 15, 18]
    # n_po_list = [ 5]
    for n_po in n_po_list:
        ev.n_po = n_po
        print(f"n_po : {n_po}")
        get_all_dat()
    pass

if __name__ == "__main__":
    main()
