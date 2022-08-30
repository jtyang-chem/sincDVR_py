#!/share/apps/anaconda3/bin/python
import copy
import ctypes 
import time
import re
import os
import numpy as np
from numba import jit
from numba import prange
from scipy.linalg import eigh
import scipy.linalg as sp
import itertools
from copy import copy
from tqdm import tqdm
# -
import env_vars as ev
import get_shift.para as para
import junit as ju
import m_tool as mt

global wfs0
global e0
global level_cut

def init_wf_and_e():
    # get unperturbated wave functions and levels from eigensolve
    global wfs0
    global e0
    global level_cut
    M = para.hPodvr
    vals, vecs = sp.eig(M)
    idx = vals.argsort() #[::-1]
    e0 = np.real(vals[idx])
    wfs0 = np.real(vecs[:,idx])
    # level_cut = len(e0)//2
    level_cut = 6 * 2

def get_index_3d(i, j, k, n_dim = 5):
    result = i* n_dim * n_dim+ j* n_dim+ k
    return result

def main2():
    #m = [[3,-2,4,-2],[5,3,-3,-2],[5,-2,2,-2],[5,-2,-3,3]]
    #M = np.array([[3,-2,4,-2],[5,3,-3,-2],[5,-2,2,-2],[5,-2,-3,3]])
    n = 700
    #M = np.random.randint(0, 10, n*n).reshape(n, n)
    M = m.hPodvr
    print(M)
    start = time.perf_counter()
    time.sleep(1.1)
    end = time.perf_counter()
    print("time sleep:", end-start)

    # values and vectors
    #start = time.perf_counter()
    #c, b = eigh(M)
    ##a,b=np.linalg.eig(M)
    #end = time.perf_counter()
    ##print("eigen values:")
    ##print(a)
    ##print("eigen vectors:")
    ##print(b)
    #print("time cost:", end-start)

    
    # values only
    start = time.perf_counter()
    #a, b = eigh(M)
    #a, b = eigh(M)
    #b,c = eigh(M)
    a = sp.eigvals(M)
    a = np.sort(np.real(a))
    end = time.perf_counter()
    print("time cost:", end-start)
    print("eigen values:")

    #b = (b-b[0]) * ju.getUnitFactor("a.u.","1/cm" )
    # a = (a-a[0]) * ju.getUnitFactor("a.u.","1/cm" )
    # print(a)
    #print(b)
    print("eigen vectors:")
    #print(b)

def test_get_3d_trans():
    v_grids = np.zeros(125)
    v_ref = np.zeros(5)
    start = time.perf_counter()
    for i in range(5):
        get_3d_trans_eig(v_grids, v_ref, v_ref, v_ref)
    end = time.perf_counter()
    print("time cost:", (end-start)/5)
    trans = get_3d_trans_eig(v_grids, v_ref, v_ref, v_ref)
    print("trans:")
    print(trans[:5])

def get_3d_trans_eig_v2(v_grids, v_ref):
    M = copy.deepcopy(para.hPodvr)
    delta_v = v_grids - v_ref
    # print(M[0, :5])
    for i, j, k  in itertools.product(range(5), range(5), range(5)):
        I = get_index_3d(i, j, k)
        M[I, I] += delta_v[I]
    result = sp.eigvals(M)
    result = np.sort(np.real(result))
    result = (result - result[0])* ju.getUnitFactor("a.u.","1/cm" )

    return result

def get_3d_projs_eig(v_grids, v_ref):
    global wfs0
    global e0
    nstates = len(e0)
    M = copy.deepcopy(para.hPodvr)
    delta_v = v_grids - v_ref
    # print(M[0, :5])
    for i, j, k  in itertools.product(range(5), range(5), range(5)):
        I = get_index_3d(i, j, k)
        M[I, I] += delta_v[I]

    vals, vecs = sp.eig(M)
    idx = vals.argsort()
    wfs = np.real(vecs[:, idx])
    result = np.zeros(nstates)
    proj_list = [4, 5]
    for i in proj_list:
        result[i] = abs(np.dot(wfs[:, i], wfs0[:, i]))

    # print(result[4: 6])

    return result

def get_3d_projs_eig_v3(v_grids, v_ref):
    global wfs0
    global e0
    nstates = len(e0)
    M = copy.deepcopy(para.hPodvr)
    delta_v = v_grids - v_ref
    # print(M[0, :5])
    for i, j, k  in itertools.product(range(5), range(5), range(5)):
        I = get_index_3d(i, j, k)
        M[I, I] += delta_v[I]

    vals, vecs = sp.eig(M)
    idx = vals.argsort()
    wfs = np.real(vecs[:, idx])
    result = np.zeros(nstates)
    # proj_list = [4, 5]
    highest_level = 64
    level_cb = 5
    res_cb = np.zeros(highest_level+1)

    for i in range(highest_level+1):
        res_cb[i] = abs(np.dot(wfs[:, level_cb], wfs0[:, i]))

    # print(res_cn, res_cb)
# print trans for check
    # levels = np.real(vals[idx])
    # trans = (levels - levels[0])* ju.getUnitFactor("a.u.", "1/cm")
    # print(trans[1:6])

# check if normalized
    projs = np.zeros(125)
    v_sum = 0.0
    for i in range(125):
        proj = abs(np.dot(wfs[:, level_cb], wfs0[:, i]))
        v_sum += (proj* proj)
        projs[i]= proj

    print(v_sum, np.argmax(projs))

    return res_cb

def get_3d_projs_eig_v2(v_grids, v_ref):
    global wfs0
    global e0
    nstates = len(e0)
    M = copy.deepcopy(para.hPodvr)
    delta_v = v_grids - v_ref
    # print(M[0, :5])
    for i, j, k  in itertools.product(range(5), range(5), range(5)):
        I = get_index_3d(i, j, k)
        M[I, I] += delta_v[I]

    vals, vecs = sp.eig(M)
    idx = vals.argsort()
    wfs = np.real(vecs[:, idx])
    result = np.zeros(nstates)
    # proj_list = [4, 5]
    level_cn = 4
    level_cb = 5
    res_cn = np.zeros(level_cn+1)
    res_cb = np.zeros(level_cb+1)

    for i in range(level_cn+1):
        res_cn[i] = abs(np.dot(wfs[:, i], wfs0[:, level_cn]))

    for i in range(level_cb+1):
        res_cb[i] = abs(np.dot(wfs[:, i], wfs0[:, level_cb]))

    # print(res_cn, res_cb)

    return res_cn, res_cb

def get_3d_trans_eig(v_grids, v_ref1, v_ref2, v_ref3):
    global wfs0

    M = copy.deepcopy(para.hPodvr)
    # print(M[0, :5])
    for i, j, k  in itertools.product(range(5), range(5), range(5)):
        I = get_index_3d(i, j, k)
        M[I, I] += ( v_grids[I] - v_ref1[i] - v_ref2[j] - v_ref3[k] )
    result = sp.eigvals(M)
    result = np.sort(np.real(result))
    result = (result - result[0])* ju.getUnitFactor("a.u.","1/cm" )

    return result

def dirac_int( left, op, right ):
# all input are numpy 1d vector
    result = sum(left*op*right)
    return result

def get_3d_trans_p2(v_grids, v_ref):
# !!! WRONG function !!!
# because unset values of diagonal of v_mtx, all eps1 are 0,
# there will be NO eps1 part in results

# get 1st order perturbation of trans level
    global wfs0
    global e0

    nstates = len(e0)
    v_mtx = np.zeros((nstates,nstates))
    e1 = np.zeros(nstates)
    e2 = np.zeros(nstates)
    delta_v_grids = v_grids -v_ref
    delta_v_grids = delta_v_grids - delta_v_grids[0]

    for i in range(nstates):
# loop index different from yz version
        for j in range(i):
            v_mtx[i,j] = dirac_int(wfs0[:, i], delta_v_grids, wfs0[:, j])
            v_mtx[j,i] = v_mtx[i, j]

    for i in range(nstates):
        eps_1 = v_mtx[i, i]
        e1[i] = e0[i] + eps_1

        term1 = 0.
        for j in range(nstates):
            if j != i:
                term1 += v_mtx[i, j] * v_mtx[i, j]/ ( e0[j]- e0[i] )
        eps_2 = -term1
        e2[i] = e1[i] + eps_2

    result = (e2 - e2[0])* ju.getUnitFactor("a.u.","1/cm" )
    return result
    
def get_3d_trans_p1(v_grids, v_ref):
# get 1st order perturbation of trans level
    global wfs0
    global e0

    nstates = len(e0)
    v_mtx = np.zeros((nstates,nstates))
    e1 = np.zeros(nstates)
    delta_v_grids = v_grids -v_ref
    delta_v_grids = delta_v_grids - delta_v_grids[0]
    for i in range(nstates):
        # v_mtx(i,i)=sum(wfs0(:,i)*wfs0(:,i)*v_grids)
        v_mtx[i,i] = dirac_int(wfs0[:, i], delta_v_grids, wfs0[:, i])
        eps_1 = v_mtx[i, i]
        e1[i] = e0[i] + eps_1

    result = (e1 - e1[0])* ju.getUnitFactor("a.u.","1/cm" )
    return result

def get_eps1(v_mtx):
    # 1st order perturbation, return eps1
    global level_cut
    return  v_mtx.diagonal()[:level_cut]

def get_eps2(v_mtx):
    # 2nd order perturbation, return eps2
    global e0
    global level_cut
    # nstates = len(e0)
    nstates = level_cut
    eps2 = np.zeros(nstates)

    for i in range(nstates):
        for j in range(nstates):
            if j != i :
                eps2[i] += (v_mtx[i,j] * v_mtx[i,j]/(e0[j]- e0[i]))
    eps2 *=  -1.0
    return eps2

def get_eps3(v_mtx):
    # 3rd order perturbation, return eps3
    global e0
    global level_cut
    # nstates = len(e0)
    nstates = level_cut
    eps3 = np.zeros(nstates)

    for i in range(nstates):
        term1 = .0
        for j in range(nstates):
            if j != i:
                for k in range(nstates):
                    if k!=i:
                        term1 += (
                              v_mtx[i, k] * v_mtx[k, j] * v_mtx[ j, i ]
                              /(
                                  (e0[k]- e0[i])
                                  *(e0[j]- e0[i])
                              )
                            )
        term2 = .0
        for k in range(nstates):
            if k!=i:
                term2 +=(
                     v_mtx[i,i] * v_mtx[i,k] * v_mtx[i,k]
                     / ((e0[k]- e0[i])**2)
                    )

        eps3[i] = term1 - term2
    return eps3

# @jit()
def get_eps4(v_mtx):
    # 4th order perturbation, return eps4
    global e0
    global level_cut
    # nstates = len(e0)
    nstates = level_cut
    eps4 = np.zeros(nstates)

    for i in range(nstates):
        term1 = .0
        # print("calc term1")
        # for j,k,l in tqdm(itertools.product(range(nstates), repeat= 3)):
        # for j,k,l in itertools.product(range(nstates), repeat= 3):
        for j in range(nstates):
            for k in range(nstates):
                for l in range(nstates):
                    if not i in [j,k,l]: # j,k,l not equal i
                        term1 += (
                                v_mtx[i,k] * v_mtx[k,j] * v_mtx[j,l] * v_mtx[l,i]
                                /(
                                    (e0[k]-e0[i]) * (e0[j]- e0[i]) * (e0[l]- e0[i])
                                 )
                            )
        term1 *= -1.0

        term2 =.0
        # print("calc term2")
        # for j,k in tqdm(itertools.product(range(nstates), repeat = 2)):
        for j in range(nstates):
            for k in range(nstates):
                if not i in [j,k]:
                    term2 += v_mtx[i,j]*v_mtx[i,j] * v_mtx[i,k]*v_mtx[i,k]/(
                            (e0[j]- e0[i])* 
                            ( (e0[k]-e0[i])*(e0[k]- e0[i]) )
                        )

        term3 =.0
        # print("calc term3")
        # for k in tqdm(range(nstates)):
        for k in range(nstates):
            if k!= i :
                term3+= v_mtx[i,i]*v_mtx[i,i] * v_mtx[k,i]*v_mtx[k,i]/(
                    (e0[k]-e0[i])*(e0[k]-e0[i])*(e0[k]-e0[i])
                    )
        term3 *= -1.0

        term4 = .0
        # print("calc term4")
        # for j,k in tqdm(itertools.product(range(nstates), repeat = 2)):
        for j in range(nstates):
            for k in range(nstates):
        # for j,k in itertools.product(range(nstates), repeat = 2):
                if not i in [j,k]:
                    term4 += v_mtx[i,i] * v_mtx[i,k] * v_mtx[k,j] * v_mtx[j,i]/(
                        (e0[k]-e0[i]) * (e0[j]-e0[i])
                        )*(
                        1.0/(e0[k]- e0[i]) + 1.0/(e0[j]- e0[i])
                          )

        eps4[i] += (term1 + term2 + term3 + term4)
        # print(f"{term1} {term2} {term3} {term4}")
    return eps4

def mk_delta_v_grids(surf_trj, surf_ref, g0, g1, g2):
    N = len(g0)
    res = np.zeros(N**3)
    for I in range(N**3):
        i,j,k =  mt.idx_1_to_3(I, N)
        posi = [g0[i], g1[j], g2[k]]
        res[I] = surf_trj(posi) - surf_ref(posi)
    return res
        

def get_3d_levels_v3(v_grids, v_ref, set_order= 1, g0 = ev.grid_cc_5, g1 = ev.grid_ch3_5, g2 = ev.grid_cn_5):
        # ! Yu Zhai <yuzhai@mail.huiligroup.org>
        # ! Rayleigh-Schrodinger perturbation theory
        # ! up to the 4th order
        # ! Ref.
        # ! Simple Theorems, Proofs, and Derivations in Quantum Chemistry
        # ! Appendix VIII
# trans to python by jtyang
    global wfs0
    global e0
    global level_cut

    

    nstates = len(e0)
    v_mtx = np.zeros((nstates,nstates))
    e1 = np.zeros(nstates)
    # v_tmp = v_grids -v_ref
    # delta_v_grids = v_tmp - v_tmp[0]
    # delta_v_grids = v_tmp
    delta_v_grids =  mk_delta_v_grids( v_grids, v_ref, g0, g1, g2 )

# construct perturbation matrix
    for i in range(nstates):
        for j in range(i, nstates):
            v_mtx[i,j] = dirac_int(wfs0[:, i], delta_v_grids, wfs0[:, j])
            v_mtx[j,i] = v_mtx[i, j]

    # origin_v_mtx =  copy(v_mtx)
    # origin_e0 =  copy(e0)

    func_list = [ get_eps1, get_eps2, get_eps3, get_eps4 ]


    if isinstance(set_order, int):
        delta_e = np.zeros(nstates)
        for f in tqdm(func_list[:set_order]):
            delta_e += f(v_mtx)
        res = e0 +  delta_e

        # print(np.array_equal(, v_mtx))
        # print(np.array_equal(e0,e0))
        return res

    elif set_order == "all":
        res = [e0[:level_cut]]
        tmp = copy(e0[:level_cut])
        # for f in tqdm(func_list):
        for f in func_list:
            tmp +=  f(v_mtx)

            res.append(copy(tmp))

        return res

    else:
        print("get_3d_trans_v3: wrong set_order value!")
        exit()

def main():
    test_get_3d_trans()

init_wf_and_e()

if __name__ == "__main__":
    main()
