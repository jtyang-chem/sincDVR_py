#!/share/apps/anaconda3/bin/python
import ctypes
import time
import re
import os
import numpy as np
import pandas as pd
from scipy.linalg import  eigh
import scipy.linalg as sp
#import m as mCommon
import junit as ju
import itertools

#def get_H0_level():
#    #start = time.perf_counter()
#    a = sp.eigvals(mCommon.hPodvr)
#    a = np.sort(np.real(a))
#    #end = time.perf_counter()
#    #print("time cost:", end-start)
#    return a

def get_index_3d(i,j,k, nDim = 5): return i*nDim*nDim + j*nDim + k

# def get_level_3d(vCross, vRef1, vRef2, vRef3, N = 125, unit ="a.u."):
#     start = time.perf_counter()
#     for i, j, k in itertools.product(*itertools.repeat(range(5), 3)):
#         I = get_index_3d(i,j,k)
#         mCommon.hPodvr[I,I] += (( vCross[I]- vRef1[i]-vRef2[j]-vRef3[k])* ju.getUnitFactor(unit, "a.u.") )
# 
#     middle = time.perf_counter()
#     #a = sp.eigvals(mCommon.hPodvr)
#     #a = np.sort(np.real(a))
#     a,b = eigh(mCommon.hPodvr)
#     end = time.perf_counter()
#     print("total time cost:", end-start)
#     print("diag time cost:", end-middle)
#     
#     return a

def get_level_3d_cpp( vCross, v1, v2, v3, N =125, unit = "a.u."):
    # call cpp lib to calculate level
    vCross*= ju.getUnitFactor(unit, "a.u.")
    v1*= ju.getUnitFactor(unit, "a.u.")
    v2*= ju.getUnitFactor(unit, "a.u.")
    v3*= ju.getUnitFactor(unit, "a.u.")
    #print(v1)

    # interface with C++
    level =  np.zeros((N))
    cdll = ctypes.CDLL("./get_shift/libprt.so")
    pV1 =  ctypes.c_void_p(v1.ctypes.data)
    pV2 =  ctypes.c_void_p(v2.ctypes.data)
    pV3 =  ctypes.c_void_p(v3.ctypes.data)
    pVCross =  ctypes.c_void_p(vCross.ctypes.data)
    pLevel =  ctypes.c_void_p(level.ctypes.data)
    cdll.get_level( pVCross, pV1, pV2, pV3,ctypes.c_void_p(level.ctypes.data) )
    return level

def get_level_3d_df(fCross, fRef, unit = "a.u."):
    # get shift dataframe by dataframe file
    # in cpp unit is a.u., for mopac is kJ/mol

    dfCross = pd.read_csv(fCross, index_col =0 ) 
    dfRef = pd.read_csv(fRef, index_col = 0)

    col = dfCross.columns
    nMethods = len(col)
    nLevel = 10
    shiftArr = np.zeros((nLevel,nMethods))
    for i in range(nMethods):
        ref = dfRef.iloc[:,i].to_numpy()
        print('ref:',ref)
        shiftArr[:,i] =  get_level_3d_cpp(dfCross.iloc[:,i].to_numpy(), ref[:5],ref[5:10], ref[10:], N=125, unit = unit)[:10]
        shiftArr[:,i] = (shiftArr[:,i] - shiftArr[0,i]) * ju.getUnitFactor('a.u.','1/cm')

    return pd.DataFrame(shiftArr, columns = col, index = range(nLevel))
    

def main():
    #vLevel = get_H0_level():
    n = 5
    vCross = np.zeros((n*n*n))
    a = np.zeros((n))
    b = np.zeros((n))
    c = np.zeros((n))
    print("start")
    start = time.perf_counter()
    v = get_level_3d_cpp(vCross, a,b,c)
    end = time.perf_counter()
    print("end")
    v = (v-v[0])* ju.getUnitFactor("a.u.","1/cm")
    print(v)
    print("time cost:", end-start)

if __name__ == "__main__":
    main()
