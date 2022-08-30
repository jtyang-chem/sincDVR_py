#!/share/apps/anaconda3/bin/python
import ctypes 
import time
import re
import os
import numpy as np
from scipy.linalg import  eigh
import scipy.linalg as sp
import m
import junit as ju

def main():
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
    start = time.perf_counter()
    c, b = eigh(M)
    #a,b=np.linalg.eig(M)
    end = time.perf_counter()
    #print("eigen values:")
    #print(a)
    #print("eigen vectors:")
    #print(b)
    print("time cost:", end-start)

    
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
    a = (a-a[0]) * ju.getUnitFactor("a.u.","1/cm" )
    print(a)
    #print(b)
    print("eigen vectors:")
    #print(b)

if __name__ == "__main__":
    main()
