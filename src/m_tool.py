import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d

def idx_3_to_1(i1,i2,i3, N=5):
    # direct product loop tool
    res = i1*N*N + i2*N + i3
    return res

def idx_1_to_3(I, N=5):
    # direct product loop tool
    i1 = I//(N*N)
    i2 = (I- i1*N*N)//N
    i3 = I- i1*N*N- i2*N
    return i1, i2, i3

def test_idx13():
    N = 10
    for I in range(10):
        i1, i2, i3 = idx_1_to_3(I, N)
        print(f"{idx_3_to_1(i1,i2,i3,N)},{I}")
    I = N*N*N
    i1, i2, i3 = idx_1_to_3(I, N)
    print(f"{idx_3_to_1(i1,i2,i3,N)},{I}")

def d3_rbf_interpolation(pes):
    # return Rbf(pes[:, 0], pes[:, 1], pes[:, 2], pes[:, 3], function='thin_plate')
    return Rbf(pes[:, 0], pes[:, 1], pes[:, 2], pes[:, 3], function='cubic')

def surf_to_3_spl(surf):
    spl0 = surf_to_spl(surf, 0)
    spl1 = surf_to_spl(surf, 1)
    spl2 = surf_to_spl(surf, 2)
    return spl0, spl1, spl2

def surf_to_spl(surf, dim = 0):
    if dim == 0:
        spl = lambda a: surf([a, .0, .0]) 
    elif dim== 1:
        spl = lambda a: surf([.0, a, .0]) 
    elif dim== 2:
        spl = lambda a: surf([.0, .0, a]) 
    return spl
    
def main():
    # test_idx13()
    # d3_pes_interpolation(0)
    pass

if __name__ == "__main__":
    main()
