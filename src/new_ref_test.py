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
import dvr_test as dvr

def get_eig_on_ref_1(surf_trj_em, surf_ref = pes.surf_zero):
# get eig on ref surface of stru 1
    levels, wfs = dvr.d3_podvr_surf(pes.spl_cc_ref1, pes.spl_ch3_ref1, pes.spl_cn_ref1, \
                           pes.surf_d3_trj_em_g15, surf_trj_em, v_ref = surf_ref)
    return levels, wfs

def get_all_dat():
    # levels, wfs = get_eig_on_ref_1(pes.surf_zero)
    # print(dvr.get_trans(levels)[:10])
    levels, wfs = get_eig_on_ref_1(pes.surf_d3_trj_em_g15_2, pes.surf_d3_trj_em_g15)
    print(dvr.get_trans(levels)[:10])

def main():
    # n_po_list = [ 5, 7, 10, 20]
    n_po_list = [ 5, 7, 10, 20]
    for n_po in n_po_list:
        ev.n_po = n_po
        print(f"n_po : {n_po}")
        get_all_dat()
    pass

if __name__ == "__main__":
    main()
