import numpy as np
import pandas as pd
import os
import sys
from copy import copy
import itertools
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
# -
import junit
import env_vars as ev
import m_tool as mt
import plot as jplt

# NOTE! : all things in atomic units

def parse_scan_dat(dat, d0, d1, d2):
    # read scan dat from files
    N = len(d0)
    vs = np.loadtxt(dat)
    pes = np.zeros(4)

    for i,j,k in itertools.product(range(N), repeat=3):
        I = mt.idx_3_to_1(i,j,k, N)
        if vs[I] != 0.0 and not np.isnan(vs[I]):
            # add to pes
            pes = np.vstack((pes, np.array([d0[i], d1[j], d2[k], vs[I]])))
    pes = pes[1:, :] # drop zeros
    return pes

# def parse_mode_scan_dat(dat, d0, d1, d2):
    # # read scan dat which along mode from files
    # p = np.zeros(3, n_scan)
    # v = np.loadtxt(dat)
    # res = np.hstack((p,v))
    # return res

def parse_1d_mode_scan_dat(dat, idx_d):
    arr = np.loadtxt(dat)
    n_row = arr.shape[0]
    pes = np.zeros((n_row, 4))
# no bad vaule filter here
    pes[:, idx_d] = arr[:, 0] * junit.getUnitFactor("Angstrom", "Bohr")
    pes[:, 3] = arr[:, 1]
    return pes

def readin_scan():
    pes = np.zeros(4)
    for idx, dat in enumerate(ev.v_d3_file_scan_list):
        tmp_pes = parse_scan_dat(dat, ev.scan_d0, ev.scan_d1, ev.scan_d2)
        pes = np.vstack((pes, tmp_pes))
    pes = pes[1:, :]
    return pes

def readin_mode_scan():
    # past v dat on grid
    pes_5 =  parse_scan_dat(ev.v_d3_file_5, ev.grid_cc_5, ev.grid_ch3_5, ev.grid_cn_5)
    pes_7 =  parse_scan_dat(ev.v_d3_file_7, ev.grid_cc_7, ev.grid_ch3_7, ev.grid_cn_7)
    pes = np.vstack((pes_5, pes_7))
    return pes

def readin_1d_dat():
    cc_pes = parse_1d_mode_scan_dat(ev.v_1d_file_cc, 0)
    ch3_pes = parse_1d_mode_scan_dat(ev.v_1d_file_ch3, 1)
    cn_pes = parse_1d_mode_scan_dat(ev.v_1d_file_cn, 2)
    pes = np.vstack((cc_pes, ch3_pes, cn_pes))
    return pes

def merge_pes_dat():
    pes_scan = readin_scan() 
    pes_mode = readin_mode_scan()
    pes_1d= readin_1d_dat()
    pes = np.vstack((pes_scan, pes_mode, pes_1d))
    # pes = np.vstack((pes_scan, pes_mode))
    # pes[:, :3] = pes[:, :3] * junit.getUnitFactor("Angstrom", "Bohr")

# delete repeat rows
    df =pd.DataFrame( pes, columns= ["x", "y", "z", "w"] ).drop_duplicates()
    pes = df.to_numpy()
    return pes

def save_pes_dat():
    pes = merge_pes_dat()
    fout = "pes.dat"
    np.savetxt(fout, pes)
    print("save_pes_dat over")

def mk_surf_3d():
    pes = merge_pes_dat()
    surf = mt.LinearNDInterpolator(pes[:, :3], pes[:, 3])
    return surf

def mk_surf_3d_by_file(fdat, N= 10):
    pot = np.loadtxt(fdat)
    pes = np.zeros((N**3, 4))
    for I in range(N**3):
        i,j,k = mt.idx_1_to_3(I, N)
        I = mt.idx_3_to_1(i,j,k,N)
        if N == 10:
            pes[I, 0] = ev.scan_d0[i]
            pes[I, 1] = ev.scan_d1[j]
            pes[I, 2] = ev.scan_d2[k]
        if N== 15:
            pes[I, 0] = ev.scan_d0_g15[i]
            pes[I, 1] = ev.scan_d1_g15[j]
            pes[I, 2] = ev.scan_d2_g15[k]
        pes[I, 3] = pot[I]

    surf_loc = mt.d3_rbf_interpolation(pes)
    surf = lambda a: surf_loc(a[0], a[1], a[2])
    return surf

def mk_surf_3d_rbf():
    pes = merge_pes_dat()
    surf = mt.d3_rbf_interpolation(pes)
    return surf

def mk_spl():
    spl_list = []
    dat_list = [ev.v_1d_file_cc, ev.v_1d_file_ch3, ev.v_1d_file_cn]
    for dat in dat_list:
        arr = np.loadtxt(dat)
        y = (arr[:, 1] - np.min(arr[:, 1]) ) #*junit.getUnitFactor("a.u.", "1/cm")
        x = arr[:, 0]* junit.getUnitFactor("Angstrom", "Bohr")
        # spl_list.append( mt.interp1d(dat[:, 0], dat[:, 1], kind= "cubic", fill_value = "extrapolate"))
        # spl = mt.interp1d(arr[:, 0], arr[:, 1], kind= "cubic", fill_value = "extrapolate")
        spl_list.append( mt.interp1d(x, arr[:, 1], kind= "cubic", fill_value = "extrapolate"))
    return spl_list


def surf_d3_cm(arr):
    global surf_d3
    return (surf_d3(arr) - ev.v_min) * junit.getUnitFactor("a.u.", "1/cm")

def surf_d3_rbf_cm(arr):
    global surf_d3_rbf_loc
    res =  surf_d3_rbf_loc(arr[0], arr[1], arr[2]) 
    res =  (res - ev.v_min) * junit.getUnitFactor("a.u.", "1/cm")
    return res

def surf_d3_rbf_au(arr):
    global surf_d3_rbf_loc
    res =  surf_d3_rbf_loc(arr[0], arr[1], arr[2])
    # res =  (res - ev.v_min) * junit.getUnitFactor("a.u.", "1/cm")
    return res

def surf_zero(a):
    return [.0]

def surf_harm_3d(x, y, z):
    ori = ev.n_po/2.0
    return (x-ori)**2 + (y-ori)**2 + (z-ori)**2

def plot_surf_d3():
    global surf_d3
    global surf_d3_rbf_cm
    # plot
    title_list = ["CC stretch", "CH sym.", "CN stretch"]
    # f_loc = lambda  b, c: surf_d3_cm([.0, b, c])
    f_loc = lambda  b, c: surf_d3_rbf_cm([.0, b, c])
    jplt.plot_2d_func_contour(f_loc, f"{title_list[0]}.png", num = 200, x_title = title_list[1], y_title = title_list[2])

    f_loc = lambda  a, c: surf_d3_rbf_cm([a, .0, c])
    jplt.plot_2d_func_contour(f_loc, f"{title_list[1]}.png", num = 200, x_title = title_list[0], y_title = title_list[2])

    f_loc = lambda  a, b: surf_d3_rbf_cm([a, b, .0])
    jplt.plot_2d_func_contour(f_loc, f"{title_list[2]}.png", num = 200, x_title = title_list[0], y_title = title_list[1])


def plot_1d_line():
    # plot_1d_line for test

    pes_scan = readin_scan() 
    pes_mode = readin_mode_scan()
    pes_1d= readin_1d_dat()
    pes_list = [pes_scan, pes_mode, pes_1d]

    for d in range(3):
        fout = f"d1_test_{d}.png"
        fig, ax = plt.subplots(1)
        for pes in pes_list:
            x = []
            y = []
            for idx in range(pes.shape[0]):
                # if other dimension is zero
                if pes[idx, d]**2 == np.sum(pes[idx,:3] *pes[idx,:3]):
                    x.append(pes[idx, d])
                    y.append((pes[idx, 3] - ev.v_min)* junit.getUnitFactor("a.u.", "1/cm"))
            ax.scatter(x, y)
        fig.savefig(fout)

surf_d3 = mk_surf_3d()
spl_cc, spl_ch3, spl_cn = mk_spl()
surf_d3_rbf_loc = mk_surf_3d_rbf()
surf_d3_ref_em = mk_surf_3d_by_file("/home/jtyang/anSolution/methodTest/shiftTest/an_wt/3d/mk_ref_3d/10x10x10/e.csv")
surf_d3_ref_em_g15 = mk_surf_3d_by_file("/home/jtyang/anSolution/methodTest/shiftTest/an_wt/3d/mk_ref_3d/15x15x15/e.csv", 15)
surf_d3_trj_em= mk_surf_3d_by_file("/home/jtyang/anSolution/md/nvt7ns/resxcluster/energy_test/3d/e10x10.csv")
surf_d3_trj_em_g15 = mk_surf_3d_by_file("/home/jtyang/anSolution/md/nvt7ns/resxcluster/energy_test/3d/e_stru1.csv", 15)
surf_d3_trj_em_g15_2 = mk_surf_3d_by_file("/home/jtyang/anSolution/md/nvt7ns/resxcluster/energy_test/3d/e_stru2.csv", 15)
spl_cc_ref1 = mt.surf_to_spl(surf_d3_trj_em_g15, 0)
spl_ch3_ref1 = mt.surf_to_spl(surf_d3_trj_em_g15, 1)
spl_cn_ref1 = mt.surf_to_spl(surf_d3_trj_em_g15, 2)
tot_v_trj = np.load("/home/jtyang/anSolution/methodTest/shiftTest/an_wt/3d/src_show/patched_dat.npy")

def get_v_trj_by_posi(res, frame):
    global tot_v_trj
    ires = res-1
    frames = np.arange(1799999, 1899999+1, 20)
    iframe = np.argwhere(frames == frame)[0,0]
    v_trj = tot_v_trj[iframe, ires]
    return v_trj



def test_devi_rbf():
    global surf_d3_trj_em_g15

    grid_5_trj =  get_v_trj_by_posi(5, 1800039).reshape(5,5,5)
    for I in range(125):
        i,j,k = mt.idx_1_to_3(I, 5)
        posi = [ev.grid_cc_5[i], ev.grid_ch3_5[j], ev.grid_cn_5[k] ]
        v_surf =  surf_d3_trj_em_g15(posi)
        devi = (v_surf -  grid_5_trj[i,j,k]) * junit.getUnitFactor("a.u.", "1/cm")
        print(devi)


def main():
    # print(readin_scan())
    # surf = mk_surf_3d()
    # print(surf([[0.0, 0.0, .0]]))
    # plot_surf_d3()
    # save_pes_dat()
    print(surf_d3_ref_em([0.0, 0.0, .0]))
    test_devi_rbf()
    # plot_1d_line()
    # print(readin_1d_dat())
    pass

if __name__ == "__main__":
    main()
