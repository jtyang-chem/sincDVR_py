import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from PIL import Image as img
from copy import copy
from os.path import exists
import os
import numpy as np
import math
from tqdm import tqdm
# -
import env_vars as ev
import m_tool as mt
import dvr_test as dvr
import pes

def plot_2d_func_contour(f, fout = "2d.png", num = 10, x_title= "", y_title = "", a = ev.a*.8, b = ev.b*.8):
    x = [e for e in np.linspace(a, b, num)]
    y = x
    fig, ax = plt.subplots(constrained_layout=True)

    dat = np.zeros((num, num))
    for i_x, p_x in tqdm(enumerate(x)):
        for i_y, p_y in enumerate(y):
            dat[i_y, i_x]= f(p_x, p_y)

    # ax.contourf(x, y, dat, 10, cmap=plt.cm.bone, origin=origin)
    ax.contourf(x, y, dat, 20)
    CS = ax.contour(x,y,dat, 20, colors='k')
    ax.clabel(CS, inline=True, fontsize=10)

# label control
    if x_title != "":
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

    # plt.show()
    fig.savefig(fout)
    #fig.savefig(fout)
    #plt.show()

def check_pes_on_d(fin, dim= 0):
    pes = np.loadtxt(fin)
    N = round(pow(len(pes), 1/3))
    x = np.arange(N, dtype= int)
    y = np.zeros(N)
    ijk = np.zeros((N, 3), dtype=int)
    ijk.fill(2)
    ijk[:, dim] = x
    for I in range(N):
        I_pes = mt.idx_3_to_1(ijk[I, 0], ijk[I, 1], ijk[I, 2], N)
        y[I] = pes[I_pes] * ev.au2cm
    y = y- np.min(y)
    # plot
    fig, ax = plt.subplots(1)
    ax.plot(x, y)
    plt.show()

def get_wfs(pot_file):
    surf_trj_em = pes.surf_d3_trj_em_g15
    surf_ref_em = pes.surf_d3_ref_em_g15
    surf_ref_abin = pes.surf_d3_rbf_au

    # eig
    surf_d3 = pes.mk_surf_3d_by_file(pot_file)
    spl0, spl1, spl2 = mt.surf_to_3_spl(surf_ref_abin)
    print("ev.n_po", ev.n_po)
    levels_eig, wfs_eig = dvr.get_eig_dat( surf_trj_em = surf_trj_em, surf_ref_em = surf_ref_em, \
                                        surf_ref_abin = surf_ref_abin, spl0 = spl0, spl1 = spl1, spl2 = spl2)

    print(dvr.get_trans(levels_eig[:8]))
    return wfs_eig

def plot_3d_pes_to_2d_directly(prefix, pes):
    #plot use points directly
    n = round(pes.shape[0]**(1.0/3))
    d = np.zeros((n,n))

    #plot
    for i in range(n):
        for j in range(n):
# make 3rd dimesion at mid-point, so odd n perfered in this function
            k = n//2+1
            I = mt.idx_3_to_1(i,j,k-1,n)
            print(i,j,k, I, n)
            d[i,j] = pes[I,3]
    fig, ax = plt.subplots(1)
    ax.contourf(d)
    fig.savefig(f"{prefix}_d_a.png")

def plot_3d_states_local2(prefix, wf):
    # make surf with wf
    N = len(wf)
    print(N)
    wf_loc = wf/np.linalg.norm(wf)
    n = round(N**(1./3))
    print(f"dimension: {N}, {n}")
    pes_loc = np.zeros((N, 4))
    # for i in range(n):
        # for j in range(n):
            # for k in range(n):
                # I = mt.idx_3_to_1(i,j,k, n)
                # pes_loc[I, :] = i,j,k, wf[I]


    for I in range(N):
        pes_loc[I, :3] = mt.idx_1_to_3(I, n)
        pes_loc[I, 3] = wf_loc[I]

    print(pes_loc)

    surf = mt.d3_rbf_interpolation(pes_loc)
    # surf = pes.surf_harm_3d

    # plot
    title_list = ["CC stretch", "CH sym.", "CN stretch"]
    for istate in range(0, 1):
        print("istate", istate)
        f_loc = lambda  b, c: surf(.0, b, c)
        plot_2d_func_contour(f_loc, f"{prefix}_a.png", num = 20, x_title = title_list[1], y_title = title_list[2], a = .0, b = n)

        f_loc = lambda  a, c: surf(a, .0, c)
        plot_2d_func_contour(f_loc, f"{prefix}_b.png", num = 20, x_title = title_list[0], y_title = title_list[2], a = .0, b = n)

        f_loc = lambda  a, b: surf(a, b, .0)
        plot_2d_func_contour(f_loc, f"{prefix}_c.png", num = 20, x_title = title_list[0], y_title = title_list[1], a = .0, b = n)

    # consider of fitting problem, using points do direct contour
    plot_3d_pes_to_2d_directly(prefix, pes_loc)




def plot_3d_states_local(prefix, wfs, states_list):
    # loop over states_list

# wfs[c_i, istate]
    for state in states_list:
        plot_3d_states_local2(prefix, wfs[:, state])
    

def plot_3d_states(pot_file_list, states_list):
    # ev.n_po  = 5
    for i, pot_file in enumerate(pot_file_list):
        wfs = get_wfs(pot_file)
        prefix = "tmp"
        plot_3d_states_local(prefix, wfs, states_list)

def main():
    pot_list = ["/home/jtyang/anSolution/md/nvt7ns/resxcluster/energy_test/3d/e_stru1.csv"]
    ev.n_po = 7
    states_list = [0]
    plot_3d_states(pot_list, states_list)
    pass

if __name__ == "__main__":
    main()

# check_pes_on_d("/home/jtyang/anSolution/md/nvt7ns/resxcluster/energy_test/3d/e_stru2.csv", dim = 0)
# check_pes_on_d("/home/jtyang/anSolution/methodTest/shiftTest/an_wt/3d/mk_ref_3d/15x15x15/e.csv", dim = 0)
