import numpy as np
import pandas as pd
import os
import sys
import junit

# --- imitate c++ version
amau = 1822.887427
au2cm = 2*109737.32
bohr2angs = 0.5291772108
angs2bohr = 1.0/bohr2angs
xmax = 0.95 * angs2bohr
xmin = -xmax
mn = 1.0 * amau

# ---
a = xmin
b = xmax
n_po = 5
n_podvr = n_po
n_sinc_dvr = 201
m = 1.0 * amau


n_scan = 10
# CCSD(T)/f12a Eh
v_d3_file_scan_list = ["../dat/pes/10x10x10/d3_server33.dat"]
v_d3_file_5 = "/home/jtyang/AN/trans_3d_check_22_June_28/3d_pot/5x5x5/d3.dat"
v_d3_file_7 = "/home/jtyang/AN/trans_3d_check_22_June_28/3d_pot/7x7x7/d3.dat"
v_1d_file_cc = "../dat/pes/1d_dat/cc_1d.dat"
v_1d_file_ch3 = "../dat/pes/1d_dat/ch3_1d.dat"
v_1d_file_cn = "../dat/pes/1d_dat/cn_1d.dat"
v_min =  -132.58062407 

# bohr
scan_d0 = np.linspace(-1., 1., 10) * junit.getUnitFactor("Angstrom", "Bohr")
scan_d1 = np.linspace(-1., 1., 10) * junit.getUnitFactor("Angstrom", "Bohr")
scan_d2 = np.linspace(-1., 1., 10) * junit.getUnitFactor("Angstrom", "Bohr")
scan_d0_g15 = np.linspace(-1., 1., 15) * junit.getUnitFactor("Angstrom", "Bohr")
scan_d1_g15 = np.linspace(-1., 1., 15) * junit.getUnitFactor("Angstrom", "Bohr")
scan_d2_g15 = np.linspace(-1., 1., 15) * junit.getUnitFactor("Angstrom", "Bohr")
grid_cc_5 = np.array([ -0.46008990, -0.24565264, -0.05763803, 0.12514921, 0.32161226]) * junit.getUnitFactor("Angstrom", "Bohr")
grid_ch3_5 = np.array([ -0.29781228, -0.13618470, 0.01091585, 0.15897932, 0.32385069]) * junit.getUnitFactor("Angstrom", "Bohr")
grid_cn_5 =  np.array([ -0.28955434, -0.15431105, -0.03554757, 0.08000185, 0.20421986 ]) * junit.getUnitFactor("Angstrom", "Bohr")
grid_cc_7 =  np.array([ -1.15225, -0.772767, -0.447902, -0.145595, 0.14978, 0.452554, 0.788928 ])
grid_ch3_7 =  np.array([  -0.73576, -0.455798, -0.208745, 0.0279747, 0.26598, 0.517097, 0.804919 ])
grid_cn_7 =  np.array([  -0.730771, -0.490232, -0.284509, -0.0930727, 0.093976, 0.285679, 0.498566 ])
# ---

def main():
    pass

if __name__ == "__main__":
    main()
