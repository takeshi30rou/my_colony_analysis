
'''
Normalize growth values in a plate

The program execute following normalization (in order)
1) Plate normalization
2) Spatial normalization
3) Row/Col normalization


Reference)
Baryshnikova A, Costanzo M, Kim Y, Ding H, Koh J, Toufighi K, Youn J-Y, Ou J, San Luis B-J, Bandyopadhyay S, Hibbs M, Hess D, Gingras A-C, Bader GD, Troyanskaya OG, Brown GW, Andrews B, Boone C, Myers CL: Quantitative analysis of fitness and genetic interactions in yeast on a genome scale. Nat Meth 2010, 7:1017-1024.
'''
import csv
import numpy as np
import scipy.signal


# parameters
SIZE_MED_FILT = 7  # 7x7
SIZE_AVE_FILT = 10  # 10x10

# Spatial normalization
# for average filter
tmp = np.ones((SIZE_AVE_FILT, SIZE_AVE_FILT))
avgfilt = tmp / tmp.size


# INPUT
# ary: sample array
# refary: reference array for spatial norm.
#
# ary is numpy 2d masked array
# (0 must be masked for log conversion process)
#
# OUTPUT
# nary: same format to ary

def norm_growth(ary, refary):
    aryp = plate_norm(ary)
    aryps = spatial_norm(aryp, refary)
    arypsr = rowcol_norm(aryps)
    nary = arypsr
    return nary


# Plate normalization
# PMM: Plate middle mean
def calc_pmm(vs):
    n = len(vs)
    ns = int(n * 0.2)
    ne = int(n * 0.8)
    vs_sort = np.sort(vs)
    pmm = np.mean(vs_sort[ns:ne])
    return pmm


def plate_norm(ary):
    ary_1d = ary.reshape(-1)
    pmm = calc_pmm(ary_1d)
    cary = ary / pmm
    return cary


def spatial_norm(ary, refary):
    # calc med filt estimetes
    ary_log = np.ma.log(ary / refary)
    # median filter
    ary_m = scipy.signal.medfilt2d(ary_log, SIZE_MED_FILT)
    # average filter
    ary_ma = scipy.signal.convolve(ary_m, avgfilt, "same")
    cary = ary * (1 / np.exp(ary_ma))
    return cary


# Row/Col normalization
def rowcol_norm(ary):
    ary[ary <= 0] = 0
    # Column fit
    col_meds = np.median(ary, axis=0)
    crs_r = np.median(col_meds) / col_meds
    # Row fit
    row_meds = np.median(ary, axis=1)
    rrs_r = np.median(row_meds) / row_meds
    # normalize
    m_rrs_r = rrs_r.reshape(-1, 1)  # vector to matrix
    m_crs_r = crs_r.reshape(-1, 1)  # vector to matrix
    cary = ary * np.dot(m_rrs_r, m_crs_r.T)

    return cary


def load_csv(fname):
    with open(fname, 'r') as f:
        header = next(csv.reader(f))
        data = [list(map(float, items)) for items in csv.reader(f)]
    table = [header] + data
    return load_table(table)


def load_table(table):
    plate_format = [32, 48]
    array_shape = plate_format + [4]

    header = table[0]  # header

    n_table = np.array(table[1:])  # change list to narray

    # get col and row from n_table[0] and n_table[1], respectively
    poss = n_table[:, :2].astype(np.int)
    ary = np.reshape(n_table[:, 2:], array_shape)

    return ary, poss.tolist()


def output_csv(nary, poss, fname):
    w = csv.writer(open(fname, 'w'))
    w.writerow(["Column", "Row", "CONV", "LTG", "MGR", "SPG"])
    for pos in poss:
        col, row = pos
        vs = nary[row - 1, col - 1, :]
        vs = [round(v, 2) for v in vs]
        out = [col, row] + vs
        w.writerow(out)


def get_ngrowth_table(table):
    ary, poss = load_table(table)
    nary = ary.copy()

    for ind in range(4):
        # reference array
        refary = np.ones(ary[:, :, ind].shape)
        nary[:, :, ind] = norm_growth(ary[:, :, ind], refary)

    ngrowth_table = [['Column', 'Row', 'CONV', 'LTG', 'MGR', 'SPG']]
    for pos in poss:
        vs = [round(v, 2) for v in nary[pos[1] - 1, pos[0] - 1, :]]
        ngrowth_table += [[*pos, *vs]]

    return ngrowth_table
