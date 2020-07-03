
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
    ns = n * 0.2
    ne = n * 0.8
    vs_sort = np.sort(vs)
    pmm = np.mean(vs_sort[int(ns):int(ne)])
    return pmm


def plate_norm(ary):
    ary_1d = ary.reshape(-1)
    pmm = calc_pmm(ary_1d)
    cary = ary / pmm
    return cary


# Spatial normalization
# for average filter
tmp = np.ones((SIZE_AVE_FILT, SIZE_AVE_FILT))
avgfilt = tmp / tmp.size


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
    cary = np.zeros(ary.shape)
    for row in range(32):
        for col in range(48):
            cary[row, col] = ary[row, col] * rrs_r[row] * crs_r[col]
    return cary


def load_csv(fname):
    header = 1
    poss = []
    ary = np.zeros((32, 48, 4))
    for items in csv.reader(open(fname, 'r')):
        if header:
            header -= 1
            continue
        col = int(items[0])
        row = int(items[1])
        poss += [[col, row]]
        v = float(items[2])
        for ind in range(4):
            v = float(items[2 + ind])
            ary[row - 1, col - 1, ind] = v
    return ary, poss


def load_table(table):
    header = 1
    poss = []
    ary = np.zeros((32, 48, 4))
    for items in table:
        if header:
            header -= 1
            continue
        col = int(items[0])
        row = int(items[1])
        poss += [[col, row]]
        v = float(items[2])
        for ind in range(4):
            v = float(items[2 + ind])
            ary[row - 1, col - 1, ind] = v
    return ary, poss


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


def main():
    # import optparse
    # parser = optparse.OptionParser(usage="%prog [input growth csv] [output csv]")
    # #parser.add_option('-f', '--flatref', action='store_true', dest='use_flatref', help='Use flat refference array instead of data-based array for spatial normalimzation', default=False)
    # (opts, args) = parser.parse_args()
    # if len(args) != 2:
    #     parser.print_help()
    #     quit()
    # fname_in = args[0]
    # fname_out = args[1]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input CSV path', required=True)
    parser.add_argument('-o', '--output', help='output CSV path', required=True)
    args = parser.parse_args()

    fname_in = args.input
    fname_out = args.output

    ary, poss = load_csv(fname_in)
    nary = ary.copy()

    for ind in range(4):
        # reference array
        refary = np.ones(ary[:, :, ind].shape)
        nary[:, :, ind] = norm_growth(ary[:, :, ind], refary)

    output_csv(nary, poss, fname_out)
    print()


if __name__ == "__main__":
    main()
