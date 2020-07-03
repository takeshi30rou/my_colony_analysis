'''
Lighting correction module

Reference)
Lawless C, Wilkinson DJ, Young A, Addinall SG, Lydall DA: Colonyzer: automated quantification of micro-organism growth characteristics on solid agar. BMC Bioinformatics 2010, 11:287.
'''
import numpy as np
import matplotlib.pylab as plt
from colony_analysis import imgop


def get_bg_smooth_1d(vary, dp, plot=False):
    vary = np.log(vary)
    pvary = vary.copy()
    step = int(len(vary) / dp)
    for i in range(step + 1):
        fary = vary[i * dp:i * dp + dp]
        tary = np.sort(fary)[-10:-1]
        mean = np.mean(tary)
        pvary[i * dp:i * dp + dp] = mean
    cary = pvary
    cary = np.exp(cary)
    if plot:
        plt.plot(np.exp(vary))
        plt.plot(cary)
        plt.ylim(180, 210)
        plt.show()
    return cary


def get_bg_smooth_2d_x(ary, dp, plot=False):
    cary2d = np.zeros(ary.shape)
    step = int(ary.shape[0] / dp)
    for i in range(step + 1):
        tary = ary[i * dp:i * dp + dp, :]
        ary1d = np.median(tary, axis=0)
        cary1d = get_bg_smooth_1d(ary1d, 100)
        cary2d[i * dp:i * dp + dp, :] = np.array([cary1d] * tary.shape[0])
    return cary2d


def get_bg_smooth_2d_y(ary, dp, plot=False):
    cary2d = np.zeros(ary.shape)
    step = int(ary.shape[1] / dp)
    for i in range(step + 1):
        tary = ary[:, i * dp:i * dp + dp]
        ary1d = np.median(tary, axis=1)
        cary1d = get_bg_smooth_1d(ary1d, 100)
        cary2d[:, i * dp:i * dp + dp] = np.array([cary1d] * tary.shape[1]).T
    return cary2d


def make_pseudo_plate(ary):
    cxary2d = get_bg_smooth_2d_x(ary, 100)
    cyary2d = get_bg_smooth_2d_y(ary, 100)
    cary2d = (cxary2d + cyary2d) / 2
    # mat = imgop.get_mat_from_array(cary2d)
    cmat = imgop.blur(cary2d, 51, 51)
    cmat = imgop.blur(cmat, 101, 101)
    cary2d = np.array(cmat)
    return cary2d


def make_corrected_plate(ary_img, ary_pseudo, plot=False):
    ary_ratio = np.median(ary_pseudo) / ary_pseudo
    cary = ary_img * ary_ratio
    med_bg = np.median(cary)

    if plot:
        plt.plot(np.mean(cary, axis=0))
        plt.plot(np.mean(ary_img, axis=0), '.')
        plt.show()
        plt.plot(np.mean(cary, axis=1))
        plt.plot(np.mean(ary_img, axis=1), '.')
        plt.show()
    return cary, med_bg
