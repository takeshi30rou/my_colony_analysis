'''
Compute three growth parameters from the growth curve
'''
import csv

import numpy as np
import matplotlib.pylab as plt
import rpy2.robjects as robj


# Parameters
CONV_TIME = 20 * 60  # minutes
M_LIM = 1

PLOT = 0


r = robj.r
r('options(warn=-1)')


def model_fitting(xs, ys):
    robj.globalenv['x'] = robj.IntVector(xs)
    robj.globalenv['y'] = robj.FloatVector(ys)
    r('res <- NULL')

    # Logistic
    # r('tryCatch(res <- nls(y~SSlogis(x, asym, b2, b3)), error=function(e) stop)')

    # Gompertz
    r('tryCatch(res <- nls(y~SSgompertz(x, a, b, c)), error=function(e) stop)')
    a, b, c = r('coef(res)')
    mgr = -a * np.log(c) / np.exp(1)
    spg = a
    ltg = (np.log(1 / b) - 1) / np.log(c)

    # round
    ltg = round(ltg, 1)
    mgr = round(mgr, 5)
    spg = round(spg, 3)

    if PLOT:
        r('px <- seq(%d,1200,by=1)' % ltg)
        pxs = np.array(list(r('px')))
        pys = np.array(list(r('predict(res, list(x=px))')))
        plt.plot(xs, ys, 'o')
        plt.plot(pxs, pys, '-k')
        plt.show()
    return (ltg, mgr, spg)


def prep_for_fitting(times, gcs):
    if sum(gcs) == 0:
        return [], []
    ts = np.array(times)
    vs = np.array(gcs)

    vs = vs - np.min(vs)

    vs[vs < M_LIM] = M_LIM
    fts = ts[vs > M_LIM][:30]
    fvs = vs[vs > M_LIM][:30]
    return fts, fvs


# Growth at fixed time point
def get_conv_value(times, vss):
    difs = np.array(times) - CONV_TIME
    ind_min = np.where(abs(difs) == min(abs(difs)))[0]
    convs = [vs[int(ind_min)] for vs in vss]
    return convs


# Growth parameters: LTG, MGR, SPG
def get_growth_params(times, vss):
    n_fail = 0
    gparams = []
    n = len(vss)
    for i, gc in enumerate(vss):
        if i % 100 == 0:
            print("%d/%d" % (i, n))

        fts, fgs = prep_for_fitting(times, gc)

        # skip no growth
        if len(fts) == 0:
            gparam = (0, 0, 0)

        # otherwise, do fitting
        else:
            try:
                gparam = model_fitting(fts, fgs)
                if gparam[2] / fgs[-1] > 1.2:
                    gparam = tuple(list(gparam[:2]) + [round(fgs[-1], 3)])
            except BaseException:
                gparam = (0, 0, 0)
                n_fail += 1
        gparams += [gparam]
    print()
    # print "Fitting failure:%d" % n_fail
    return gparams


def load_csv(fname, vtype='cmass', head=1):
    vtype2ind = {
        'area': 4,
        'mass': 5,
        'cmass': 6
    }
    try:
        ind = vtype2ind[vtype]
    except BaseException:
        quit("no value type")

    timepos2v = {}
    for items in csv.reader(open(fname, 'r')):
        if head:
            head -= 1
            continue
        time = int(items[1])
        pos = tuple(map(int, items[2:4]))
        timepos = (time, pos)
        v = float(items[ind])
        timepos2v[timepos] = v

    times = sorted(set([i[0] for i in timepos2v.keys()]))
    poss = sorted(set([i[1] for i in timepos2v.keys()]),
                  key=lambda x: (x[1], x[0]))

    vss = []
    for pos in poss:
        vs = []
        for time in times:
            timepos = (time, pos)
            vs += [timepos2v[timepos]]
        vss += [vs]
    return times, poss, vss


def load_table(table, vtype='cmass', head=1):
    vtype2ind = {
        'area': 4,
        'mass': 5,
        'cmass': 6
    }
    try:
        ind = vtype2ind[vtype]
    except BaseException:
        quit("no value type")
    timepos2v = {}
    for items in table:
        if head:
            head -= 1
            continue
        time = int(items[1])
        pos = tuple(map(int, items[2:4]))
        timepos = (time, pos)
        v = float(items[ind])
        timepos2v[timepos] = v

    times = sorted(set([i[0] for i in timepos2v.keys()]))
    poss = sorted(set([i[1] for i in timepos2v.keys()]),
                  key=lambda x: (x[1], x[0]))

    vss = []
    for pos in poss:
        vs = []
        for time in times:
            timepos = (time, pos)
            vs += [timepos2v[timepos]]
        vss += [vs]
    return times, poss, vss


def output_as_csv(poss, convs, gparams, fname):
    w = csv.writer(open(fname, 'w'))
    w.writerow(["Column", "Row", "CONV", "LTG", "MGR", "SPG"])
    for pos, conv, gparam in zip(poss, convs, gparams):
        out = list(pos) + [conv] + list(gparam)
        w.writerow(out)


def get_growth_table(table):
    # conventional value
    times, poss, vss = load_table(table, vtype='area')
    convs = get_conv_value(times, vss)
    # growth paramenters
    times, poss, vss = load_table(table, vtype='cmass')
    # times, poss, vss = load_csv(fname_in, vtype='mass')
    gparams = get_growth_params(times, vss)

    table = [['Column', 'Row', 'CONV', 'LTG', 'MGR', 'SPG']]
    for pos, conv, gparam in zip(poss, convs, gparams):
        table += [[*pos, conv, *gparam]]
    return table
