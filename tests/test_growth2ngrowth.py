import colony_analysis.growth2ngrowth as g2n
import pandas as pd
import numpy as np
import csv

growth_talbe = pd.read_csv("tests/hoge2.csv").values.tolist()
growth_talbe = [['Column', 'Row', 'CONV', 'LTG', 'MGR', 'SPG']] + growth_talbe

ngrowth_table = pd.read_csv("tests/hoge3.csv").values.tolist()
ngrowth_table = [['Column', 'Row', 'CONV', 'LTG', 'MGR', 'SPG']] + ngrowth_table

def _load_table(table):
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

def _load_csv(fname):
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

def test_get_growth_table():
    assert ngrowth_table == g2n.get_ngrowth_table(growth_talbe)

def test_load_table():
    expected = _load_table(growth_talbe)
    result = g2n.load_table(growth_talbe)

    np.testing.assert_allclose(expected[0], result[0]) # ary
    assert expected[1] == result[1] # poss

def test_load_csv():
    expected = _load_csv("tests/hoge2.csv")
    result = g2n.load_csv("tests/hoge2.csv")

    np.testing.assert_allclose(expected[0], result[0]) # ary
    assert expected[1] == result[1] # poss