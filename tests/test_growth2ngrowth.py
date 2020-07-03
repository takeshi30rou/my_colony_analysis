import colony_analysis.growth2ngrowth as g2n
import pandas as pd


def test_get_growth_table():
    growth_talbe = pd.read_csv("tests/hoge2.csv").values.tolist()
    growth_talbe = [['Column', 'Row', 'CONV', 'LTG', 'MGR', 'SPG']] + growth_talbe

    ngrowth_table = pd.read_csv("tests/hoge3.csv").values.tolist()
    ngrowth_table = [['Column', 'Row', 'CONV', 'LTG', 'MGR', 'SPG']] + ngrowth_table

    assert ngrowth_table == g2n.get_ngrowth_table(growth_talbe)
