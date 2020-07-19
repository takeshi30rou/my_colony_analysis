import colony_analysis.colony2growth as c2g
import pandas as pd


def test_get_growth_table():
    colony_table = pd.read_csv("tests/pict2colony.csv").values.tolist()
    colony_table = [['fname', 'Time(min)', 'Column', 'Row', 'Area', 'Mass', 'cmass']] + colony_table

    growth_talbe = pd.read_csv("tests/colony2growth.csv", float_precision="high").values.tolist()
    growth_talbe = [['Column', 'Row', 'CONV', 'LTG', 'MGR', 'SPG']] + growth_talbe

    assert growth_talbe == c2g.get_growth_table(colony_table)
