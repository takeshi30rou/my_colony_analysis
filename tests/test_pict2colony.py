import colony_analysis.pict2colony as p2c
import subprocess
from pathlib import Path
import configparser
import pandas as pd


def _load_img(path_img):
    res = subprocess.run(f'ls -v {path_img}', capture_output=True, shell=True, text=True)
    fnames_img = res.stdout.splitlines()
    fnames_img = [str(Path(path_img) / f) for f in fnames_img]
    return fnames_img


def test_get_colony_table():
    colony_table = pd.read_csv("tests/hoge.csv").values.tolist()
    colony_table = [['fname', 'Time(min)', 'Column', 'Row', 'Area', 'Mass', 'cmass']] + colony_table

    config = configparser.ConfigParser()
    config.read("tests/pict2colony.ini")
    
    assert colony_table == p2c.get_colony_table(config)


def test_load_img():
    path_img = "tests/5478"
    print(p2c.load_img(path_img))
    print(_load_img(path_img))

    assert _load_img(path_img) == p2c.load_img(path_img)
