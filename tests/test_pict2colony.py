import colonylive.pict2colony as p2c
import subprocess
from pathlib import Path


def _load_img(path_img):
    res = subprocess.run(f'ls -v {path_img}', capture_output=True, shell=True, text=True)
    fnames_img = res.stdout.splitlines()
    fnames_img = [str(Path(path_img) / f) for f in fnames_img]
    return fnames_img


def test_load_img():
    path_img = "tests/5478"
    print(p2c.load_img(path_img))
    print(_load_img(path_img))

    assert _load_img(path_img) == p2c.load_img(path_img)
