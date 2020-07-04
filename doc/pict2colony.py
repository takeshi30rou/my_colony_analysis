import argparse
import os

from colony_analysis import imgop

from colony_analysis import pict2colony as p2c


DEBUG = 0
MIN_COLONY_AREA = 20
DIF_INT_COLONY = 10  # required optical intensity

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cut-level', type=int, dest='cut_level', help='Specify thresholding levels. default=2', default=2)
parser.add_argument('-d', '--debug', dest='debug_mode', action='store_true')
parser.add_argument('-i', '--imgpath', help='input img path')
parser.add_argument('-o', '--output', help='output csv path')
parser.add_argument('-f', '--format', type=int, dest='plate_format', help='colony format of plate (96,384,1536). default=1536', default=1536)
parser.add_argument('-m', '--manual-pos', dest='m_pos', help='Specify pixel position of center of Left Top colony and Right Bottom colony. The format is [LT-x,LT-y,RB-x,RB-y]. e.g. 100,100,1500,1800', default='')
parser.add_argument('-r', '--radius', type=int, help='Radius of center colony region. default=8', default=8)
parser.add_argument('-g', '--grid', type=str, dest='grid_cfg', help='specify grid configuration. default=none', default='')
parser.add_argument('-s', '--spot', action='store_true', help='Spot analysis (Use fix colony area of last image). default=False')
args = parser.parse_args()


def load_args():
    if args.debug_mode:
        global DEBUG
        DEBUG = 1
        imgop.DEBUG = 1
        imgop.IMG_OUT_LEVEL = 2

    path_img = args.imgpath

    fnames_img = p2c.load_img(path_img)
    try:
        return fnames_img, args.output
    except:
        return fnames_img


def analyze(fnames_img, label_out):
    os.getcwd()

    fname_img_last = fnames_img[-1]
    fname = fname_img_last

    # colony detection
    img = p2c.load_plate(fname, args.cut_level)
    imgobj = p2c.get_imgobj(img)
    grid = p2c.get_colony_grid(img, imgobj)
    img = p2c.make_light_flatten_gray_img(img, grid)
    colony = p2c.get_colony(img, grid)
    p2c.draw_colony(img, colony, label_out)

    # colony quantification
    growth_packss = []
    n = len(fnames_img)
    fnames_img.reverse()
    for i, fname_img in enumerate(fnames_img):
        print(f'{i + 1}/{n}')
        fname = fname_img
        img = p2c.load_plate(fname, args.cut_level)
        # img = make_light_flatten_gray_img(img, grid)
        img = p2c.get_bg(img, grid, colony)
        growth_packss += [p2c.get_growth(img, grid, colony, args.radius)]
    return growth_packss, grid.poss


def main():
    fnames_img, out_fname = load_args()
    growth_packss, poss = analyze(fnames_img, 'output')
    p2c.output_csv(fnames_img, growth_packss, poss, out_fname)


if __name__ == "__main__":
    main()
