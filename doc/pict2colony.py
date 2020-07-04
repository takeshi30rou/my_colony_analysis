import argparse
import csv
import os

import numpy

from colony_analysis import bgflatten
from colony_analysis import imgop as imgop
from colony_analysis import motsu as motsu

from colony_analysis.pict2colony import PlateGrid
from colony_analysis.pict2colony import ImageObject
from colony_analysis.pict2colony import Colony
from colony_analysis.pict2colony import Growth

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


def get_imgobj(img):
    # get threshold
    if img.oint_bg == 0:
        # threshold = get_threshold_first_peak(img.gray)
        ts = motsu.get_thresholds(img.gray, img.cut_level)
        threshold = ts[-1]
    else:
        threshold = img.oint_bg - DIF_INT_COLONY
    # unsharp mask
    if args.spot:
        imgMask = img.gray
    else:
        imgMask = imgop.unsharp_mask(img.gray)
        imgMask = imgop.unsharp_mask(imgMask)
    # cut with threshold value
    img.binary = imgop.binarize_with_threshold(imgMask, threshold)
    # find objects
    imgobj = ImageObject(img)
    imgobj.find_objects(img)
    return imgobj


def get_colony_grid(img, imgobj):
    grid = PlateGrid(args.plate_format)
    if args.grid_cfg != "":
        grid.load_grid(args.grid_cfg)
    elif args.m_pos != "":
        grid.make_grid_by_manual(args.m_pos)
    else:
        grid.make_grid(imgobj)
    grid.draw_grid_lines(img.color)
    grid.make_crippos()
    return grid


def make_light_flatten_gray_img(img, grid):
    img_crop = imgop.crop(img.gray, grid.xyTL, grid.xyBR, 2)
    # thresholding
    ts = motsu.get_thresholds(img_crop, img.cut_level)
    t_cut = ts[-1]
    # make psuedo plate then correct
    # imgary_crip = imgop.get_array_from_image(img_crop)
    img_pseudo = bgflatten.make_pseudo_plate(img_crop)
    img_correct, med_bg = bgflatten.make_corrected_plate(img_crop, img_pseudo)
    # image output
    imgop.create_img(img_pseudo, "pseudo_empty_plate", 2, 1)
    imgop.create_img(img_correct, "light_corrected_plate", 2, 1)
    # paste crop image to original image
    img_paste = imgop.paste(img.gray, img_correct, grid.xyTL, grid.xyBR)
    img.oint_bg = med_bg
    img.gray = img_paste
    return img


def get_colony(img, grid):
    # light correction
    # img = make_light_flatten_gray_img(img, grid)
    # re-detect the image object
    imgobj = get_imgobj(img)
    # image object -> colony
    colony = Colony(grid)
    imgobj._make_objid_ary()
    colony.find_colony_from_image_objects(img, imgobj, grid)
    colony.make_colony_xys(img.color, imgobj, grid)
    return colony


def draw_colony(img, colony, label_out):
    imgop.draw_edges(img.color, colony.edges_all, label_out, 1)


def get_bg(img, grid, colony):
    ary = img.gray
    rx, ry = grid.xyDif
    # for gpos in grid.poss:
    #     (xc, yc) = colony.gridpos2xy_center[gpos]
    #     if xc == 0 and yc == 0:
    #         continue
    #     ary_colony = colony.gridpos2xys[gpos]
    #     t = ary[int(yc - ry + 1):int(yc + ry), int(xc - rx + 1):int(xc + rx)]
    #     t[ary_colony] = 0
    #     ary[int(yc-ry+1):int(yc+ry), int(xc-rx+1):int(xc+rx)] = t
    ary = ary[int(grid.xyTL[1]):int(grid.xyBR[1]), int(grid.xyTL[0]):int(grid.xyBR[0])]
    mary = numpy.ma.masked_array(ary, ary == 0)
    m = numpy.ma.median(mary)
    m = numpy.ma.mean(mary[numpy.abs(mary - m) < 10])
    img.oint_bg = m
    return img


def get_growth(img, grid, colony, r):
    growth = Growth(grid.poss, grid.xyDif, r)
    growth.get_growth(img, colony)
    growth_packs = []
    for gpos in growth.gridposs:
        area = growth.gridpos2area[gpos]
        mass = growth.gridpos2mass[gpos]
        cmass = growth.gridpos2cmass[gpos]
        growth_packs += [(area, mass, cmass)]
    return growth_packs


def output_csv(fnames_img, growth_packss, poss, fname_out):
    w = csv.writer(open(fname_out, 'w'))
    w.writerow(["fname", "Time(min)", 'Column', 'Row', 'Area', 'Mass', 'cmass'])
    for fname_img, gpacks in zip(fnames_img, growth_packss):
        fname_wopath = fname_img.split("/")[-1]
        t = int((fname_img.split("/")[-1].split(".")[0]).split("-")[1])
        for gpack, pos in zip(gpacks, poss):
            col, row = pos
            area, mass, cmass = gpack
            out = [fname_wopath, t, col, row, area, mass, cmass]
            w.writerow(out)


def analyze(fnames_img, label_out):
    ccwd = os.getcwd()

    fname_img_last = fnames_img[-1]
    fname = fname_img_last

    # colony detection
    img = p2c.load_plate(fname)
    imgobj = get_imgobj(img)
    grid = get_colony_grid(img, imgobj)
    img = make_light_flatten_gray_img(img, grid)
    colony = get_colony(img, grid)
    draw_colony(img, colony, label_out)

    # colony quantification
    growth_packss = []
    n = len(fnames_img)
    fnames_img.reverse()
    for i, fname_img in enumerate(fnames_img):
        print(f'{i + 1}/{n}')
        fname = fname_img
        img = p2c.load_plate(fname)
        # img = make_light_flatten_gray_img(img, grid)
        img = get_bg(img, grid, colony)
        growth_packss += [get_growth(img, grid, colony, args.radius)]
    return growth_packss, grid.poss


def get_colony_table(config):
    conf = config['conf']
    global args
    arg_lst = ['-c', conf['cut-level'],
               '-i', conf['imgpath'],
               '-f', conf['plate_format'],
               '-m', conf['manual-pos'],
               '-r', conf['radius'],
               '-g', conf['grid']]
    if conf.getboolean('debug_mode'):
        arg_lst += '-d'
    if conf.getboolean('spot'):
        arg_lst += '-s'
    args = parser.parse_args(arg_lst)
    fnames_img, out_fname = load_args()
    growth_packss, poss = analyze(fnames_img, 'output')

    colony_table = [['fname', 'Time(min)', 'Column', 'Row', 'Area', 'Mass', 'cmass']]
    for fname_img, gpacks in zip(fnames_img, growth_packss):
        fname_wopath = fname_img.split("/")[-1]
        t = int((fname_img.split("/")[-1].split(".")[0]).split("-")[1])
        for gpack, pos in zip(gpacks, poss):
            colony_table += [[fname_wopath, t, *pos, *gpack]]

    return colony_table


def main():
    fnames_img, out_fname = load_args()
    growth_packss, poss = analyze(fnames_img, 'output')
    output_csv(fnames_img, growth_packss, poss, out_fname)


if __name__ == "__main__":
    main()
