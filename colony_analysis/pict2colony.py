import argparse
import configparser
import csv
import os
import subprocess
from pathlib import Path

import cv2
import numpy

from colony_analysis import bgflatten
from colony_analysis import imgop
from colony_analysis import motsu

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


class PlateGrid:
    def __init__(self, plate_format):
        self.xyCenters = [(0, 0)]  # x,y position of colony center
        self.pos2xyCenter = {}
        self.pos2objid = {}
        self.cols, self.rows, self.space_r, self.coef_radius = \
            self._get_cols_rows_space_r(plate_format)
        self.poss = self._get_poss()
        self.xyInit = (0, 0)
        self.xyDif = (0, 0)
        self.xyRadius = (0, 0)
        self.xyTL = (0, 0)
        self.xyBR = (0, 0)

    def _get_cols_rows_space_r(self, plate_format):
        pf = plate_format
        if pf == 6144:
            rows = range(1, 65)
            cols = range(1, 97)
            space_r = 2.0
            coef_radius = 1
        elif pf == 1536:
            rows = range(1, 32 + 1)
            cols = range(1, 48 + 1)
            space_r = 1.5
            coef_radius = 1
        elif plate_format == 384:
            rows = range(1, 16 + 1)
            cols = range(1, 24 + 1)
            space_r = 0.7
            coef_radius = 1
        elif plate_format == 96:
            rows = range(1, 8 + 1)
            cols = range(1, 12 + 1)
            space_r = 0.5
            coef_radius = 1.8
        else:
            print('Wrong plate format')
            quit()
        return cols, rows, space_r, coef_radius

    def _get_poss(self):
        poss = []
        for row in self.rows:
            for col in self.cols:
                pos = (col, row)
                poss.append(pos)
        return poss

    def calc_pos2xyCenter(self, xyInit=0, xyDif=0):
        if xyInit == 0 and xyDif == 0:
            xyInit = self.xyInit
            xyDif = self.xyDif
        for row in self.rows:
            for col in self.cols:
                x = int(xyInit[0] + xyDif[0] * (col - 1))
                y = int(xyInit[1] + xyDif[1] * (row - 1))
                self.pos2xyCenter[col, row] = (x, y)

    def load_grid(self, grid_cfg):
        conf = configparser.RawConfigParser()
        conf.read(grid_cfg)
        xInit = conf.getint('general', 'x_init')
        yInit = conf.getint('general', 'y_init')
        xDif = conf.getfloat('general', 'x_dif')
        yDif = conf.getfloat('general', 'y_dif')
        self.xyInit = (xInit, yInit)
        self.xyDif = (xDif, yDif)
        self.xyRadius = (int(xDif / self.coef_radius), int(yDif / self.coef_radius))
        self.calc_pos2xyCenter()

    def make_grid_by_manual(self, m_pos):
        mpos_list = m_pos.split(',')
        (xLT, yLT, xRB, yRB) = [int(i) for i in mpos_list]
        xInit = xLT
        yInit = yLT
        xDif = (xRB - xLT) / (len(self.cols) - 1)
        yDif = (yRB - yLT) / (len(self.rows) - 1)
        self.xyInit = (xInit, yInit)
        self.xyDif = (xDif, yDif)
        self.xyRadius = (int(xDif / self.coef_radius), int(yDif / self.coef_radius))
        self.calc_pos2xyCenter()

    def make_crippos(self):
        rx = self.xyDif[0]
        ry = self.xyDif[1]
        xStart = int(self.xyInit[0] - rx * self.space_r)
        yStart = int(self.xyInit[1] - ry * self.space_r)
        xEnd = int(self.xyInit[0] + rx * (len(self.cols) - 1 + self.space_r))
        yEnd = int(self.xyInit[1] + ry * (len(self.rows) - 1 + self.space_r))
        self.xyTL = (xStart, yStart)
        self.xyBR = (xEnd, yEnd)

    def make_grid(self, imgobj):
        grid_format = (len(self.cols), len(self.rows))
        vss_colony = self._remove_noise(imgobj.xyCenters, grid_format)
        xyDif = self._find_dif(vss_colony)
        xyInit = self._find_init(vss_colony, xyDif, grid_format)
        self.xyDif = xyDif
        self.xyInit = xyInit
        self.xyRadius = (int(xyDif[0] / self.coef_radius), int(xyDif[1] / self.coef_radius))
        self.calc_pos2xyCenter()

    def _remove_noise(self, xyCenters, grid_format):
        max_dif = 3
        cont_len = 0.6
        vss_colony = []
        for axis in [0, 1]:
            n_min = int(grid_format[abs(axis - 1)] * cont_len)
            vs = [i[axis] for i in xyCenters]
            vs.sort()
            ds = numpy.diff(vs)
            aves = []
            tmp = [vs[0]]
            for ind, d in enumerate(ds):
                if d > max_dif:
                    if len(tmp) > n_min:
                        aves += [numpy.mean(tmp)]
                    tmp = [vs[ind + 1]]
                    continue
                tmp += [vs[ind + 1]]
            if len(tmp) > n_min:
                aves += [numpy.mean(tmp)]
            vss_colony += [aves]
        return vss_colony

    # axis:  0= x axis, 1= y axis
    # max_dif: limit value for diff sequence
    # cont len = ratio for contiuous length
    def _find_dif(self, vss):
        return [numpy.median(numpy.diff(vs)) for vs in vss]
        # for axis in [0,1]:
        #     vs = [i for i in vss[axis]]
        #     vDif = numpy.median(numpy.diff(vs))
        #     xyDif[axis] = vDif
        # return tuple(xyDif)

    def _find_init(self, vss, xyDif, grid_format):
        xyInit = [0, 0]
        for axis in [0, 1]:
            n = grid_format[axis]
            vs = [i for i in vss[axis]]
            if len(vs) != n:
                if len(vs) > n:
                    n_rm = len(vs) - n
                    vs_resi = numpy.array(vs) % xyDif[axis]
                    vs_resi_ave = numpy.mean(vs_resi)
                    ds = list(abs(vs_resi - vs_resi_ave))
                    ds_rm = sorted(ds, reverse=True)[:n_rm]
                    for ind, d in enumerate(ds):
                        if d in ds_rm:
                            vs.pop(ind)
                else:
                    print('error')
                    print("grid number is wrong: %d != %d" % (n, len(vs)))
                    quit()
            xyInit[axis] = vs[0]
        return xyInit

    # vs: sorted values of x / y position of grid
    # min_n_cont: minimum value for continue value (=edge region)
    def _get_edge_position(self, vs, min_n_cont, max_dif):
        vs_edge = []
        for i, v in enumerate(vs):
            if i == 0:
                continue
            dif = abs(v - vs[i - 1])

            if dif < max_dif:
                n_cont = len(vs_edge)
                if n_cont >= min_n_cont:
                    v_mean_edge = numpy.mean(vs_edge)
                    return v_mean_edge
                vs_edge.append(v)
            else:
                vs_edge = []
        print("cannot find edge position")
        quit()

    def draw_grid_lines(self, img):
        yMax, xMax, _ = img.shape
        (xInit, yInit) = self.xyInit
        (xDif, yDif) = self.xyDif
        lines = []
        for col in self.cols:
            x = int(xInit + xDif * (col - 1))
            line = ((x, 0), (x, yMax))
            lines.append(line)
        for row in self.rows:
            y = int(yInit + yDif * (row - 1))
            line = ((0, y), (xMax, y))
            lines.append(line)
        imgop.draw_lines(img, lines)


class ImageObject:
    def __init__(self, img):
        self.areas = [0]
        self.rects = [0]
        self.edgeSets = [[]]
        self.xyCenters = [(0, 0)]
        self.yMax, self.xMax = img.gray.shape

    def find_objects(self, img):
        contours = imgop.find_contours(img.binary)
        contours_chosen = imgop.choose_contours_by_area_and_shape(img.color, contours, 20, 0.5)
        for contour in contours_chosen:
            self._store_contour_data(contour)

    def _store_contour_data(self, contour, xy_offset=(0, 0)):
        # area = imgop.get_contour_area(contour)
        # rect = imgop.get_contour_rect(contour)
        # edges = imgop.get_contour_edges(contour)

        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        # edges = list(contour)
        edges = contour.reshape(-1, 2)

        xCenter, yCenter = imgop.get_centroid(contour)
        # xCenter = rect[0] + rect[2]/2
        # yCenter = rect[1] + rect[3]/2
        if xy_offset != (0, 0):
            cedges = []
            for xy in edges:
                cx = xy[0] + xy_offset[0]
                cy = xy[1] + xy_offset[1]
                cedges.append([cx, cy])
            edges = cedges
            xCenter += xy_offset[0]
            yCenter += xy_offset[1]
        self.xyCenters.append((xCenter, yCenter))
        self.areas.append(area)
        self.rects.append(rect)
        self.edgeSets.append(edges)

    def _make_objid_ary(self):
        objid_ary = numpy.zeros((self.xMax, self.yMax), dtype=int)
        for i, element in enumerate(self.xyCenters):
            (x, y) = element
            objid_ary[x][y] = i
        self.objid_ary = objid_ary

    def get_ary_colony_from_imgobj_edge(self, img, objid, rx, ry):
        xyColonys = []
        ary = numpy.zeros((ry * 2, rx * 2), dtype=numpy.int32)
        # load all edges
        edges = self.edgeSets[objid]
        # local matrix
        (x_grid, y_grid) = self.xyCenters[objid]
        for edge in edges:
            x = edge[0] - x_grid + rx
            y = edge[1] - y_grid + ry
            ary[int(y), int(x)] = 1
        # y direction
        for y in range(ry * 2):
            inds = numpy.where(ary[y, :] == 1)[0]
            if len(inds) > 1:
                x_left = inds[0]
                x_right = inds[-1]
                for x in range(x_left, x_right):
                    ary[y, x] = 1
        # x direction
        for x in range(rx * 2):
            inds = numpy.where(ary[:, x] == 1)[0]
            if len(inds) > 1:
                y_left = inds[0]
                y_right = inds[-1]
                for y in range(y_left, y_right):
                    ary[y, x] = 1
        ary_colony = ary > 0
        return ary_colony


class Colony:
    def __init__(self, grid):
        self.gridposs = grid.poss
        self.gridxyDif = grid.xyDif
        self.gridxyRadius = grid.xyRadius
        self.gridpos2objid = {}
        self.gridpos2xy_center = {}
        self.gridpos2xys = {}
        self.edges_all = []

    def _get_colony_area_limit(self, xyDif, factor=2):
        square = xyDif[0] * xyDif[1]
        colony_area_limit = square * factor
        return colony_area_limit

    def find_colony_from_image_objects(self, img, imgobj, grid):
        # window = grid.xyDif[1]/3
        window_in = grid.xyDif[1] / 3
        window_out = grid.xyDif[1] / 2
        colony_area_limit = self._get_colony_area_limit(grid.xyDif)
        imgary_bin = img.binary

        xs_pos = [i[0] for i in grid.poss]
        xs_pos_out = [min(xs_pos), max(xs_pos)]
        ys_pos = [i[1] for i in grid.poss]
        ys_pos_out = [min(ys_pos), max(ys_pos)]

        idary = imgobj.objid_ary
        for gpos in grid.poss:
            window = window_in
            if gpos[0] in xs_pos_out or gpos[1] in ys_pos_out:
                window = window_out
            (x, y) = grid.pos2xyCenter[gpos]
            idary_focus = idary[int(x - window):int(x + window + 1), int(y - window):int(y + window + 1)]
            ids = idary_focus[idary_focus > 0]
            id_select = 0
            if len(ids) == 0:
                id_select = self._redetect_colony(imgary_bin, imgobj, grid, x, y, gpos)
                # print "Redetect: col=%2d row=%2d\tobjid:%d" % (gpos[0],gpos[1],id_select)
            amax = 0
            for i in ids:
                area = imgobj.areas[i]
                if area > amax and area < colony_area_limit:
                    id_select = i
                    amax = area
            self.gridpos2objid[gpos] = id_select
            self.gridpos2xy_center[gpos] = imgobj.xyCenters[id_select]

        edges_all = []
        for gpos in grid.poss:
            objid = self.gridpos2objid[gpos]
            edges = imgobj.edgeSets[objid]
            edges_all += list(edges)
        self.edges_all = edges_all
        imgop.draw_edges(img.color, edges_all, "colony", 2)

    def _redetect_colony(self, imgary_bin, imgobj, grid, cx, cy, gpos):
        objid = 0
        window = grid.xyDif[1] / 8
        imgary_bin_c = imgary_bin.copy()
        imgary_focus = imgary_bin[int(cy - window):int(cy + window + 1), int(cx - window):int(cx + window + 1)]
        if len(imgary_focus[imgary_focus == 0]) < MIN_COLONY_AREA:
            return objid

        rx = int(grid.xyDif[0] / 1.8)
        ry = int(grid.xyDif[1] / 1.8)
        imgary_focus = imgary_bin_c[int(cy - ry):int(cy + ry + 1), int(cx - rx):int(cx + rx + 1)]
        imgary_focus = self._put_value_to_circumference(imgary_focus, 255, 2)
        area_limit = (rx * 2 - 2) * (ry * 2 - 2)
        xy_offset = (cx - rx, cy - ry)
        found = imgobj.find_local_areamax_object(imgary_focus, area_limit, xy_offset)
        if found:
            objid = len(imgobj.areas) - 1
        imgop.create_img(imgary_focus, 're-C%2d-R%2d' % (gpos[0], gpos[1]), 2)
        return objid

    def _put_value_to_circumference(self, ary, value, width):
        ary[:, :width] = value
        ary[:width, :] = value
        ary[:, -width:] = value
        ary[-width:, :] = value
        return ary

    def make_colony_xys(self, img, imgobj, grid):
        rx = grid.xyRadius[0]
        ry = grid.xyRadius[1]
        for gpos in grid.poss:
            objid = self.gridpos2objid[gpos]
            xys = imgobj.get_ary_colony_from_imgobj_edge(img, objid, rx, ry)
            self.gridpos2xys[gpos] = xys


class Growth:
    window = 40  # focus window

    def __init__(self, poss, xyDif, r=8):
        self.r = r       # radius of central colony region
        self.gridposs = poss
        self.gridxyDif = xyDif
        self.gridpos2xys = {}
        self.gridpos2area = {}
        self.gridpos2mass = {}
        self.gridpos2cmass = {}
        self.gridpos2psgr = {}
        for gpos in self.gridposs:
            self.gridpos2psgr[gpos] = 0.0
        self.make_circular_mask()

    def get_growth(self, img, colony):
        imgary = img.gray
        oint_req = img.oint_bg - DIF_INT_COLONY  # required optical intensity as a colony
        # (rx, ry) = map(int, colony.gridxyDif)
        # rx = colony.gridxyRadius[0]
        # ry = colony.gridxyRadius[1]
        rx, ry = colony.gridxyRadius
        for gpos in self.gridposs:
            (xc, yc) = colony.gridpos2xy_center[gpos]
            ary_colony = colony.gridpos2xys[gpos]
            imgary_tile = imgary[int(yc - ry):int(yc + ry), int(xc - rx):int(xc + rx)]

            if imgary_tile.size == 0:
                area, mass, cmass = 0, 0, 0
                ary_colony_t = numpy.array([])
            else:
                oint_bg = self.get_bg_tile(ary_colony, imgary_tile, img.oint_bg)
                # oints = imgary_tile[numpy.invert(ary_colony)]
                dif = (imgary_tile < oint_req)
                ary_colony_t = dif * ary_colony
                oints = imgary_tile[ary_colony_t]
                oints = oints[oints > 0]
                area = len(oints)
                # mass = numpy.sum(-numpy.log10(oints/img.oint_bg))
                mass = numpy.sum(-numpy.log10(oints / oint_bg))
                mass = round(mass, 2)
                # cmass = self._get_centermass(imgary, img.oint_bg, xc, yc, oint_req)
                cmass = self._get_centermass(imgary, oint_bg, xc, yc, oint_req)

            self.gridpos2area[gpos] = area
            self.gridpos2mass[gpos] = mass
            self.gridpos2xys[gpos] = ary_colony_t
            self.gridpos2cmass[gpos] = cmass

    def get_bg_tile(self, ary_colony, imgary_tile, oint_bg_plate):
        oints = imgary_tile[numpy.invert(ary_colony)]
        p = oints.flatten()
        p = p[numpy.abs(p - oint_bg_plate) < 10]
        oint_bg = numpy.mean(p)
        return oint_bg

    def make_circular_mask(self):
        # make circular mask
        yp, xp = self.window / 2, self.window / 2
        y, x = numpy.ogrid[int(-yp):int(self.window - yp), int(-xp):int(self.window - xp)]
        mask = x * x + y * y <= self.r * self.r
        self.mask = mask
        # plt.imshow(mask)
        # plt.show()
        # quit()

    def _get_centermass(self, imgary, oint_bg, xc, yc, oint_req):
        # make window
        side = int(self.window / 2)

        xs = xc - side
        xe = xc + side
        ys = yc - side
        ye = yc + side

        # get values within circle
        imgary_tile = imgary[int(ys):int(ye), int(xs):int(xe)]
        # imgary_tile = imgary
        # vs = imgary_tile[self.mask]
        # oints = vs[((vs<oint_req) * (vs>0))]
        # oints = vs[vs>0]

        oints = imgary_tile[self.mask]
        # vals = vals[self.mask]
        # to check
        # imgary[self.mask] = 255
        # imgary_tile[side, side] = 255
        # imgary_tile[self.mask] = 255

        cmass = numpy.sum(-numpy.log10(oints / oint_bg))
        cmass = round(cmass, 2)

        # plt.imshow(imgary_tile)
        # plt.show()
        return cmass


def load_img(path_img):
    fnames_img = subprocess.check_output(['ls', '-v', path_img]).decode('utf-8').split()
    fnames_img = [str(Path(path_img) / f) for f in fnames_img]
    return fnames_img


def load_args():
    if args.debug_mode:
        global DEBUG
        DEBUG = 1
        imgop.DEBUG = 1
        imgop.IMG_OUT_LEVEL = 2

    path_img = args.imgpath

    fnames_img = load_img(path_img)
    try:
        return fnames_img, args.output
    except:
        return fnames_img


def load_plate(fname):
    # load image
    img = imgop.ImageStore(fname)
    img.cut_level = args.cut_level
    return img


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
    img = load_plate(fname)
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
        img = load_plate(fname)
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
    # table = get_colony_table(fnames_img, growth_packss, poss, out_fname)
    # print(table)


if __name__ == "__main__":
    main()
