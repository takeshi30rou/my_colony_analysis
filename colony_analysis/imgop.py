'''
Image handling module
'''
import cv2
import numpy as np

# color value for drawing image
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

outId = 0   # start ID for output image file
out_folder = '.tmp'
msgFlag = False
IMG_OUT_LEVEL = 1
DEBUG = 0

# def get_mat_from_array(a):
#     # dtype2depth = {
#     #     'uint8':   cv.IPL_DEPTH_8U,
#     #     'int8':    cv.IPL_DEPTH_8S,
#     #     'uint16':  cv.IPL_DEPTH_16U,
#     #     'int16':   cv.IPL_DEPTH_16S,
#     #     'int32':   cv.IPL_DEPTH_32S,
#     #     'float32': cv.IPL_DEPTH_32F,
#     #     'float64': cv.IPL_DEPTH_64F,
#     # }
#     try:
#         nChannels = a.shape[2]
#     except:
#         nChannels = 1
#     # cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
#     #     dtype2depth[str(a.dtype)], nChannels)
#     cv.SetData(cv_im, a.tostring(),
#         a.dtype.itemsize*nChannels*a.shape[1])
#     mat = cv.GetMat(cv_im)
#     return mat


def create_img(img, fname, outLevel=3, ary_flag=False):
    if outLevel > IMG_OUT_LEVEL:
        return
    global outId
    outId += 1

    # if ary_flag:
    #     img = get_mat_from_array(img)

    # cv.SaveImage('%s/%02d-%s.jpg'%(out_folder,outId,fname), img)
    if DEBUG:
        cv2.imwrite(f'{out_folder}/{outId:02}-{fname}.png', img)
    else:
        cv2.imwrite(f'{out_folder}/{fname}.png', img)


# def convert_to_gray(img, outLevel=2):

    # print_msg('convert to gray')
    # imgGray = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
    # cv.CvtColor(img, imgGray, cv.CV_BGR2GRAY)
    # #create_img(imgGray, "convert_to_gray", outLevel)
    # create_img(imgGray, "convert_to_gray", 2)
    # return imgGray


# class for handling image
class ImageStore:
    def __init__(self, fname, outLevel=2, grayFlag=False):
        self.color = 0
        self.gray = 0
        self.binary = 0
        self.cut_level = 2
        self.oint_bg = 0  # optical intensity of background
        if grayFlag:
            self.gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            self.color = cv2.imread(fname)
            self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
            # self.gray = convert_to_gray(self.color, outLevel)


def print_msg(msg):
    if msgFlag:
        print(msg)


# def rotate(img, deg, outLevel=2):
#     print_msg('rotate')
#     imgOut = cv.CloneImage(img)
#     mat = cv.CreateMat(2,3,cv.CV_32FC1)
#     cv.GetRotationMatrix2D((0,0), deg, 1, mat)
#     cv.WarpAffine(img, imgOut, mat, cv.CV_WARP_FILL_OUTLIERS+cv.CV_WARP_INVERSE_MAP,(255,255,255))
#     create_img(imgOut, "rotate", outLevel)
#     return imgOut


# def canny_edge_detect(imgGray, param1, param2, ksize, outLevel=2):
#     print_msg('canny edge')
#     imgBin = cv.CreateImage(cv.GetSize(imgGray), cv.IPL_DEPTH_8U,1)
#     cv.Canny(imgGray, imgBin, param1, param2, ksize)
#     create_img(imgBin, "canny_edge_detect", outLevel)
#     return imgBin


def binarize_with_threshold(imgGray, threshold, outLevel=2):
    print_msg('binarizing with threshold')
    # imgBin = cv.CreateImage(cv.GetSize(imgGray), cv.IPL_DEPTH_8U,1)
    # cv.Threshold(imgGray, imgBin, threshold, 255, cv.CV_THRESH_BINARY)
    _, imgBin = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY)
    create_img(imgBin, "binarize_with_threshold", outLevel)
    return imgBin


# def smooth(img, ksize=3, outLevel=2):
#     print_msg('smoothing')
#     imgOut = cv.CloneImage(img)
#     cv.Smooth(img,imgOut,cv.CV_GAUSSIAN, ksize)
#     create_img(imgOut, "smooth", outLevel)
#     return imgOut


def blur(mat, p1=51, p2=51, outLevel=2):
    # mat_out = cv.CloneMat(mat)
    # cv.Smooth(mat, mat_out, cv.CV_BLUR, param1=p1, param2=p2)
    print_msg('smoothing')
    mat_out = cv2.blur(mat, (p1, p2))
    create_img(mat_out, "smooth", outLevel)
    return mat_out


def unsharp_mask(imgGray, k=5, kernel=21, outLevel=2):
    print_msg('unsharp masking')
    # imgSmooth = cv.CreateImage(cv.GetSize(imgGray), cv.IPL_DEPTH_8U,1)
    # imgSub = cv.CreateImage(cv.GetSize(imgGray), cv.IPL_DEPTH_8U,1)
    # imgOut = cv.CreateImage(cv.GetSize(imgGray), cv.IPL_DEPTH_8U,1)
    # imgt = cv.CloneImage(imgGray)
    # cv.Smooth(imgt,imgSmooth,cv.CV_GAUSSIAN, kernel)
    # cv.Sub(imgGray, imgSmooth, imgSub)

    imgSmooth = cv2.GaussianBlur(imgGray, (kernel, kernel), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    imgSub = cv2.subtract(imgGray, imgSmooth)

    # m = cv.GetMat(imgSub)
    # a = numpy.asarray(m)
    # a = k * imgSub
    # mat = cv.fromarray(a)
    # cv.Add(imgGray, mat, imgOut)
    mat = imgSub * k
    imgOut = cv2.add(imgGray, mat)
    create_img(imgOut, "unsharp masking", outLevel)

    return imgOut  # gray image


# def dilate(img, iterSize=6, outLevel=2):
#     print_msg('dilating')
#     imgOut = cv.CloneImage(img)
#     cv.Dilate(img, imgOut, iterations=iterSize)
#     create_img(imgOut, "dilate", outLevel)
#     return imgOut


# def dilate_and_erode(img, iterSize=6, outLevel=2):
#     print_msg('dilate and erode')
#     imgDilate = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U,1)
#     imgErode = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U,1)
#     imgOut = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U,1)
#     cv.Dilate(img,imgDilate,iterations=iterSize)
#     cv.Erode(imgDilate,imgOut,iterations=iterSize)
#     create_img(imgOut, "dilate_and_erode", outLevel)
#     return imgOut


def crop(img, xyTL, xyBR, outLevel=2):
    return img[xyTL[1]:xyBR[1], xyTL[0]:xyBR[0]]
    # print_msg('crop image')
    # imgOut = cv.CreateImage(((xyBR[0]-xyTL[0],xyBR[1]-xyTL[1])), cv.IPL_DEPTH_8U,1)
    # imgOut = np.empty(img.shape, dtype=np.uint8)
    # rectInfo = (xyTL[0], xyTL[1], xyBR[0]-xyTL[0], xyBR[1]-xyTL[1])
    # region = cv.GetSubRect(img, rectInfo)
    # cv.Copy(region, imgOut)
    # create_img(imgOut, "crop", outLevel)


def paste(img, srcAry, xyTL, xyBR, outLevel=2):
    print_msg('paste image')
    # imgOut = cv.CloneImage(img)
    # imgOutAry = get_array_from_image(imgOut)
    imgOutAry = np.copy(img)
    imgOutAry[xyTL[1]:xyBR[1], xyTL[0]:xyBR[0]] = srcAry
    create_img(imgOutAry, "paste", outLevel, 1)
    return imgOutAry


# def split_color_ch(channel, outLevel=2):
#     print_msg('split color channel')
#     size = cv.GetSize(self.color)
#     imgRch = cv.CreateImage(size, cv.IPL_DEPTH_8U,1)
#     imgGch = cv.CreateImage(size, cv.IPL_DEPTH_8U,1)
#     imgBch = cv.CreateImage(size, cv.IPL_DEPTH_8U,1)
#     cv.Split(self.color, imgBch, imgGch, imgRch, None)
#     if channel == "R":
#         imgOut = imgRch
#     if channel == "G":
#         imgOut = imgGch
#     if channel == "B":
#         imgOut = imgBch
#     create_img(imgOut, "split_color_ch_to_%s"%channel, outLevel)
#     return imgOut


def find_contours(imgBin):
    print_msg('find contours')
    # img = cv.CloneImage(imgBin)
    # contours = cv.FindContours(img, cv.CreateMemStorage(), method=cv.CV_CHAIN_APPROX_NONE)
    # del img
    contours, hierarchy = cv2.findContours(
        imgBin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


# def find_contours_ary(ary):
#     print_msg('find contours')
#     aryin = ary.copy()
#     aryin = npary2cvipl(aryin)
#     contours = cv.FindContours(aryin, cv.CreateMemStorage(), method=cv.CV_CHAIN_APPROX_NONE)
#     return contours


# def draw_contours(img, contours):
#     while contours:
#         rect = cv.BoundingRect(contours)
#         cv.Rectangle(imgOut, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), RED, 1)
#         cv.DrawContours(imgOut, contours, RED, RED, 1, 3)
#         contours = contours.h_next()


def draw_edges(img, edges, outName="draw_edge", outLevel=2):
    for edge in edges:
        img[edge[1], edge[0]] = RED
    create_img(img, outName, outLevel)


# def draw_contours_edge(img, contours):
#     RED = (0,0,255)
#     for pos in contours:
#         for p in pos:
#             img[p[1],p[0]] = RED
#     return img


# def draw_points(img, points):
#     RED = (100,100,100)
#     for xy in points:
#         img[xy[1],xy[0]] = RED
#     create_img(img, "fill_area", 1)


def choose_contours_by_area_and_shape(img, contours, minArea, maxSqDegree, outLevel=1):
    imgOut = np.copy(img)
    contoursSel = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < minArea:
            continue
        rect = cv2.boundingRect(cnt)
        sq_degree = abs(rect[2] / rect[3] - 1.0)
        if sq_degree > maxSqDegree:
            continue
        rn = get_roundness(cnt)
        if abs(1 - rn) > 0.5:
            continue
        contoursSel.append(cnt)
        # draw_contours_edge(imgOut, cnt)

    create_img(imgOut, "choose_contours", 2)
    return contoursSel

    # while contours:
    #     # check area
    #     area = cv2.contourArea(contours)
    #     if area < minArea:
    #         contours = contours.h_next()
    #         continue
    #     # check square degree (width / height)
    #     rect = cv.BoundingRect(contours)
    #     sqDegree = abs(float(rect[2])/float(rect[3]) - 1.0)
    #     if sqDegree > maxSqDegree:
    #         contours = contours.h_next()
    #         continue
    #     # check roundness
    #     rn = get_roundness(contours)
    #     if abs(1-rn) > 0.5:
    #         contours = contours.h_next()
    #         continue
    #     contoursSel.append(contours)
    #     #cv.Rectangle(imgOut, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), RED, 1)
    #     draw_contours_edge(imgOut, contours)
    #     contours = contours.h_next()
    create_img(imgOut, "choose_contours", 2)
    del contours
    return contoursSel


# def choose_contours_by_size_and_shape(contours, cinfo, minArea):
#     print('find contours')
#     contoursSel = []
#     while contours:
#         area = cv.ContourArea(contours)
#         # check area
#         if area < minArea:
#             contours = contours.h_next()
#             continue
#         rect = cv.BoundingRect(contours)
#         # check square degree (width / height)
#         square_degree = float(rect[2])/float(rect[3])
#         if square_degree > 2 or square_degree < 0.5:
#             contours = contours.h_next()
#             continue

#         rn = get_roundness(contours)
#         print(rn)
#         if abs(1-rn) > 0.2:
#             contours = contours.h_next()
#             continue

#         xCenter = rect[0] + rect[2]/2
#         yCenter = rect[1] + rect[3]/2
#         xCenter, yCenter = get_centroid(contours)
#         contoursSel.append(contours)

#         cinfo.xyCenters.append((xCenter,yCenter))
#         cinfo.areas.append(area)
#         cinfo.rects.append(rect)
#         cinfo.edgeSets.append([i for i in contours])
#         contours = contours.h_next()
#     del contours
#     return contoursSel


def get_roundness(contour):
    area = cv2.contourArea(contour)
    circ = cv2.arcLength(contour, True)
    roundness = 4 * 3.14 * area / (circ ** 2)
    return roundness


def get_centroid(contour):
    m = cv2.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return cx, cy

# def get_contour_area(contour):
#     return cv.ContourArea(contour)

# def get_contour_rect(contour):
#     return cv.BoundingRect(contour)

# def get_contour_edges(contour):
#     edges = [i for i in contour]
#     return edges

# def get_img_size(img):
#     return cv.GetSize(img)

# def npary2cvipl(npary):
#     img = cv.CreateImageHeader((npary.shape[1],npary.shape[0]),cv.IPL_DEPTH_8U,1)
#     cv.SetData(img, npary.tostring())
#     return img


def draw_lines(img, lines, mode="grid"):
    imgOut = np.copy(img)
    if mode == "grid":
        for line in lines:
            cv2.line(imgOut, line[0], line[1], BLUE, 1)
        create_img(imgOut, "make grid", 2)
    elif mode == "boundary":
        for line in lines:
            cv2.line(imgOut, line[0], line[1], 255, 1)
        create_img(imgOut, "make boundary")
        return imgOut


# def get_array_from_image(img):
#     mat = cv.GetMat(img)
#     ary = np.array(mat)
#     return ary
    """
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }
    arrdtype=img.depth
    a = np.fromstring(
        img.tostring(),
        dtype=depth2dtype[img.depth],
        count=img.width*img.height*img.nChannels)
    a.shape = (img.height,img.width,img.nChannels)
    a = a
    return a
    """
