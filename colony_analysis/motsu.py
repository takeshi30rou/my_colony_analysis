'''
Multilevel thresholding module

Required module:  opencv, numpy

Reference)
Ping-Sung Liao, Tse-sheng Chen and Pau-choo Chung (2001) "A Fast Algorithm for Multilevel Thresholding"
'''
import sys
import numpy
import cv2

L = 256

# def load_image(fname):
#     img = cv.LoadImage(fname)
#     return img


def get_thresholds(img, level):
    pAry = get_histogram(img)
    pAry = ignore_extreme_value(pAry)
    thresholds = calc_thresholds(pAry, level)
    return thresholds


def get_histogram(img):
    (pAry, ed) = numpy.histogram(img, range(0, 257))
    return pAry


def ignore_extreme_value(pAry):
    pAry[0] = 0
    pAry[255] = 0
    return pAry


def build_lookup_table(pAry):
    pTable = numpy.zeros((L, L))
    sTable = numpy.zeros((L, L))
    hTable = numpy.zeros((L, L))

    for i in range(0, L):
        pTable[i][i] = pAry[i]
        sTable[i][i] = pAry[i] * float(i)

    for i in range(1, L - 1):
        pTable[1][i + 1] = pTable[1][i] + pAry[i + 1]
        sTable[1][i + 1] = sTable[1][i] + pAry[i + 1] * float(i + 1)

    for i in range(2, L):
        for j in range(i + 1, L):
            pTable[i][j] = pTable[1][j] - pTable[1][i - 1]
            sTable[i][j] = sTable[1][j] - sTable[1][i - 1]

    for i in range(1, L):
        for j in range(i + 1, L):
            if pTable[i][j] != 0:
                hTable[i][j] = sTable[i][j]**2 / pTable[i][j]
            else:
                hTable[i][j] = 0
    return hTable


# Calculate threshold
# M = 2, 3, 4
def calc_thresholds(pAry, M=2):
    hTable = build_lookup_table(pAry)
    ts = numpy.zeros(M, dtype=int)
    bcVarMax = 0.

    if M == 2:
        for i in range(1, L - M):
            bcVar = hTable[1][i] + hTable[i + 1][L - 1]
            if bcVar > bcVarMax:
                ts[1] = i
                bcVarMax = bcVar
    elif M == 3:
        for i in range(1, L - M):
            for j in range(i + 1, L - M + 1):
                bcVar = hTable[1][i] + hTable[i + 1][j] + hTable[j + 1][L - 1]
                if bcVar > bcVarMax:
                    ts[1] = i
                    ts[2] = j
                    bcVarMax = bcVar
    elif M == 4:
        for i in range(1, L - M):
            for j in range(i + 1, L - M + 1):
                for k in range(j + 1, L - M + 2):
                    bcVar = hTable[1][i] + hTable[i + 1][j] + \
                        hTable[j + 1][k] + hTable[k + 1][L - 1]
                    if bcVar > bcVarMax:
                        ts[1] = i
                        ts[2] = j
                        ts[3] = k
                        bcVarMax = bcVar
    else:
        print('Threshold number, M = 2 ~ 4.')
        exit()
    return ts


def main(fname, M=3):
    pAry = cv2.imread(fname)
    ts = calc_thresholds(pAry, M)
    return ts


if __name__ == "__main__":
    argvs = sys.argv
    # if len(argvs) != 3:
    #     print("usage: ./motsu.py [image filename] [threshold number]")
    #     exit()
    print(argvs)
    fname = argvs[1]
    M = int(argvs[2])
    ts = main(fname, M)
    print("Tresholds: ", ts)
