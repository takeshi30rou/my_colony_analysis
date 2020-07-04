
'''
Normalize growth values in a plate

The program execute following normalization (in order)
1) Plate normalization
2) Spatial normalization
3) Row/Col normalization


Reference)
Baryshnikova A, Costanzo M, Kim Y, Ding H, Koh J, Toufighi K, Youn J-Y, Ou J, San Luis B-J, Bandyopadhyay S, Hibbs M, Hess D, Gingras A-C, Bader GD, Troyanskaya OG, Brown GW, Andrews B, Boone C, Myers CL: Quantitative analysis of fitness and genetic interactions in yeast on a genome scale. Nat Meth 2010, 7:1017-1024.
'''
import argparse
import numpy as np
import colony_analysis.growth2ngrowth as g2n


# INPUT
# ary: sample array
# refary: reference array for spatial norm.
#
# ary is numpy 2d masked array
# (0 must be masked for log conversion process)
#
# OUTPUT
# nary: same format to ary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input CSV path', required=True)
    parser.add_argument('-o', '--output', help='output CSV path', required=True)
    args = parser.parse_args()

    fname_in = args.input
    fname_out = args.output

    ary, poss = g2n.load_csv(fname_in)
    nary = ary.copy()

    for ind in range(4):
        # reference array
        refary = np.ones(ary[:, :, ind].shape)
        nary[:, :, ind] = g2n.norm_growth(ary[:, :, ind], refary)

    g2n.output_csv(nary, poss, fname_out)


if __name__ == "__main__":
    main()
