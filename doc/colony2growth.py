'''
Compute three growth parameters from the growth curve
'''

from colony_analysis import colony2growth as c2g


def main():
    # import optparse
    # parser = optparse.OptionParser(usage="%prog [input csv file] [output csv file]")
    # (options, args) = parser.parse_args()
    # if len(args) != 2:
    #     parser.print_help()
    #     quit()
    # fname_in = args[0]
    # fname_out = args[1]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input CSV path', required=True)
    parser.add_argument(
        '-o',
        '--output',
        help='output CSV path',
        required=True)
    args = parser.parse_args()

    fname_in = args.input
    fname_out = args.output
    # conventional value
    times, poss, vss = c2g.load_csv(fname_in, vtype='area')
    convs = c2g.get_conv_value(times, vss)
    # growth paramenters
    times, poss, vss = c2g.load_csv(fname_in, vtype='cmass')
    # times, poss, vss = load_csv(fname_in, vtype='mass')
    gparams = c2g.get_growth_params(times, vss)

    c2g.output_as_csv(poss, convs, gparams, fname_out)


if __name__ == '__main__':
    main()
