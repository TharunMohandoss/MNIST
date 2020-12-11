import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-m', '--mode',
        metavar='M',
        default='None',
        help='The Mode')
    args = argparser.parse_args()
    return args
