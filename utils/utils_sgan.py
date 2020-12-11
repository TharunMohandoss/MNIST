import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e', '--exp_name',
        type=str,
        metavar='M',
        required=True,
        default='None',
        help='Enter the experiment name')
    argparser.add_argument(
        '-d', '--data_set',
        type=str,
        metavar='M',
        required=True,
        default='None',
        help='Enter the dataset name')
    args = argparser.parse_args()
    return args
