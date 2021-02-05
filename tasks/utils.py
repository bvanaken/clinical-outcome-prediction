import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--seed', default=123, type=int)

    return parser.parse_args()


def parse_args_combine():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--data_dirs', nargs='+', help='Data directories to combine', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--seed', default=123, type=int)

    return parser.parse_args()
