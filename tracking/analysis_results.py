import _init_paths
import matplotlib.pyplot as plt
import argparse

plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser(description='analysis config')
    parser.add_argument('--dataset_name', type=str, help='')
    parser.add_argument('--tracker_name', type=str, help='')
    parser.add_argument('--tracker_version', type=str, help='')
    parser.add_argument('--num_epoch', type=int, default=500, help='')

    args = parser.parse_args()

    trackers = []
    dataset_name = args.dataset_name
    # 'uav'

    """asymtrack"""
    trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_version, dataset_name=dataset_name,
                                run_ids=None, display_name=args.tracker_version, num_epoch=args.num_epoch))

    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                  force_evaluation=True)


if __name__ == '__main__':
    main()
