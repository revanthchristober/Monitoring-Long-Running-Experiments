import argparse

from experiments.experiment1 import run_experiment1
from experiments.experiment2 import run_experiment2

def main():
    parser = argparse.ArgumentParser(description="Run different experiments.")
    parser.add_argument('--experiment', type=str, choices=['experiment1', 'experiment2'], required=True, help='Specify which experiment to run.')

    args = parser.parse_args()

    if args.experiment == 'experiment1':
        run_experiment1()
    elif args.experiment == 'experiment2':
        run_experiment2()

if __name__ == '__main__':
    main()
