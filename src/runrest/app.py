import argparse
from runrest.parser import parse_input_file
from runrest.rest.wrapper import run_rest
from runrest.rest.utils.rsmt_utils import plot_rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')
    parser.add_argument('--output', type=str, help='Output file')
    
    args = parser.parse_args()
    
    pointslist = parse_input_file(args.FILE)
    reslist = run_rest(pointslist)
    if args.output:
        with open(args.output, 'w') as f:
            for res in reslist:
                f.write(' '.join([str(coord) for coord in res]) + '\n')
    else:
        for res in reslist:
            print(' '.join([str(coord) for coord in res]))    
    # plot_rest(pointslist[0], reslist[0])


if __name__ == "__main__":
    main()