"""Simple script to produce metadata file"""

# System
import argparse
import logging

# Locals
from utils.metadata import prepare_metadata, save_metadata

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing graph data')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    logging.info('Preparing metadata in %s', args.data_dir)
    metadata = prepare_metadata(args.data_dir)
    save_metadata(metadata, args.data_dir)
    logging.info('All done!')

if __name__ == '__main__':
    main()
