import argparse

from common.model_utils import find_checkpoint_with_max_step, find_checkpoint_with_latest_date, clean_checkpoint_directory

if __name__ == '__main__':

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoints', help='Directory with checkpoints.', default='models/mevea/mantsinen/ppo/model_checkpoints')
    parser.add_argument('-a', '--all', help='Delete all checkpoints', default=True, type=bool)
    args = parser.parse_args()

    # delete checkpoints

    if args.all:
        clean_checkpoint_directory(args.checkpoints, exclude=[])
    else:
        max_step_checkpoint = find_checkpoint_with_max_step(args.checkpoints)
        last_date_checkpoint = find_checkpoint_with_latest_date(args.checkpoints)
        clean_checkpoint_directory(args.checkpoints, exclude=[last_date_checkpoint, max_step_checkpoint])
