import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_path',
        default='dataset/winbus_train.txt',
        type=str,
        help='Training data')
    parser.add_argument(
        '--test_path',
        default='dataset/winbus_test.txt',
        type=str,
        help='Test data')
    parser.add_argument(
        '--resume_path',
        default = None,
        type=str,
        help='Resume data path')
    parser.add_argument(
        '--n_epochs',
        default=10,
        type=int,
        help='Train eopchs')
    parser.add_argument(
        '--lr',
        default=0.1,
        type=float,
        help='Learning rate')
    parser.add_argument(
        '--lr_patience',
        default=500,
        type=int,
        help='Learning rate down patience')
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch-size')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='No training.')
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='No validation.')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test weight.')
    parser.add_argument(
        '--gen_data_num',
        default=0,
        type=int,
        help='Generate simulate data number')

    args = parser.parse_args()

    return args
