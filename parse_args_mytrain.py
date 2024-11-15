from model.utils import *


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='KD-UIRnet_train')

    parser.add_argument('--train_dataset_dir', type=str, default='D:/IRNet/datasets/')
    parser.add_argument('--test_dataset_dir', type=str, default='D:/IRNet/datasets/')
    parser.add_argument('--save_dir', type=str, default='save_model')
    parser.add_argument('--save_prefix', type=str, default="")
    parser.add_argument('--save_student_model', type=str, default="save_student_model")
    parser.add_argument('--save_dis_model', type=str, default="save_dis_model/")
    parser.add_argument('--save_model', type=str, default="save_model/")
    parser.add_argument('--result_WS', type=str, default="D:/IRNet/result_WS/")
    parser.add_argument('--model_weight', type=str, default="model_weight.pth.tar")
    parser.add_argument('--split_method', type=str, default='70_20/',
                        help='70_20')
    parser.add_argument('--train_txt', type=str, default='train.txt',
                        help='train.txt')
    parser.add_argument('--test_txt', type=str, default='test.txt',
                        help='test.txt')
    parser.add_argument('--epochs', type=int, default=150,
                        help='epochs(default: 150)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad, SGD')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['Cosin eAnnealingLR', 'ReduceLROnPlateau', 'StepLR'])
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    args = parser.parse_args()

    # the parser
    return args
