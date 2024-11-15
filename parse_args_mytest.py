from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='KD-UIRnet_detect')

    parser.add_argument('--train_dataset_dir', type=str, default='D:/IRNet/datasets/')
    parser.add_argument('--test_dataset_dir', type=str, default='D:/IRNet/datasets/')
    parser.add_argument('--save_dir', type=str, default='save_model')
    parser.add_argument('--save_prefix', type=str, default="")
    parser.add_argument('--result_WS', type=str, default="D:/AweIRmodule/result_WS/")
    parser.add_argument('--save_model', type=str, default="save_model/")
    parser.add_argument('--model_weight', type=str, default="model_weight.pth.tar")
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='threshold(default: 0.01)')
    parser.add_argument('--threshold_2', type=float, default=0.15,
                        help='threshold_2(default: 0.15)')
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
