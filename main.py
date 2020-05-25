import argparse
import os
import tensorflow as tf

tf.set_random_seed(19)
from model import cyclegan

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_dir', dest='dataset_dir', default='MRCT', help='path of the dataset')

parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1000, help='# images used to train')

parser.add_argument('--load_size0', dest='load_size0', type=int, default=400, help='scale images to this size')
parser.add_argument('--load_size1', dest='load_size1', type=int, default=284, help='scale images to this size')
parser.add_argument('--fine_size0', dest='fine_size0', type=int, default=384, help='then crop to this size')
parser.add_argument('--fine_size1', dest='fine_size1', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')

parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='BtoA', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')

parser.add_argument('--save_freq', dest='save_freq', type=int, default=3000,
                    help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=200,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True,
                    help='if continue training, load the latest model: 1: true, 0: false')

parser.add_argument('--sigma', dest='sigma', type=float, default=2.0, help='sigma in gaussian kernel')
parser.add_argument('--eps', dest='eps', type=float, default=1e-5, help='epslion added to V in MIND')
parser.add_argument('--neigh_size', dest='neigh_size', type=float, default=9, help='neighborhood size of MIND')
parser.add_argument('--patch_size', dest='patch_size', type=float, default=7, help='patch size of MIND')
parser.add_argument('--L2_lambda', dest='L2_lambda', type=float, default=5.0, help='weight of MIND in objective')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True,
                    help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=0,
                    help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)
            # else model.valid_CT(args)


if __name__ == '__main__':
    tf.app.run()
