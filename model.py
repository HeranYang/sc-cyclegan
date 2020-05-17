from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from collections import namedtuple
import numpy as np

from module import *
from utils import *
from PIL import Image
from skimage import transform
import imageio


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size0 = args.fine_size0
        self.image_size1 = args.fine_size1
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.L2_lambda = args.L2_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        self.MIND = MIND

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size0 image_size1 \
                              gf_dim df_dim output_c_dim is_training  \
                               sigma eps neigh_size patch_size')
        self.options = OPTIONS._make((args.batch_size, args.fine_size0, args.fine_size1,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train',
                                      args.sigma, args.eps, args.neigh_size, args.patch_size))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.pool = ImagePool(args.max_size)


    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size0, self.image_size1,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")

        self.real_A_MIND = self.MIND(self.real_A, self.options, name='realA_MIND')
        self.fake_B_MIND = self.MIND(self.fake_B, self.options, name='fakeB_MIND')
        self.real_B_MIND = self.MIND(self.real_B, self.options, name='realB_MIND')
        self.fake_A_MIND = self.MIND(self.fake_A, self.options, name='fakeA_MIND')

        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_MIND = abs_criterion(self.real_A_MIND, self.fake_B_MIND) \
            + abs_criterion(self.real_B_MIND, self.fake_A_MIND)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + self.L2_lambda * abs_criterion(self.real_A_MIND, self.fake_B_MIND) \
            + self.L2_lambda * abs_criterion(self.real_B_MIND, self.fake_A_MIND)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size0, self.image_size1,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size0, self.image_size1,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_MIND_sum = tf.summary.scalar("g_loss_MIND", self.g_loss_MIND)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_MIND_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size0, self.image_size1,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size0, self.image_size1,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)


    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        saver_all = tf.train.Saver(max_to_keep = None)

        for epoch in range(args.epoch):
            # A: CT
            # B: T1
            trdataSize = 27
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)

            idA_all = np.arange(trdataSize) + 1
            np.random.shuffle(idA_all)

            dataInfoFile = open(r'dataset/trainInfo.txt','r')
            sourceInLines = dataInfoFile.readlines()
            dataInfoFile.close()
            dataInfo = []
            for line in sourceInLines:
                temp1 = line.strip('\n')
                temp2 = temp1.split(' ')
                dataInfo.append(temp2)

            randBd = 5

            for idA in idA_all:

                sliceNumA = int(dataInfo[idA-1][3])
                sliceA_all = np.arange(sliceNumA)
                np.random.shuffle(sliceA_all)

                batch_idxs = sliceNumA // self.batch_size

                for idx in range(0, batch_idxs):

                    idB_all = np.arange(trdataSize) + 1
                    idB_all = np.delete(idB_all, idA - 1)
                    np.random.shuffle(idB_all)

                    sliceNumB = int(dataInfo[idB_all[0] - 1][3])

                    dataA_idxs = []
                    dataB_idxs = []
                    for i in sliceA_all[idx * self.batch_size:(idx + 1) * self.batch_size]:
                        dataA_idxs.append(i + 1)

                        i_Bslice = round( i / (sliceNumA-1.) * (sliceNumB-1.) )
                        if i_Bslice >= randBd and i_Bslice <= sliceNumB-randBd-1:
                            i_Bslice = i_Bslice + np.random.randint((-1)*randBd,randBd)
                        dataB_idxs.append(i_Bslice + 1)

                    batch_files = list(zip(dataA_idxs, dataB_idxs))

                    batch_images = [
                        load_train_data(batch_file, idA, idB_all[0], args.load_size0, args.load_size1, args.fine_size0,
                                        args.fine_size1) for batch_file in batch_files]
                    batch_images = np.array(batch_images).astype(np.float32)

                    if idA == idB_all[0]:
                        raise RuntimeError('MRI and CT should come from different subjects!')

                    # Update G network and record fake outputs
                    fake_A, fake_B, _, summary_str = self.sess.run(
                        [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                        feed_dict={self.real_data: batch_images, self.lr: lr})
                    self.writer.add_summary(summary_str, counter)
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])

                    # Update D network
                    _, summary_str = self.sess.run(
                        [self.d_optim, self.d_sum],
                        feed_dict={self.real_data: batch_images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B,
                                   self.lr: lr})
                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    print(("Epoch: [%2d] SubsetA: [%2d] [%4d/%4d] time: %4.4f" % (
                        epoch, idA, idx, batch_idxs, time.time() - start_time)))

                    if np.mod(counter, args.print_freq) == 1:
                        self.sample_model(args.sample_dir, epoch, idA, idx)

                    if np.mod(counter, args.save_freq) == 2:
                        self.save(args.checkpoint_dir, counter)

            save_epoch_file = os.path.join(args.checkpoint_dir, 'MRCT_epoch')
            if not os.path.exists(save_epoch_file):
                os.makedirs(save_epoch_file)
            saver_all.save(self.sess, os.path.join(save_epoch_file, 'cyclegans.epoch'), global_step=epoch)


    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s_%s" % (self.dataset_dir, self.image_size0, self.image_size1)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load_valid(self, checkpoint_dir, epoch):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_epoch" % (self.dataset_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt_name = os.path.basename('cyclegans.epoch-{}'.format(epoch))
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_dir, self.image_size0, self.image_size1)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def sample_model(self, sample_dir, epoch, idA, idx):

        randBd = 5
        trdataSize = 27

        dataInfoFile = open(r'dataset/trainInfo.txt', 'r')
        sourceInLines = dataInfoFile.readlines()
        dataInfoFile.close()
        dataInfo = []
        for line in sourceInLines:
            temp1 = line.strip('\n')
            temp2 = temp1.split(' ')
            dataInfo.append(temp2)

        idB_all = np.arange(trdataSize) + 1
        idB_all = np.delete(idB_all, idA - 1)
        np.random.shuffle(idB_all)

        sliceNumA = int(dataInfo[idA - 1][3])
        sliceNumB = int(dataInfo[idB_all[0] - 1][3])

        sliceA_all = np.arange(sliceNumA)
        np.random.shuffle(sliceA_all)

        dataA_idxs = []
        dataB_idxs = []
        for i in sliceA_all[:self.batch_size]:
            dataA_idxs.append(i + 1)

            i_Bslice = round(i / (sliceNumA - 1.) * (sliceNumB - 1.))
            if i_Bslice >= randBd and i_Bslice <= sliceNumB - randBd - 1:
                i_Bslice = i_Bslice + np.random.randint((-1) * randBd, randBd)
            dataB_idxs.append(i_Bslice + 1)

        batch_files = list(zip(dataA_idxs, dataB_idxs))
        sample_image = [load_train_data(batch_file, idA, idB_all[0], is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_image).astype(np.float32)

        if idA == idB_all[0]:
            raise RuntimeError('MRI and CT should come from different subjects!')

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )

        name_inputB = r'dataset/T1/InputNorm_id{:0>2d}_slice{:0>3d}_T1.npy'.format(
            idB_all[0], int(dataB_idxs[0]))
        name_real_A = r'dataset/CT/InputNorm_id{:0>2d}_slice{:0>3d}_CT.npy'.format(
            idB_all[0], int(dataB_idxs[0]))
        inputB = np.load(name_inputB)
        real_A = np.load(name_real_A)
        inputB = (inputB + 1.) * 127.5
        real_A = (real_A + 1.) * 127.5

        pad_inputB_size = self.image_size0 - inputB.shape[0]
        pad_real_A_size = self.image_size0 - real_A.shape[0]

        inputB = np.pad(inputB, ((int(pad_inputB_size // 2), int(pad_inputB_size) - int(pad_inputB_size // 2)), (0, 0)),
                        mode='constant',
                        constant_values=0)
        real_A = np.pad(real_A, ((int(pad_real_A_size // 2), int(pad_real_A_size) - int(pad_real_A_size // 2)), (0, 0)),
                        mode='constant',
                        constant_values=0)

        name_inputA = r'dataset/CT/InputNorm_id{:0>2d}_slice{:0>3d}_CT.npy'.format(
            idA, int(dataA_idxs[0]))
        name_real_B = r'dataset/T1/InputNorm_id{:0>2d}_slice{:0>3d}_T1.npy'.format(
            idA, int(dataA_idxs[0]))
        inputA = np.load(name_inputA)
        real_B = np.load(name_real_B)
        inputA = (inputA + 1.) * 127.5
        real_B = (real_B + 1.) * 127.5

        pad_inputA_size = self.image_size0 - inputA.shape[0]
        pad_real_B_size = self.image_size0 - real_B.shape[0]

        inputA = np.pad(inputA, ((int(pad_inputA_size // 2), int(pad_inputA_size) - int(pad_inputA_size // 2)), (0, 0)),
                        mode='constant',
                        constant_values=0)
        real_B = np.pad(real_B, ((int(pad_real_B_size // 2), int(pad_real_B_size) - int(pad_real_B_size // 2)), (0, 0)),
                        mode='constant',
                        constant_values=0)


        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idA, idx))
        imageio.imwrite('./{}/A_{:02d}_{:02d}_{:04d}_input.jpg'.format(sample_dir, epoch, idA, idx),inputB)
        imageio.imwrite('./{}/A_{:02d}_{:02d}_{:04d}_areal.jpg'.format(sample_dir, epoch, idA, idx),real_A)

        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idA, idx))
        imageio.imwrite('./{}/B_{:02d}_{:02d}_{:04d}_input.jpg'.format(sample_dir, epoch, idA, idx),inputA)
        imageio.imwrite('./{}/B_{:02d}_{:02d}_{:04d}_areal.jpg'.format(sample_dir, epoch, idA, idx),real_B)


    def valid_CT(self, args):
        """valid cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        val_id_vec = [28, 29, 30]
        sample_files = r'dataset/InputNorm_{:0>2d}_T1.nii'
        gt_name = r'dataset/InputNorm_{:0>2d}_CT.nii'
        isCT = False
        namehd = 'synCT'
        input_val = 0
        output_val = -1000

        epochNum = 10
        epochVec = np.arange(epochNum)

        dataInfoFile = open(r'dataset/trainInfo.txt', 'r')
        sourceInLines = dataInfoFile.readlines()
        dataInfoFile.close()
        dataInfo = []
        for line in sourceInLines:
            temp1 = line.strip('\n')
            temp2 = temp1.split(' ')
            dataInfo.append(temp2)

        valid_dir = './valid'
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)

        for epoch in epochVec:

            self.load_valid(args.checkpoint_dir, epoch)

            for val_id in val_id_vec:

                sliceNum = int(dataInfo[val_id-1][3])
                sliceVec = np.arange(sliceNum)

                imgTemp = nib.load(sample_files.format(val_id))
                teResults = np.ones([imgTemp.shape[0], args.fine_size0, args.fine_size1], dtype=np.int16)*int(output_val)
                inputImage = np.ones([imgTemp.shape[0], args.fine_size0, args.fine_size1], dtype=np.int16)*int(input_val)
                gtImage = np.ones([imgTemp.shape[0], args.fine_size0, args.fine_size1], dtype=np.int16)*int(output_val)

                for iSlicet in sliceVec:

                    iSlice = iSlicet + int(dataInfo[val_id - 1][1]) - int(dataInfo[val_id - 1][5])
                    print('Processing image: id ' + str(val_id) + 'slice' + str(iSlicet))
                    sample_image = [load_test_data(sample_files.format(val_id), isCT, iSlice, args.fine_size0, args.fine_size1)]
                    sample_image = np.array(sample_image).astype(np.float32)
                    sample_image = sample_image.reshape([1, args.fine_size0, args.fine_size1, 1])

                    if epoch == 0:

                        gt_imageAll = nib.load(gt_name.format(val_id))
                        gt_image = gt_imageAll.get_data()[int(iSlice), :, :].astype('int16')

                        gtzm = gt_imageAll.get_header().get_zooms()
                        gtsz = gt_imageAll.shape
                        gt_resize_1 = args.fine_size1
                        gt_resize_0 = round(gtsz[1] * gtzm[1] * gt_resize_1 / (gtsz[2] * gtzm[2]))

                        gt_pad_size = int(args.fine_size0) - gt_resize_0
                        gt_image = transform.resize(gt_image, (gt_resize_0, gt_resize_1), preserve_range=True)
                        gt_image = np.pad(gt_image, ((int(gt_pad_size // 2), int(gt_pad_size) - int(gt_pad_size // 2)), (0, 0)), mode='constant',
                                     constant_values=output_val)

                        gtImage[int(iSlice), :, :] = np.array(gt_image).astype('int16')


                        input_imageAll = nib.load(sample_files.format(val_id))
                        input_image = input_imageAll.get_data()[int(iSlice), :, :].astype('int16')

                        inputzm = input_imageAll.get_header().get_zooms()
                        inputsz = input_imageAll.shape
                        input_resize_1 = args.fine_size1
                        input_resize_0 = round(inputsz[1] * inputzm[1] * input_resize_1 / (inputsz[2] * inputzm[2]))

                        input_pad_size = int(args.fine_size0) - input_resize_0
                        input_image = transform.resize(input_image, (input_resize_0, input_resize_1), preserve_range=True)
                        input_image = np.pad(input_image, ((int(input_pad_size // 2), int(input_pad_size) - int(input_pad_size // 2)), (0, 0)), mode='constant',
                                     constant_values=input_val)

                        inputImage[int(iSlice), :, :] = np.array(input_image).astype('int16')


                    fake_img = self.sess.run(self.testA, feed_dict={self.test_B: sample_image})
                    #fake_img_255 = np.exp((fake_img + 1.) * 4. * np.log(2)) - 1.
                    fake_img_255 = (fake_img + 1.) * 127.5
                    if isCT:
                        temp = fake_img_255 / 255. * (3500. - 0.) + 0.
                    else:
                        temp = fake_img_255 / 255. * (3500. + 1000.) - 1000.
                    teResults[int(iSlice), :, :] = np.array(temp).astype('int16').reshape([args.fine_size0,args.fine_size1])

                head_output = imgTemp.get_header()
                head_output.set_zooms([head_output.get_zooms()[0] * args.fine_size1 / (head_output.get_zooms()[2]*imgTemp.shape[2]),1.0,1.0])
                affine_output = imgTemp.affine
                affine_output[1][1] = np.sign(affine_output[1][1])
                affine_output[0][0] = np.sign(affine_output[0][0]) * head_output.get_zooms()[0]
                saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
                nib.save(saveResults, '{}/{}_{:0>2d}_epoch{}.nii'.format(valid_dir,namehd,val_id,epoch))

                if epoch == 0:

                    gtResults = nib.Nifti1Image(gtImage, affine_output, head_output)
                    gt_path = os.path.join(valid_dir, '{}'.format(os.path.basename(gt_name).format(val_id)))
                    nib.save(gtResults, gt_path)

                    inputResults = nib.Nifti1Image(inputImage, affine_output, head_output)
                    input_path = os.path.join(valid_dir, '{}'.format(os.path.basename(sample_files).format(val_id)))
                    nib.save(inputResults, input_path)


    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = r'dataset/InputNorm_T{:0>2d}_CT.nii'
            gt_name = r'dataset/InputNorm_T{:0>2d}_T1.nii'
            isCT = True
            namehd = 'synT1'
            input_val = -1000
            output_val = 0
        elif args.which_direction == 'BtoA':
            sample_files = r'dataset/InputNorm_T{:0>2d}_T1.nii'
            gt_name = r'dataset/InputNorm_T{:0>2d}_CT.nii'
            isCT = False
            namehd = 'synCT'
            input_val = 0
            output_val = -1000
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        epoch = 2
        self.load_valid(args.checkpoint_dir, epoch)

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        tedataSize = 15
        teIdVec = np.arange(tedataSize) + 1

        dataInfoFile = open(r'dataset/testInfo.txt', 'r')
        sourceInLines = dataInfoFile.readlines()
        dataInfoFile.close()
        dataInfo = []
        for line in sourceInLines:
            temp1 = line.strip('\n')
            temp2 = temp1.split(' ')
            dataInfo.append(temp2)

        for teId in teIdVec:

            sliceNum = int(dataInfo[teId-1][3])
            sliceVec = np.arange(sliceNum)

            imgTemp = nib.load(sample_files.format(teId))
            teResults = np.ones([imgTemp.shape[0], args.fine_size0, args.fine_size1], dtype=np.int16)*int(output_val)
            inputImage = np.ones([imgTemp.shape[0], args.fine_size0, args.fine_size1], dtype=np.int16)*int(input_val)
            gtImage = np.ones([imgTemp.shape[0], args.fine_size0, args.fine_size1], dtype=np.int16)*int(output_val)

            for iSlicet in sliceVec:

                iSlice = iSlicet + int(dataInfo[teId - 1][1]) - int(dataInfo[teId - 1][5])
                print('Processing image: id ' + str(teId) + 'slice' + str(iSlicet))
                sample_image = [load_test_data(sample_files.format(teId), isCT, iSlice,args.fine_size0, args.fine_size1)]
                sample_image = np.array(sample_image).astype(np.float32)
                sample_image = sample_image.reshape([1, args.fine_size0, args.fine_size1, 1])


                gt_imageAll = nib.load(gt_name.format(teId))
                gt_image = gt_imageAll.get_data()[int(iSlice), :, :].astype('int16')

                gtzm = gt_imageAll.get_header().get_zooms()
                gtsz = gt_imageAll.shape
                gt_resize_1 = args.fine_size1
                gt_resize_0 = round(gtsz[1] * gtzm[1] * gt_resize_1 / (gtsz[2] * gtzm[2]))

                gt_pad_size = int(args.fine_size0) - gt_resize_0
                gt_image = transform.resize(gt_image, (gt_resize_0, gt_resize_1), preserve_range=True)
                gt_image = np.pad(gt_image, ((int(gt_pad_size // 2), int(gt_pad_size) - int(gt_pad_size // 2)), (0, 0)), mode='constant',
                             constant_values=output_val)

                gtImage[int(iSlice), :, :] = np.array(gt_image).astype('int16')


                input_imageAll = nib.load(sample_files.format(teId))
                input_image = input_imageAll.get_data()[int(iSlice), :, :].astype('int16')

                inputzm = input_imageAll.get_header().get_zooms()
                inputsz = input_imageAll.shape
                input_resize_1 = args.fine_size1
                input_resize_0 = round(inputsz[1] * inputzm[1] * input_resize_1 / (inputsz[2] * inputzm[2]))

                input_pad_size = int(args.fine_size0) - input_resize_0
                input_image = transform.resize(input_image, (input_resize_0, input_resize_1), preserve_range=True)
                input_image = np.pad(input_image, ((int(input_pad_size // 2), int(input_pad_size) - int(input_pad_size // 2)), (0, 0)), mode='constant',
                             constant_values=input_val)

                inputImage[int(iSlice), :, :] = np.array(input_image).astype('int16')


                fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
                #fake_img_255 = np.exp((fake_img + 1.) * 4. * np.log(2)) - 1.
                fake_img_255 = (fake_img + 1.) * 127.5
                if isCT:
                    temp = fake_img_255 / 255. * (3500. - 0.) + 0.
                else:
                    temp = fake_img_255 / 255. * (3500. + 1000.) - 1000.
                teResults[int(iSlice), :, :] = np.array(temp).astype('int16').reshape([args.fine_size0,args.fine_size1])

            head_output = imgTemp.get_header()
            head_output.set_zooms([head_output.get_zooms()[0] * args.fine_size1 / (head_output.get_zooms()[2]*imgTemp.shape[2]),1.0,1.0])
            affine_output = imgTemp.affine
            affine_output[1][1] = np.sign(affine_output[1][1])
            affine_output[0][0] = np.sign(affine_output[0][0]) * head_output.get_zooms()[0]
            saveResults = nib.Nifti1Image(teResults, affine_output, head_output)
            nib.save(saveResults, '{}/{}_T{:0>2d}.nii'.format(args.test_dir,namehd,teId))

            gtResults = nib.Nifti1Image(gtImage, affine_output, head_output)
            gt_path = os.path.join(args.test_dir, '{}'.format(os.path.basename(gt_name).format(teId)))
            nib.save(gtResults, gt_path)

            inputResults = nib.Nifti1Image(inputImage, affine_output, head_output)
            input_path = os.path.join(args.test_dir, '{}'.format(os.path.basename(sample_files).format(teId)))
            nib.save(inputResults, input_path)
