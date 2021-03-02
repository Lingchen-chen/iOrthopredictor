from models.module import *
from models.loss import *
from data.data_loader import *
from util.image_util import *
from extern.metrics.FID import calculate_fid_given_paths
import os


class BaseSolver:

    def __init__(self, sess, opt):

        self.opt = opt
        self.sess = sess
        self.image_height = opt.load_size
        self.image_width = opt.load_size
        self.batch_size = opt.batch_size
        self.is_training = opt.isTrain

    def load(self, saver, checkpoint_dir):
        print(" [*] Reading latest checkpoint from folder %s." % (checkpoint_dir))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print("Load Failure!")
            return False


class TSynNetSolver(BaseSolver):

    def __init__(self, sess, opt):
        super().__init__(sess, opt)

        # Rendering Net params
        self.use_gan = opt.use_gan
        self.use_ext = opt.use_ext
        self.use_skip = opt.use_skip
        self.use_style_cont = opt.use_style_cont
        self.n_latent = opt.n_latent
        self.edge_dim = len(opt.teeth_label_name)

        # define dir for the model
        self.root_dir = os.path.join(opt.expr_dir, opt.name)
        self.train_log_dir = os.path.join(self.root_dir, 'logs', 'train')
        self.checkpoint_dir = os.path.join(self.root_dir, "checkpoint")
        self.checkpoint_dir_best_FID = os.path.join(self.root_dir, "checkpoint_best_FID")

        util.mkdirs(self.train_log_dir)
        util.mkdirs(self.checkpoint_dir)
        util.mkdirs(self.checkpoint_dir_best_FID)

        # set data augmentor and loader for training
        if self.is_training:
            # data loader params
            self.result_dir = opt.result_dir
            self.train_data_dir = opt.train_data_dir
            self.val_data_dir = opt.val_data_dir

            # training params
            self.lr_decay_begin = opt.niter
            self.lr_decay_end = opt.niter + opt.niter_decay
            self.kl_weight = opt.kl_weight
            self.content_weight = opt.content_weight

            # saving params
            self.display_iter = opt.display_freq
            self.save_iter = opt.save_latest_freq
            self.learning_rate = opt.lr

            self.vgg_checkpoint_dir = opt.vgg_checkpoint_dir
            self.train_loader = data_loader(self.train_data_dir, opt)
            self.val_loader = data_loader(self.val_data_dir, opt)
            self.data_augmentor = Augmentor(img_hsv_prob=0.0,
                                            horizon_flip=True,
                                            img_rotate_prob=0.8,
                                            img_rotate_range=0.2,
                                            label_clip=True)
            self.D = Discriminator("RenderingNetDiscriminator", opt)
        else:
            self.test_data_dir = opt.test_data_dir
            self.test_loader = data_loader_test(self.test_data_dir, opt)

        self.RenderingNet = RenderingNet(opt)

        self.build_graph()

    def create_info(self, x_raw, e_raw, m_raw):

        # we can override x_, e_, m_ to ignore the data augmentation
        self.x_ = x_raw
        self.e_ = e_raw
        self.m_ = m_raw

        if self.is_training:
            print("in training")
            self.x_, l_ = self.data_augmentor(self.x_, tf.concat([self.e_, self.m_], axis=-1))
            self.e_, self.m_ = tf.split(l_, [self.edge_dim, 1], axis=-1)

        # data pre-processing HWC -> CHW
        em = tf.transpose(tf.stop_gradient(tf.concat([self.e_, self.m_], axis=-1)), (0, 3, 1, 2))
        xm = tf.transpose(tf.stop_gradient(self.x_ * self.m_), (0, 3, 1, 2))
        xm_o = tf.transpose(tf.stop_gradient(self.x_ * (1 - self.m_)), (0, 3, 1, 2))
        x_aug = tf.transpose(self.x_, (0, 3, 1, 2))
        m_aug = tf.transpose(self.m_, (0, 3, 1, 2))

        return xm, em, x_aug, m_aug, xm_o

    def build_graph(self):

        self.initialized = False

        # set the input tensors
        self.x_raw = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_height, self.image_width, 3])
        self.e_raw = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_height, self.image_width, self.edge_dim])
        self.m_raw = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_height, self.image_width, 1])

        # input preparation
        self.xm, self.em, self.x_aug, self.m_aug, self.xm_o = self.create_info(self.x_raw, self.e_raw, self.m_raw)
        self.info = {}
        self.info["em"] = self.em
        self.info["teeth_ref"] = self.xm
        self.info["teeth_oth"] = self.xm_o
        self.info["teeth_msk"] = self.m_aug

        # network graph
        self.r_all, self.p_pos = self.RenderingNet(self.info.copy(), randomize_noise=self.use_gan)
        self.rm = self.r_all * self.m_aug

        if not self.is_training:
            saver = tf.train.Saver(slim.get_variables("RenderingNet"))
            if not self.load(saver, self.checkpoint_dir_best_FID if self.opt.use_best_FID else self.checkpoint_dir):
                return False

        self.initialized = True

        return True

    def train(self):

        if not self.initialized:
            print("Initialization failed.")
            return

        with tf.name_scope(name="RenderingNet"):

            # some definitions
            global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
            learning_rate = util.makeLinearWeight(global_step,
                                                  self.lr_decay_begin, self.lr_decay_end,
                                                  self.learning_rate, 0,
                                                  0, self.learning_rate)

            # -------------------------------- build network ------------------------------ #
            # kl divergence loss
            with tf.name_scope("Divergence_Loss"):
                kl_divergence_loss = normal_kl_loss(self.p_pos)
                kl_divergence_loss *= self.kl_weight

            # self reconstruction loss
            with tf.name_scope("Reconstruction_Loss"):
                content_weights = {"relu_0": 1.0 / tf.maximum(1., tf.reduce_sum(self.m_aug, axis=[1, 2, 3])),
                                   "relu1_2": self.content_weight / tf.maximum(1., tf.reduce_sum(self.m_aug, axis=[1, 2, 3])),
                                   "relu2_2": self.content_weight / tf.maximum(1., tf.reduce_sum(self.m_aug, axis=[1, 2, 3]) / 4.),
                                   "relu3_4": self.content_weight / tf.maximum(1., tf.reduce_sum(self.m_aug, axis=[1, 2, 3]) / 16.)}
                x_aug_ = tf.transpose(self.x_aug, (0, 2, 3, 1))
                r_all_ = tf.transpose(self.r_all, (0, 2, 3, 1))
                total_content_loss, l1_loss = reconstruct_loss(x_aug_, r_all_, content_weights)

            # gan loss
            D_loss = G_loss = D_reg = 0.
            with tf.name_scope("GAN_Loss"):
                if self.use_gan:
                    D_loss, D_reg, G_loss, G_reg = adversarial_loss(self.D, self.xm, tf.stop_gradient(self.rm), self.m_aug)

            G_total_loss = kl_divergence_loss + (total_content_loss + l1_loss) + G_loss
            D_total_loss = D_loss + D_reg

            D_var = slim.get_variables("RenderingNetDiscriminator/")
            G_var = slim.get_variables("RenderingNet/")

            with tf.control_dependencies([learning_rate]):
                if self.use_gan:
                    D_train_step = tf.train.AdamOptimizer(learning_rate).minimize(D_total_loss, var_list=D_var)
                G_train_step = tf.train.AdamOptimizer(learning_rate).minimize(G_total_loss, var_list=G_var, global_step=global_step)

            # record summary
            loss_summary = []
            loss_summary.append(tf.summary.scalar("kl_divergence_loss", kl_divergence_loss))
            loss_summary.append(tf.summary.scalar("total_content_loss", total_content_loss))
            loss_summary.append(tf.summary.scalar("l1_loss", l1_loss))
            loss_summary.append(tf.summary.scalar("G_loss", G_loss))
            loss_summary.append(tf.summary.scalar("D_loss", D_loss))
            loss_summary.append(tf.summary.scalar("D_reg", D_reg))
            loss_summary.append(tf.summary.scalar("G_total_loss", G_total_loss))
            loss_summary.append(tf.summary.scalar("D_total_loss", D_total_loss))
            loss_summary.append(tf.summary.scalar("learning_rate", learning_rate))

            image_summary = []
            def add_image_summary(name, img):
                image_summary.append(tf.summary.image(name, util.tensor2im(img, 1)))
            add_image_summary("xm", self.xm)
            add_image_summary("gt", self.x_aug)
            add_image_summary("em", self.em+self.xm_o)  # hard code
            add_image_summary("re_all", self.r_all)
            add_image_summary("xm_o", self.xm_o)

            step_summary = tf.summary.merge(loss_summary + image_summary)

            trainWriter = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)

            # Initialize
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            save_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
            save_path_best_fid = os.path.join(self.checkpoint_dir_best_FID, 'model.ckpt')

            if not self.opt.continue_train:
                saver_vgg = tf.train.Saver(slim.get_variables('vgg_19'))
                if not self.load(saver_vgg, self.vgg_checkpoint_dir):
                    print("loading vgg failed, training is wrong")
            else:
                if not self.load(saver, self.checkpoint_dir):
                    print("loading latest failed, training from scratch")

            FID = 1000
            for iteration in range(self.sess.run(global_step) + 1, self.lr_decay_end + 1):

                imgs, edges, mmask = self.train_loader.get_one_batch_data()
                feed_dict = {self.x_raw: imgs, self.e_raw: edges, self.m_raw: mmask}
                if self.use_gan:
                    self.sess.run(D_train_step, feed_dict=feed_dict)
                self.sess.run(G_train_step, feed_dict=feed_dict)

                if iteration % self.display_iter == 0:
                    imgs, edges, mmask = self.train_loader.get_one_batch_data()
                    feed_dict = {self.x_raw: imgs, self.e_raw: edges, self.m_raw: mmask}

                    summaries = self.sess.run(step_summary, feed_dict=feed_dict)
                    trainWriter.add_summary(summaries, global_step=iteration)

                if iteration % self.save_iter == 0:
                    if iteration > self.lr_decay_begin:
                        fid = self.FID_test()
                        if fid < FID:
                            print(f"previous best FID: {FID}, current best FID: {fid}")
                            FID = fid
                            saver.save(self.sess, save_path_best_fid, global_step=iteration)

                    saver.save(self.sess, save_path, global_step=iteration)

            saver.save(self.sess, save_path, global_step=self.lr_decay_end)

    def FID_test(self):

        if not self.is_training:
            return None

        real_dir = os.path.join(self.result_dir, "real")
        fake_dir = os.path.join(self.result_dir, "fake")
        util.mkdirs(real_dir)
        util.mkdirs(fake_dir)

        for i in range(self.val_loader.get_iters()):
            imgs, edges, mmask = self.val_loader.get_one_batch_data()
            feed_dict = {self.x_: imgs, self.e_: edges, self.m_: mmask}  # ignore the data augmentation
            real, fake = self.sess.run([self.x_, self.r_all], feed_dict=feed_dict)
            real, fake = util.numpy2im(real), util.numpy2im(fake)   # do no use tensor2im

            for j, img in enumerate(real):
                id = i * self.batch_size + j
                cv2.imwrite(os.path.join(real_dir, f"{id}.jpg"), img)

            for j, img in enumerate(fake):
                id = i * self.batch_size + j
                cv2.imwrite(os.path.join(fake_dir, f"{id}.jpg"), img)

        with tf.Graph().as_default():
            FID = calculate_fid_given_paths([real_dir, fake_dir])

        return FID

    def test(self):

        if not self.initialized:
            print("Initialization failed.")
            return

        print('Doing %d cases' % len(self.test_loader))
        for case in self.test_loader.get_one_case():
            img, edges, mmask, save_path = case
            save_path = os.path.join(save_path, self.opt.test_save_dir)
            util.mkdirs(save_path)
            for step, edge in edges.items():
                save_file = os.path.join(save_path, f"{step}.jpg")
                feed_dict = {self.x_: img, self.e_: edge, self.m_: mmask}  # ignore the data augmentation
                result = self.sess.run(self.r_all, feed_dict=feed_dict)
                result = util.numpy2im(result)
                cv2.imwrite(save_file, result)



