from .ops import*


class RenderingNet(object):

    def __init__(self, opt):

        self.opt = opt
        self.n_latent = opt.n_latent
        self.use_ext = opt.use_ext
        self.use_style_cont = opt.use_style_cont
        self.architecture = 'skip' if opt.use_skip else 'orig'

        self.ngf = opt.ngf
        self.max_ngf = opt.max_ngf

        self.input_nc = opt.input_nc
        self.label_nc = opt.label_nc
        self.output_nc = opt.output_nc
        self.resolution = opt.load_size
        self.n_mapping_layers = int(np.log2(self.resolution)) - self.n_latent
        self.latent_size = min(self.max_ngf, self.ngf * 2**self.n_mapping_layers)

        self.is_training = opt.isTrain
        self.reuse = {}

        print("========================================")
        print("Network Info:")
        print("Use Background Info at the Begining: ", self.use_ext)
        print("Use Style Modulation: ", self.use_style_cont)
        print("Architecture: ", self.architecture)
        print('Latent Space Resolution: {} x {}'.format(2**self.n_latent, 2**self.n_latent))
        print("========================================")

    def __call__(self, info, randomize_noise=True):

        em = info.pop("em")
        teeth_ref = info.pop("teeth_ref")
        teeth_oth = info.pop("teeth_oth")
        teeth_msk = info.pop("teeth_msk")

        if self.use_ext:
            # including the background info into the network. If training data size is small, not recommended.
            em += teeth_oth   # hard code

        with tf.variable_scope("RenderingNet"):
            p_pos, z_pos, latents = self.TextureMapping(teeth_ref, scope="TextureMapping")
            teethSyn = self.RenderingNet(z_pos if self.is_training else p_pos, latents, em, teeth_oth, teeth_msk,
                                         scope="RenderingNet", randomize_noise=randomize_noise)

            return teethSyn, p_pos

    def TextureMapping(self, x, scope):
        if scope not in self.reuse:
            self.reuse[scope] = None

        if self.reuse[scope]:
            print(f"{self.reuse[scope]} Module reused")

        with tf.variable_scope(scope, reuse=self.reuse[scope]):
            dlatents = mapping(x,
                               num_channels=self.input_nc,
                               resolution=self.resolution,
                               fmap_min=self.ngf,
                               fmap_max=self.max_ngf,
                               architecture="orig",
                               last_layer=self.n_latent)

            p_pos, z_pos = latent_sample(dlatents[-1], self.latent_size)
            latents = None
            if self.use_style_cont:
                latents = G_mapping(z_pos if self.is_training else p_pos,   # not need resampling during testing
                                    latent_size=self.latent_size,
                                    dlatent_size=self.latent_size,
                                    mapping_layers=2,
                                    mapping_fmaps=self.latent_size)
            self.reuse[scope] = True
            return p_pos, z_pos, latents

    def RenderingNet(self, dlatents, latents_in, geo_maps, teeth_1_o=None, mask_1=None, scope=None, randomize_noise=True):
        if scope not in self.reuse:
            self.reuse[scope] = None

        if self.reuse[scope]:
            print(f"{self.reuse[scope]} Module reused")

        with tf.variable_scope(scope, reuse=self.reuse[scope]):

            # down sample geometry maps
            with tf.variable_scope("Geometry"):
                geometry_in = mapping(geo_maps,
                                      num_channels=self.label_nc,
                                      resolution=self.resolution,
                                      fmap_min=max(self.ngf//4, 4), # reduce channels for teeth geometry for simplicity.
                                      fmap_max=max(self.max_ngf//4, 16),
                                      architecture="orig",
                                      last_layer=self.n_latent)

                if not self.use_ext:
                    # for learnable natural blending within the last two layers
                    geometry_in[0] = tf.concat([geometry_in[0], teeth_1_o], axis=1)
                    geometry_in[1] = tf.concat([geometry_in[1], downsample_2d(teeth_1_o)], axis=1)

            # rendering teeth coonditional on geometry map (geometry_in) & appearance code (dlatents and latents_in)
            with tf.variable_scope("Rendering"):
                outs = G_rendering(dlatents,
                                   latents_in=latents_in,
                                   geometry_in=geometry_in,
                                   num_channels=self.output_nc,
                                   randomize_noise=randomize_noise,
                                   fmap_max=self.max_ngf,
                                   architecture=self.architecture)

                outs = tf.nn.sigmoid(outs) * 1.2 - 0.1  # relax a little bit

            if teeth_1_o is not None and mask_1 is not None:
                outs = outs * mask_1 + teeth_1_o

            self.reuse[scope] = True
            return outs


class Discriminator(object):

    def __init__(self, name, opt):

        self.name = name
        self.opt = opt
        self.input_nc = opt.input_nc
        self.resolution = opt.load_size

        self.ndf = opt.ndf
        self.max_ndf = opt.max_ndf

        self.reuse = None

    def __call__(self, x):

        if self.reuse:
            print(f"{self.name} Module reused")

        with tf.variable_scope(self.name, reuse=self.reuse):
            scores = D(x,
                       num_channels=self.input_nc,
                       resolution=self.resolution,
                       fmap_min=self.ndf,
                       fmap_max=self.max_ndf)

        self.reuse = True

        return scores