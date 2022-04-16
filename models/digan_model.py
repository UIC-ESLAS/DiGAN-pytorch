import torch
import itertools
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util
from data.base_dataset import get_transform
import cv2



class DiGANModel(BaseModel):
    """
    This class implements the DiGAN model, for learning object transfiguration in field of image-to-image translation.

    The model training requires '--dataset_mode segment' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator,
    and a least-square GANs objective ('--gan_mode lsgan').
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For DiGAN, in addition to GAN losses, 
        we introduce lambda_Y, lambda_X, lambda_cycle, lambda_identity, lambda_feature and lambda_segmentation for the following losses.
        X (source domain), Y (target domain).
        Generators: G_Y: X -> Y; G_X: Y -> X.
        Discriminators: D_X: X vs. G_X(Y) vs. G_X(X); D_Y: Y vs. G_Y(X) vs. G_Y(Y).
        Forward cycle loss:  lambda_cycle * lambda_Y * ||G_X(G_Y(X)) - X||
        Backward cycle loss: lambda_cycle * lambda_X * ||G_Y(G_X(Y)) - Y||
        Identity loss: lambda_identity * (||G_Y(Y) - Y|| * lambda_Y + ||G_X(X) - X|| * lambda_X)
        Feature loss: lambda_feature * (lambda_X * ||f(G_X(Y)) - f(Y)|| + lambda_Y * ||f(G_Y(X)) - f(X)||)
        Segmentation loss: lambda_segmentation * (lambda_X * ||s(G_X(Y)) - s(Y)|| + lambda_Y * ||s(G_Y(X)) - s(X)||)
        """
        parser.set_defaults(no_dropout=True)  # default DiGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_Y', type=float, default=15.0, help='weight for cycle loss (X -> Y -> X)')
            parser.add_argument('--lambda_X', type=float, default=10.0, help='weight for cycle loss (Y -> X -> Y)')
            parser.add_argument('--lambda_cycle', type=float, default=1, help='use cycle consistency.')
            parser.add_argument('--lambda_identity', type=float, default=1, help='use identity mapping.')
            parser.add_argument('--lambda_f', type=float, default=0.5, help='use feature consistency.')
            parser.add_argument('--lambda_s', type=float, default=0.5, help='use segmentation consistency.')

        return parser

    def __init__(self, opt):
        """Initialize the DiGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_Y', 'G_Y', 'D_X', 'G_X', 'cycle_X', 'cycle_Y', 'idt', 'f', 's']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_Y = ['real_X', 'fake_Y', 'rec_X', 'idt_X']
        visual_names_X = ['real_Y', 'fake_X', 'rec_Y', 'idt_Y']

        self.visual_names = visual_names_Y + visual_names_X  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_X', 'G_Y', 'D_X', 'D_Y']
        else:  # during test time, only load Gs
            self.model_names = ['G_X', 'G_Y']

        # define networks (both Generators and discriminators)
        self.netG_Y = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_X = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.opt = opt

        if self.isTrain:  # define discriminators
            self.netD_Y = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_X = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_X_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_Y_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.idt_X_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.idt_Y_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionF = torch.nn.L1Loss()
            self.criterionS = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_Y.parameters(), self.netG_X.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Y.parameters(), self.netD_X.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def setpredictor(self,predictor):
        # set predictor for Detectron2 mask-rcnn
        self.predictor = predictor

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain X and domain Y.
        """
        XtoY = self.opt.direction == 'AtoB'

        # use original image and segmented image to create condiational imput
        self.real_X = input['A' if XtoY else 'B'].to(self.device)
        self.real_Xmask = input['Amask' if XtoY else 'Bmask'].to(self.device)
        self.real_Xplus = torch.cat(tensors=(self.real_X, self.real_Xmask), dim=1).to(self.device)
        self.image_paths = input['A_paths' if XtoY else 'B_paths']
        
        if self.isTrain:
            self.real_Y = input['B' if XtoY else 'A'].to(self.device)
            self.real_Ymask = input['Bmask' if XtoY else 'Amask'].to(self.device)
            self.real_Yplus = torch.cat(tensors=(self.real_Y, self.real_Ymask), dim=1).to(self.device)
            
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Y = self.netG_Y(self.real_Xplus)  # G_Y(X)
        # create conditional input fake_Yplus
        fake_Y = util.tensor2im(self.fake_Y)
        fake_Ymask = networks.segmentation(self.predictor, fake_Y)
        transform = get_transform(self.opt, grayscale=True)
        self.fake_Ymask = transform(fake_Ymask).reshape(-1,1,256,256).to(self.device)
        self.fake_Yplus = torch.cat(tensors=(self.fake_Y, self.fake_Ymask), dim=1).to(self.device)
        self.rec_X = self.netG_X(self.fake_Yplus)   # G_X(G_Y(X))
        self.idt_X = self.netG_X(self.real_Xplus)  # G_X(X)
        # create conditional input idt_Xplus
        idt_X = util.tensor2im(self.idt_X)
        idt_Xmask = networks.segmentation(self.predictor, idt_X)
        self.idt_Xmask = transform(idt_Xmask).reshape(-1,1,256,256).to(self.device)
        self.idt_Xplus = torch.cat(tensors=(self.idt_X, self.idt_Xmask), dim=1).to(self.device)
        if self.opt.gpu_ids==[]:
            self.featureX1 = self.netG_Y.feature_extraction(self.real_Xplus)
            self.featureX2 = self.netG_Y.feature_extraction(self.fake_Yplus)
        else:
            self.featureX1 = self.netG_Y.module.feature_extraction(self.real_Xplus)
            self.featureX2 = self.netG_Y.module.feature_extraction(self.fake_Yplus)


        if self.isTrain:
            self.fake_X = self.netG_X(self.real_Yplus)  # G_X(Y)
            # create conditional input fake_Xplus
            fake_X = util.tensor2im(self.fake_X)
            fake_Xmask = networks.segmentation(self.predictor, fake_X)
            self.fake_Xmask = transform(fake_Xmask).reshape(-1,1,256,256).to(self.device)
            self.fake_Xplus = torch.cat(tensors=(self.fake_X, self.fake_Xmask), dim=1).to(self.device)
            self.rec_Y = self.netG_Y(self.fake_Xplus)   # G_Y(G_X(Y))
            self.idt_Y = self.netG_Y(self.real_Yplus)   # G_Y(Y)
            # create conditional input idt_Yplus
            idt_Y = util.tensor2im(self.idt_Y)
            idt_Ymask = networks.segmentation(self.predictor, idt_Y)
            self.idt_Ymask = transform(idt_Ymask).reshape(-1,1,256,256).to(self.device)
            self.idt_Yplus = torch.cat(tensors=(self.idt_Y, self.idt_Ymask), dim=1).to(self.device)
            if self.opt.gpu_ids==[]:
                self.featureY1 = self.netG_X.feature_extraction(self.real_Yplus)
                self.featureY2 = self.netG_X.feature_extraction(self.fake_Xplus)
            else:
                self.featureY1 = self.netG_X.module.feature_extraction(self.real_Yplus)
                self.featureY2 = self.netG_X.module.feature_extraction(self.fake_Xplus)

    def backward_D_basic(self, netD, real, fake, idt):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- fake images generated by a generator
            idt (tensor array)  -- identity images generated by another generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_idt = netD(idt.detach())
        loss_D_idt = self.criterionGAN(pred_idt, False)
        # Combined loss and calculate gradients
        loss_D = loss_D_real * 0.5 + (loss_D_fake + loss_D_idt) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_Y(self):
        """Calculate GAN loss for discriminator D_Y"""
        fake_Yplus = self.fake_Y_pool.query(self.fake_Yplus)
        idt_Yplus = self.idt_Y_pool.query(self.idt_Yplus)
        self.loss_D_Y = self.backward_D_basic(self.netD_Y, self.real_Yplus, fake_Yplus, idt_Yplus)

    def backward_D_X(self):
        """Calculate GAN loss for discriminator D_X"""
        fake_Xplus = self.fake_X_pool.query(self.fake_Xplus)
        idt_Xplus = self.idt_X_pool.query(self.idt_Xplus)
        self.loss_D_X = self.backward_D_basic(self.netD_X, self.real_Xplus, fake_Xplus, idt_Xplus)

    def backward_G(self):
        """Calculate the loss for generators G_Y and G_X"""
        lambda_X = self.opt.lambda_X
        lambda_Y = self.opt.lambda_Y
        lambda_idt = self.opt.lambda_identity
        lambda_cycle = self.opt.lambda_cycle
        lambda_f = self.opt.lambda_f
        lambda_s = self.opt.lambda_s
        

        # GAN loss G_Y
        self.loss_G_Y = self.criterionGAN(self.netD_Y(self.fake_Yplus), True) + self.criterionGAN(self.netD_Y(self.idt_Yplus), True)
        # GAN loss G_X
        self.loss_G_X = self.criterionGAN(self.netD_X(self.fake_Xplus), True) + self.criterionGAN(self.netD_X(self.idt_Xplus), True)
        # Forward cycle loss || G_Y(G_X(Y)) - Y||
        self.loss_cycle_Y = lambda_cycle * self.criterionCycle(self.rec_Y, self.real_Y) * lambda_X
        # Backward cycle loss || G_X(G_Y(X)) - X||
        self.loss_cycle_X = lambda_cycle * self.criterionCycle(self.rec_X, self.real_X) * lambda_Y
        # Identity loss
        # G_Y should be identity if real_Y is fed: ||G_Y(Y) - Y||
        loss_idt_Y = self.criterionIdt(self.idt_Y, self.real_Y) * lambda_Y * lambda_idt
        # G_X should be identity if real_X is fed: ||G_X(X) - X||
        loss_idt_X = self.criterionIdt(self.idt_X, self.real_X) * lambda_X * lambda_idt
        self.loss_idt = loss_idt_X + loss_idt_Y
        # Feature loss
        self.loss_f = lambda_f * (self.criterionF(self.featureX1, self.featureX2) * lambda_Y + self.criterionF(self.featureY1, self.featureY2) * lambda_X)
        # segmentation loss
        self.loss_s = lambda_s * (self.criterionS(self.real_Xmask, self.fake_Ymask) * lambda_Y + self.criterionS(self.real_Ymask, self.fake_Xmask) * lambda_X)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_Y + self.loss_G_X + self.loss_cycle_Y + self.loss_cycle_X + self.loss_idt + self.loss_f + self.loss_s
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_Y and G_X
        self.set_requires_grad([self.netD_Y, self.netD_X], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_Y and G_X's gradients to zero
        self.backward_G()             # calculate gradients for G_Y and G_X
        self.optimizer_G.step()       # update G_Y and G_X's weights
        # D_Y and D_X
        self.set_requires_grad([self.netD_Y, self.netD_X], True)
        self.optimizer_D.zero_grad()   # set D_Y and D_X's gradients to zero
        self.backward_D_Y()      # calculate gradients for D_Y
        self.backward_D_X()      # calculate graidents for D_X
        self.optimizer_D.step()  # update D_Y and D_X's weights

    def test(self):
        """Only generate fakeY from real X, you can change the self.visual_names if you want to output other kinds of images"""
        self.visual_names = ['real_X', 'real_Xmask', 'fake_Y']
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
