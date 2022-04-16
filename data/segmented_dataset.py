import os
from util import util
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class SegmentedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets (use segmented_prepro.py to get masks of original images in advance).

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    And two two directories to host the mask of training images from domain A '/path/to/data/trainAmask'
    and from domain B '/path/to/data/trainBmask' respectively.
    You need to get the masks of the training images by segmented_prepro.py firstly.

    You can train the model with the dataset flag '--dataroot /path/to/data'.

    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    For the testing data, you don't need to prepare the masks.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_Amask = os.path.join(opt.dataroot, opt.phase + 'Amask') # create a path '/path/to/data/trainAmask'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_Bmask = os.path.join(opt.dataroot, opt.phase + 'Bmask') # create a path '/path/to/data/trainBmask'


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.Amask_paths = sorted(make_dataset(self.dir_Amask, opt.max_dataset_size))  # load images from '/path/to/data/trainAmask'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.Bmask_paths = sorted(make_dataset(self.dir_Bmask, opt.max_dataset_size))  # load images from '/path/to/data/trainBmask'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.transform_A = get_transform(self.opt)
        self.transform_Amask = get_transform(self.opt, grayscale=True)
        self.transform_B = get_transform(self.opt)
        self.transform_Bmask = get_transform(self.opt, grayscale=True)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        Amask_path = self.Amask_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        Bmask_path = self.Bmask_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        Amask_img = Image.open(Amask_path).convert("L")
        B_img = Image.open(B_path).convert("RGB")
        Bmask_img = Image.open(Bmask_path).convert("L")

        # apply image transformation
        A = self.transform_A(A_img)
        Amask = self.transform_Amask(Amask_img)
        B = self.transform_B(B_img)
        Bmask = self.transform_Bmask(Bmask_img)

        return {'A': A,'Amask': Amask, 'B': B, 'Bmask': Bmask, 'A_paths': A_path,'Amask_paths': Amask_path, 'B_paths': B_path,'Bmask_paths': Bmask_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
