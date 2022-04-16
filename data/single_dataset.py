from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg



class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating testing results from realA to fakeB.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.transform = get_transform(opt)
        self.transform_mask = get_transform(opt, grayscale=True)

        # initialize Detectron2 mask-rcnn
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        Amask = self.mask(cv2.imread(A_path)).convert('L')
        A = self.transform(A_img)
        Amask = self.transform_mask(Amask)
        return {'A': A, 'A_paths': A_path, 'Amask': Amask}

    def mask(self,img):
        """Return the segmented result for the input image"""
        outputs = self.predictor(img)
        pred_masks = outputs["instances"].pred_masks.cpu().data.numpy()
        masked_image = np.zeros((img.shape[0], img.shape[1]))
        for c in range(pred_masks.shape[0]):
            masked_image = np.add(masked_image, pred_masks[c, :, :]*255)
        masked_image = np.ones((img.shape[0], img.shape[1]))*255 - masked_image
        masked_image = Image.fromarray(np.uint8(masked_image))
        return masked_image

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
