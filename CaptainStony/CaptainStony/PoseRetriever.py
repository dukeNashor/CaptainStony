import numpy as np
from PIL import Image

import logging

class PoseRetriever(object):

    def __init__(self, load_image_into_memory = False):
        super(self.__class__, self).__init__()
        logging.info("PoseRetriever:initializing with load_image_into_memory = {}".format(load_image_into_memory))

        self.load_image_into_memory = load_image_into_memory
        self.image_dict = {}

    def Load3DPWProcessedData(self, processed_dir):
        logging.info("PoseRetriever:Load3DPWProcessedData()")
        pass

    def Query(self, pose):
        logging.info("PoseRetriever:Query()")
        pass

    def GetImage(self, key):
        if key not in self.image_dict:
            return None
        elif not self.load_image_into_memory:
            return self.image_dict[key]
        else:
            return np.asarray(Image.open(self.image_dict[key]))