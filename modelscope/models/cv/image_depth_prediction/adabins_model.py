# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.preprocessors import LoadImage
from modelscope.models.cv.image_depth_prediction.models import unet_adaptive_bins, model_io, infer
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

@MODELS.register_module(
    Tasks.image_depth_prediction, module_name=Models.adabins_depth_prediction)
class DepthPrediction(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        
        model = unet_adaptive_bins.UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=10)
        model, _, _ = model_io.load_checkpoint(os.path.join(model_dir, 'pytorch_model.pt'), model)
        
        model.eval()
        model.to('cuda:0')
        
        self.model = model
        
        logger.info('Depth prediction model, pipeline init')

    def forward(self, Inputs):
        bin_centers, pred = self.model(Inputs['images'])
        Inputs['depth'] = pred
    
        return Inputs
    
    def postprocess(self, Inputs):
        inferHelper = infer.InferenceHelper()
        
        depth_result = inferHelper.postprocess(Inputs['depth'])
        
        # plt.imshow(depth_result.squeeze(), cmap='magma_r')
        # plt.show()
        # cv2.waitKey(0)
        
        # print('s2')

        results = {OutputKeys.DEPTHS: depth_result}
        return results
        
    def inference(self, data):
        results = self.forward(data)

        return results

