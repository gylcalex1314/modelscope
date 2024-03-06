# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union


import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.models.cv.image_depth_prediction.models import infer  
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import depth_to_color
from modelscope.utils.logger import get_logger

logger = get_logger()

@PIPELINES.register_module(
    Tasks.image_depth_prediction, module_name=Pipelines.image_depth_prediction)
class ImageDepthPredictionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image depth prediction pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('Image Depth Prediction model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        print('start preprocess')
        image = LoadImage.convert_to_ndarray(input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))

        image = image / 255.

        transform = infer.ToTensor()
        image = transform(image).unsqueeze(0).float().to('cuda:0')
        data = {'images': image}
        print('finish preprocess')

        return data
    
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        print('start infer')
        results = self.model.inference(input)
        print('finish infer')
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        print('start postprocess')
        results = self.model.postprocess(inputs)

        depth_image = results[OutputKeys.DEPTHS]
        depths_color = depth_to_color(depth_image.squeeze())
        outputs = {
            OutputKeys.DEPTHS: depth_image,
            OutputKeys.DEPTHS_COLOR: depths_color
        }
        print('finish postprocess')

        return outputs
