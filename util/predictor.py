#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:23:48 2019

@author: Moad Hani
"""

import torch

def predict(image, model, topk=1):
    """
    Function:
        Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        image -- path of image file
        model -- trained model that will make the predictoin
        topk -- range of class to display
    """
    with torch.no_grad():
        model.eval() 
        image = torch.FloatTensor(image)
        y_ = model.forward(image[None])
        ps = torch.exp(y_)
        top_p, top_class = ps.topk(topk, dim=1)
        return top_p, top_class

	