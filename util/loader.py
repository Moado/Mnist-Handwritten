#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:45:10 2019

@author: Moad Hani
"""
import torch 
from .network import NeuralNetwork


def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath, map_location={'cuda:0': 'cpu'})
    model = NeuralNetwork(input_size=checkpoint['input_size'],
                             output_size=checkpoint['output_size'],
                             hidden_layers=checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
