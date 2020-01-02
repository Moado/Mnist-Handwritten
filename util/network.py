#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:44:33 2019

@author: Moad Hani

from torch import nn

class NeuralNetwork(nn.Module):
  
  def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
    
    super().__init__()
    
    self.input_size = input_size
    
    
    
    self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
    
    for input, output in zip(hidden_layers[:-1], hidden_layers[1:]):
      self.hidden_layers.extend([nn.Linear(input, output)])
    
    self.output_layer = nn.Linear(hidden_layers[-1], output_size)
    
    
    
    
    self.relu = nn.ReLU()
    
    self.drop = nn.Dropout(p=drop_p)
    
  def forward(self,x):
    x = x.view(-1, self.input_size)
    for layer in self.hidden_layers:
      x = layer(x)
      x = self.relu(x)
      x = self.drop(x)
      
    x = self.output_layer(x)
    
    return x
