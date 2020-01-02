import socketio
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import torch

from util.loader import load_checkpoint
from util.predictor import predict

net = load_checkpoint('checkpoint.pth')

sio = socketio.Client()


@sio.on('connect')
def on_connect():
    print('connection established')
    
   
@sio.on('news')
def on_message(data):
    print('event news returned ', data)
    
@sio.on('user_input')
def on_image(data):
    encoded_image = data.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(BytesIO(decoded_image))
    img = np.asarray(img, dtype='uint8')  
    img  = cv2.resize(img,(28,28))
    img = 255 - img
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/(255/2)-1
    img = torch.FloatTensor(img)
    img = img.reshape(1,28,28)
    
    _,c =predict(img,net,1)   
    sio.emit('result', {'response': int(c)})
    print('Image received with data = ',int(c))


@sio.on('disconnect')
def on_disconnect():
    print('disconnected from server')

sio.connect('http://localhost:3000')
sio.wait()


