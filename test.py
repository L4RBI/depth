import os
import glob
import argparse
import matplotlib
import numpy as np
from PIL import Image
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images, save_images
from matplotlib import pyplot as plt



def createDir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='../input/kitty-weights/kitti.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

path = "../input/thermal-images/kaist-cvpr15/images"
data = []
Listset=["set00","set01","set02","set06","set07","set08","set09","set10","set11"] 
for sets in Listset:
    for v in os.listdir(path + '/' + sets):
        _tmp = os.listdir(path + '/' + sets + "/" + v + '/visible')
        _tmp = [path + '/' + sets + "/" + v + '/visible/' + x for x in _tmp]
        data.extend(_tmp)
# Input images
print(len(data))
path = "./depth"
createDir(path)

for i in range(len(data)):
    temp = data[i].split("/")
    set = temp[-4]
    _set = os.path.join(path,set)
    createDir(_set)
    v = temp[-3]
    _v = os.path.join(_set,v)
    createDir(_v)
    image = temp[-1]
    _path = os.path.join(_v,image)
    inputs = np.asarray(Image.open( data[i] ), dtype=float) / 255
    outputs = predict(model, inputs)
    save_images(_path, outputs)
# Compute results
# outputs = predict(model, inputs)

# print(inputs)

# save_images("idk.jpg", outputs)
# # Display results
# viz = display_images(outputs.copy())
# plt.figure(figsize=(10,5))
# plt.imshow(viz)
# plt.savefig('test.png')
# plt.show()
