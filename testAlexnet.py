import caffe
import numpy as np
import time
import cProfile

caffe.set_mode_cpu()
model_def = 'deploy.prototxt'
mdoel_weights = 'bvlc_alexnet.caffemodel'
img_file = 'test.jpeg'
lables_file = 'synset_words.txt'

net = caffe.Net(model_def, mdoel_weights, caffe.TEST)

with open(lables_file, 'r') as f:
    lines = f.readlines()
lables = []
for i in lines:
    lables.append(i.strip('\n').split(',')[1:])

mu = np.load('./ilsvrc_2012_mean.npy')
mu = mu.mean(1)
mu = mu.mean(1)

transforms = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transforms.set_transpose('data', (2,0,1))
transforms.set_mean('data', mu)
transforms.set_raw_scale('data', 255)
transforms.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(1,3,227,227)

image = caffe.io.load_image(img_file)
image_pre = transforms.preprocess('data', image)
net.blobs['data'].data[...] = image_pre
t0 = time.clock()
for i in range(0, 10):
    cProfile.run('output = net.forward()')
t = time.clock() - t0
print(t)
output_pro = output['prob'][0]
#print(output_pro)
print(lables[output_pro.argmax()-1])
