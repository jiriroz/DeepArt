# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import random
import time

print "Dependencies imported"

import caffe

print "Caffe imported"

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

class DeepDream:
    
    def __init__(self):
        """Loading DNN model."""
        model_path = '/home/jiri/caffe/models/bvlc_googlenet/'
        net_fn   = model_path + 'deploy.prototxt'
        param_fn = model_path + 'bvlc_googlenet.caffemodel'
        #model_path = '/home/jiri/caffe/models/oxford102/'
        #net_fn   = model_path + 'deploy.prototxt'
        #param_fn = model_path + 'oxford102.caffemodel'

        # Patching model to be able to compute gradients.
        # Note that you can also manually add "force_backward: true" line
        #to "deploy.prototxt".
        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(net_fn).read(), model)
        model.force_backward = True
        open('tmp.prototxt', 'w').write(str(model))

        # ImageNet mean, training set dependent
        mean =  np.float32([104.0, 116.0, 122.0])
        # the reference model has channels in BGR order instead of RGB
        chann_sw = (2,1,0)
        self.net = caffe.Classifier('tmp.prototxt', param_fn, mean=mean, channel_swap=chann_sw)

    def loadimg(self, filename):
        return np.float32(PIL.Image.open(filename))

    def showarray(self, a, name, fmt='jpeg'):
        a = np.uint8(np.clip(a, 0, 255))
        #f = StringIO()
        #PIL.Image.fromarray(a).save(f, fmt)
        #PIL.Image.fromarray(a).save(name + '.' + fmt, fmt)
        #display(Image(data=f.getvalue()))
        if fmt == 'jpeg':
            outputfmt = 'jpg'
        else:
            outputfmt = fmt
        PIL.Image.fromarray(a).save(name + '.' + outputfmt, fmt)

    # a couple of utility functions for converting to and from Caffe's input image layout
    def preprocess(self, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - self.net.transformer.mean['data']
    def deprocess(self, img):
        return np.dstack((img + self.net.transformer.mean['data'])[::-1])

    """Producing dreams."""
    def objective_L2(self, dst, guide_features):
        dst.diff[:] = dst.data 

    '''Basic gradient ascent step.'''
    def make_step(self, step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=None, guide_features=None):
        if objective == None:
            objective = self.objective_L2

        endL, gf = self.choose_end(end, guide_features)
        src = self.net.blobs['data'] # input image is stored in Net's 'data' blob
        dst = self.net.blobs[endL]

        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
        self.net.forward(end=endL)
        # specify the optimization objective
        objective(dst, gf)
        self.net.backward(start=endL)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size/np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
        if clip:
            bias = self.net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255-bias)    

    def dream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
                end='inception_4c/output', clip=True, guide_features=None, name="dream", **step_params):
        # prepare base images for all octaves
        octaves = [self.preprocess(base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
        
        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)
    
            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base+detail
            for i in xrange(iter_n):
                self.make_step(end=end, clip=clip, guide_features=guide_features, **step_params)
            
                # visualization
                vis = self.deprocess(src.data[0])
                # adjust image contrast if clipping is disabled
                if not clip:
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                print octave, i, end, vis.shape
                clear_output(wait=True)
            
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        self.showarray(vis, name)
        # returning the resulting image
        return self.deprocess(src.data[0])

    def guide_dream(self, end, name, **dream_params):
        if type(end) is str:
            gf = self.extract_guide_features(self.guide, end=end)
        else:
            gf = [self.extract_guide_features(self.guide, end=e) for e in end]
        return self.dream(self.img, end=end, guide_features=gf, name=name, objective=self.objective_guide, **dream_params)

    def objective_guide(self, dst, guide_features):
        x = dst.data[0].copy()
        y = guide_features
        ch = x.shape[0]
        x = x.reshape(ch,-1)
        y = y.reshape(ch,-1)
        A = x.T.dot(y) # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    def extract_guide_features(self, guide, end='inception_3b/output'):
        h, w = guide.shape[:2]
        src, dst = self.net.blobs['data'], self.net.blobs[end]
        src.reshape(1,3,h,w)
        src.data[0] = self.preprocess(guide)
        self.net.forward(end=end)
        guide_features = dst.data[0].copy()
        return guide_features

    def choose_end(self, end, gfs):
        if type(end) is str:
            return end, gfs
        i = random.randint(0, len(end)-1)
        return end[i], gfs[i]

def vanGogh(imgname):
    artist = DeepDream()
    artist.img = artist.loadimg(imgname)
    artist.guide = artist.loadimg("vanGogh/goghS.jpg")
    end = "inception_3b/output"
    name = imgname.split('.')[0] + "-gogh"
    t = time.time()
    _ = artist.guide_dream(end, name, iter_n=7, octave_n=4, octave_scale=1.4)
    print str(time.time() - t) + " seconds."

def monet(imgname):
    artist = DeepDream()
    artist.img = artist.loadimg(imgname)
    artist.guide = artist.loadimg("Monet/monet2.jpg")
    end = ['inception_3b/output', 'inception_3b/output', 'inception_3b/output', 'conv2/3x3', 'conv2/3x3']
    name = imgname.split('.')[0] + "-monet"
    t = time.time()
    _ = artist.guide_dream(end, name, iter_n=12, octave_n=3, octave_scale=1.5)
    print str(time.time() - t) + " seconds."

def munch(imgname):
    artist = DeepDream()
    artist.img = artist.loadimg(imgname)
    artist.guide = artist.loadimg("Munch/munchS.jpg")
    end = ['inception_3b/output', 'inception_3b/output', 'inception_3b/output', 'conv2/3x3', 'conv2/3x3']
    name = imgname.split('.')[0] + "-munch"
    t = time.time()
    _ = artist.guide_dream(end, name, iter_n=12, octave_n=3, octave_scale=1.5)
    print str(time.time() - t) + " seconds."

