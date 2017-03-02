#Slightly modified from Francis Dutil : https://github.com/Dutil/IFT6266

import os
import sys
import numpy as np
import PIL.Image as Image
import glob
import pickle as pkl

class Iterator(object):


    def __init__(self, which_set, root_path="datasets\inpainting",
                 #img_path = 'train2014',
                 caps_path='dict_key_imgID_value_caps_train_and_valid.pkl',
                 batch_size=128, nb_sub=None, extract_center=True, load_caption=True):
        image_path=''
        if which_set=='train':
            img_path = 'train2014'
        elif which_set=='valid' or which_set=='val':
            img_path = 'valid2014'
        else:
            raise ValueError('Set must be train or valid')


        self.root_path = root_path
        self.img_path = os.path.join(sys.path[0],root_path, img_path)
        self.caps_path = os.path.join(sys.path[0],root_path, caps_path)
        self.imgs = glob.glob(self.img_path + "/*.jpg")#Name of all of the imgs
        self.batch_size = batch_size
        self.batch_idx = 0
        self.n_batches = int(len(self.imgs)/self.batch_size)
        self.extract_center = extract_center #Remove center from image
        self.load_caption = load_caption

        if nb_sub is not None:
            self.imgs = self.imgs[:nb_sub]

        if load_caption:
            with open(self.caps_path, "rb") as fd:
                print "Loading the captions..."
                self.caption_dict = pkl.load(fd)
                print "Done"


    def _get_img(self, i):
        #Get ith image form self.imgs
        #If self.extract_center=True then center of img is white
        #Return input (contour), target (center), caption

        img_path = self.imgs[i]
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        #center pixel of the image
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            if self.extract_center:
                input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
        else:
            # Ignore the gray images.
            return None

        cap = None
        if self.load_caption:
            cap= self.caption_dict[cap_id]

        return input.astype('float32')/255., target.astype('float32')/255., cap

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, key):


        if isinstance(key, slice):
            # Get the start, stop, and step from the slice

            res = [self[ii] for ii in xrange(*key.indices(len(self)))]
            xs, ys, caps = zip(*[x for x in res if x is not None])
            return np.array(xs), np.array(ys), caps

        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError, "The index (%d) is out of range." % key
            return self._get_img(key)  # Get the data from elsewhere
        else:
            raise TypeError, "Invalid argument type."


    def __iter__(self):

        for batch_idx in range(int(len(self)/self.batch_size)):

            if (batch_idx+1)*self.batch_size < len(self):
                yield self[batch_idx*self.batch_size: (batch_idx+1)*self.batch_size]
            else:
                yield self[batch_idx * self.batch_size:]



if __name__=='__main__':

    train_iter = Iterator(batch_size = 100)
