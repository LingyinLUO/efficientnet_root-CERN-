import os.path as osp
import torch.utils.data as data
import numpy as np
import uproot
import torchvision.transforms as transforms



ROOT_CLASSES = ('sgn','bkg') #信号，本底

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join('/home/toandm2', "data/VOCdevkit/")

def root_weight(root,image_sets):
    imgpath = osp.join(root, image_sets)
    file = uproot.open(imgpath)
    bkg = np.array(list(file['bkg'].arrays().values())).T.reshape((-1, 2, 32, 32))
    sgn = np.array(list(file['sgn'].arrays().values())).T.reshape((-1, 2, 32, 32))
    mean = [np.row_stack((bkg[:, 0, :, :], sgn[:, 0, :, :])).mean(),
                 np.row_stack((bkg[:, 1, :, :], sgn[:, 1, :, :])).mean()]
    std = [np.row_stack((bkg[:, 0, :, :], sgn[:, 0, :, :])).std(),
                np.row_stack((bkg[:, 1, :, :], sgn[:, 1, :, :])).std()]
    return mean,std

class ROOTDetection(data.Dataset):
    """ROOT Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets='imagesData_1000.root',
                 transform=None):
        self.root = root                                              #文件夹地址
        self.image_set = image_sets                                   #root文件名
        self.transform = transform
        self._imgpath = osp.join(self.root,image_sets)
        self.file=uproot.open(self._imgpath)
        self.bkg = np.array(list(self.file['bkg'].arrays().values())).T.reshape((-1, 2, 32, 32))
        self.sgn = np.array(list(self.file['sgn'].arrays().values())).T.reshape((-1, 2, 32, 32))
        self.num = len(self.sgn)

    def __getitem__(self, index):
        img=[]
        if index < self.num:
            img = self.sgn[index]
            target = 1 
        else:
            img = self.bkg[index-self.num]
            target = 0

        sample = [img.T,target]
        if self.transform is not None:
            sample[0] = self.transform(sample[0])
        return sample


    def __len__(self):
        return 2 * self.num

    def num_classes(self):
        return len(ROOT_CLASSES)

    def label_to_name(self, label):
        return ROOT_CLASSES[label]

    def load_annotations(self, index):
        if index < self.num:
            target = 'sgn'
        else:
            target = 'bkg'
        return target

