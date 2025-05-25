import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import scipy.io as scio
import h5py
import glob
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import cv2
from torchvision.transforms import InterpolationMode


BICUBIC = InterpolationMode.BICUBIC

class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
    
def train_transform2():
    color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p = 0.7),
        transforms.RandomGrayscale(p = 0.2),
        GaussianBlur(3),
        transforms.ToTensor(),
        normalize,
    ])

def train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),                         
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def query_transform2():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

def query_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

class CIFAR10(VisionDataset):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    database_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        model: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        nouns_emb = None,
        train_index = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.model = model
        self.nouns_emb = nouns_emb
        self.train_index = train_index

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.model == 'train':
            downloaded_list = self.train_list
        elif self.model == 'query':
            downloaded_list = self.test_list
        elif self.model == 'retrieval':
            downloaded_list = self.train_list

        self.data: Any = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        if self.model == 'train':
            features_train = []
            features_train.append(self.data[self.train_index])
            self.data = np.vstack(features_train)

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            if self.model == 'train':
                trans = query_transform2()
                img_ori = trans(img)
                img_aug1 = self.transform(img)
                img_aug2 = self.transform(img)
                return img_ori, img_aug1, img_aug2
            else:
                img = self.transform(img)
                return img, target
        else:
            text = self.nouns_emb[index]
            trans = train_transform2()
            img_aug1 = trans(img)
            img_aug2 = trans(img)
            return img_aug1, img_aug2, torch.from_numpy(text).float()


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

class Flickr25k(Dataset):
    def __init__(self, img_path, label_path, transform=None, nouns_emb=None, mode=None, index=None):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.mode = mode
        self.nouns_emb = nouns_emb
        self.index = index
        self.labels = scio.loadmat(self.label_path)['LAll'][:]

    def __getitem__(self, index):
        idx = self.index[index]
        with h5py.File(self.img_path, 'r') as file:
            img_data = file['IAll'][idx]
        img = Image.fromarray(np.transpose(img_data, (2, 1, 0))).convert('RGB')
        if self.transform is not None:
            if self.mode == 'train':
                trans = query_transform()
                img_ori = trans(img)
                img_aug1 = self.transform(img)
                img_aug2 = self.transform(img)
                return img_ori, img_aug1, img_aug2
            else:
                img = self.transform(img)
                return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            text = self.nouns_emb[index]
            trans = train_transform()
            img_aug1 = trans(img)
            img_aug2 = trans(img)
            return img_aug1, img_aug2, torch.from_numpy(text).float()

    def __len__(self):
        return self.index.shape[0]
    
class NusWideDatasetTC21(Dataset):
    def __init__(self, root, img_txt, label_txt, transform=None, nouns_emb=None):
        self.root = root
        self.img_txt = img_txt
        self.transform = transform
        self.nouns_emb = nouns_emb

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)

        # Read files
        with open(img_txt_path, 'r') as f:
            self.data = np.array(['Flickr/*/'+i.strip().split('/')[-1] for i in f])
        self.targets = np.loadtxt(label_txt_path, dtype=np.float32)

    def __getitem__(self, index):
        img = Image.open(glob.glob(os.path.join(self.root, self.data[index]))[0]).convert('RGB')
        if self.transform is not None:
            if self.img_txt=='train_img.txt':
                trans = query_transform()
                img_ori = trans(img)
                img_aug1 = self.transform(img)
                img_aug2 = self.transform(img)
                return img_ori, img_aug1, img_aug2
            else:
                img = self.transform(img)
                return img, torch.tensor(self.targets[index], dtype=torch.float32)
        else:
            text = self.nouns_emb[index]
            trans = train_transform()
            img_aug1 = trans(img)
            img_aug2 = trans(img)
            return img_aug1, img_aug2, torch.from_numpy(text).float()

    def __len__(self):
        return len(self.data)

class MScoco(Dataset):
    def __init__(self, root, img_txt, transform=None, nouns_emb=None):
        self.root = root
        self.img_txt = img_txt
        self.transform = transform
        self.nouns_emb = nouns_emb

        img_txt_path = os.path.join(root, self.img_txt)

        # Read files
        with open(img_txt_path, 'r') as f:
            self.data = np.array([('MSCOCO/'+i).split() for i in f])
        self.img = self.data[:, 0]
        self.targets = self.data[:, 1:].astype(float)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img[index])).convert('RGB')
        if self.transform is not None:
            if self.img_txt == 'train.txt':
                trans = query_transform()
                img_ori = trans(img)
                img_aug1 = self.transform(img)
                img_aug2 = self.transform(img)
                return img_ori, img_aug1, img_aug2
            else:
                img = self.transform(img)
                return img, torch.tensor(self.targets[index], dtype=torch.float32)
        else:
            text = self.nouns_emb[index]
            trans = train_transform()
            img_aug1 = trans(img)
            img_aug2 = trans(img)
            return img_aug1, img_aug2, torch.from_numpy(text).float()

    def __len__(self):
        return len(self.data)
