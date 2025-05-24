import numpy as np
import torch
import os
from dataset import CIFAR10, Flickr25k, NusWideDatasetTC21, MScoco, train_transform2, query_transform2, train_transform, query_transform
from torch.utils.data import DataLoader
from models import CLIPModel

def data_prepare(dataset_name):
    batch_size = 512

    if dataset_name == "CIFAR-10":
        train_index = np.loadtxt('/media/hdd4/sqh/Hash_Part/Hash_TAC/data/cifar-10-batches-py/train_index.txt', dtype=int)
        train_dataset = CIFAR10(root="./data", model='train', download=True, transform=train_transform2(), train_index=train_index)
        query_dataset = CIFAR10(root="./data", model='query', download=True, transform=query_transform2())
        retrieval_dataset = CIFAR10(root="./data", model='retrieval', download=True, transform=query_transform2())

    elif dataset_name == "flickr25k":
        data_path = '/media/hdd4/sqh/MIRFLICKR25K/mirflickr25k-iall.mat'
        label_path = '/media/hdd4/sqh/MIRFLICKR25K/mirflickr25k-lall.mat'
        train_index = np.loadtxt('/media/hdd4/sqh/Hash_Part/Hash_TAC/data/flickr25k/train_index.txt', dtype=int)
        retrieval_index = np.loadtxt('/media/hdd4/sqh/Hash_Part/Hash_TAC/data/flickr25k/retrieval_index.txt', dtype=int)
        query_index = np.loadtxt('/media/hdd4/sqh/Hash_Part/Hash_TAC/data/flickr25k/query_index.txt', dtype=int)

        train_dataset = Flickr25k(data_path, label_path, transform=train_transform(), mode='train', index=train_index)
        query_dataset = Flickr25k(data_path, label_path, transform=query_transform(), mode='query', index=query_index)
        retrieval_dataset = Flickr25k(data_path, label_path, transform=query_transform(), mode='retrieval', index=retrieval_index)

    elif dataset_name == "nus-wide":
        root = '/media/hdd4/sqh/Hash_Part/Hash_TAC/data/nuswide21/NUSWIDE'
        train_dataset = NusWideDatasetTC21(root, img_txt='train_img.txt', label_txt='train_label_onehot.txt', transform=train_transform())
        query_dataset = NusWideDatasetTC21(root, img_txt='test_img.txt', label_txt='test_label_onehot.txt', transform=query_transform())
        retrieval_dataset = NusWideDatasetTC21(root, img_txt='database_img.txt', label_txt='database_label_onehot.txt', transform=query_transform())

    elif dataset_name == "mscoco":
        root = '/media/hdd4/sqh/Hash_Part/Hash_TAC/data/mscoco'
        train_dataset = MScoco(root, img_txt='train.txt', transform=train_transform())
        query_dataset = MScoco(root, img_txt='test.txt', transform=query_transform())
        retrieval_dataset = MScoco(root, img_txt='database.txt', transform=query_transform())
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported. Please choose from: 'CIFAR-10', 'flickr25k', 'nus-wide', 'mscoco'.")

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    dataloader_query = DataLoader(query_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    dataloader_retrieval = DataLoader(retrieval_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
        
    return dataloader_train, dataloader_query, dataloader_retrieval

if __name__ == "__main__":
    for dataset_name in ["CIFAR-10", "nus-wide", "flickr25k", "mscoco"]:
        print(f'------   loading the dataset: {dataset_name}   ------')
        dataloader_train, dataloader_query, dataloader_retrieval = data_prepare(dataset_name)

        # CLIP
        model = CLIPModel(model_name="ViT-B/16").cuda()
        model.eval()

        features = []
        features_aug1 = []
        features_aug2 = []
        print("Inferring train image features...")
        for iteration, (x, x_aug1, x_aug2) in enumerate(dataloader_train):
            x = x.cuda()
            x_aug1 = x_aug1.cuda()
            x_aug2 = x_aug2.cuda()
            with torch.no_grad():
                feature = model.encode_image(x)
                feature_aug1 = model.encode_image(x_aug1)
                feature_aug2 = model.encode_image(x_aug2)
            features.append(feature.cpu().numpy())
            features_aug1.append(feature_aug1.cpu().numpy())
            features_aug2.append(feature_aug2.cpu().numpy())
            if iteration % 10 == 0:
                print(f"[Iter {iteration}/{len(dataloader_train)}]")
        features = np.concatenate(features, axis=0)
        features_aug1 = np.concatenate(features_aug1, axis=0)
        features_aug2 = np.concatenate(features_aug2, axis=0)
        print("Feature shape:", features.shape, "Aug_Feature shape:", features_aug1.shape)

        query_features = []
        query_labels = []
        print("Inferring query image features and labels...")
        for iteration, (x, y) in enumerate(dataloader_query):
            x = x.cuda()
            with torch.no_grad():
                query_feature = model.encode_image(x)
            query_features.append(query_feature.cpu().numpy())
            query_labels.append(y.numpy())
            if iteration % 10 == 0:
                print(f"[Iter {iteration}/{len(dataloader_query)}]")
        query_features = np.concatenate(query_features, axis=0)
        query_labels = np.concatenate(query_labels, axis=0)
        print("Feature shape:", query_features.shape, "Label shape:", query_labels.shape)

        retrieval_features = []
        retrieval_labels = []
        print("Inferring retrieval image features and labels...")
        for iteration, (x, y) in enumerate(dataloader_retrieval):
            x = x.cuda()
            with torch.no_grad():
                retrieval_feature = model.encode_image(x)
            retrieval_features.append(retrieval_feature.cpu().numpy())
            retrieval_labels.append(y.numpy())
            if iteration % 10 == 0:
                print(f"[Iter {iteration}/{len(dataloader_retrieval)}]")
        retrieval_features = np.concatenate(retrieval_features, axis=0)
        retrieval_labels = np.concatenate(retrieval_labels, axis=0)
        print("Feature shape:", retrieval_features.shape, "Label shape:", retrieval_labels.shape)
    
        os.makedirs("data_extract", exist_ok=True)

        np.save("./data_extract/" + dataset_name + "_new_image_embedding_train.npy", features)
        np.save("./data_extract/" + dataset_name + "_new_image_embedding_train_aug.npy", features_aug1)
        np.save("./data_extract/" + dataset_name + "_new_image_embedding_train_aug2.npy", features_aug2)
        np.save("./data_extract/" + dataset_name + "_image_embedding_query.npy", query_features)
        np.save("./data_extract/" + dataset_name + "_image_embedding_retrieval.npy", retrieval_features)

        np.savetxt("./data_extract/" + dataset_name + "_labels_query.txt", query_labels)
        np.savetxt("./data_extract/" + dataset_name + "_labels_retrieval.txt", retrieval_labels)
