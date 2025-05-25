import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eval_hash import compress, calculate_top_map, calculate_PR_curve, calculate_P_at_topK_curve
from torch.utils.data import DataLoader, TensorDataset
from math import cos, pi
from models import EmbeddingModel
import time


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_time():
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    formatted_time = time.strftime('%H:%M:%S', local_time)
    print(f"Time: {formatted_time}")
    
def combine_nouns(nouns_embedding, images_embedding, tau):
    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]

    retrieval_embeddings = []
    batch_size = 8192
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
        similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
        similarity = torch.softmax(similarity / tau, dim=1)
        retrieval_embedding = (similarity @ nouns_embedding).cpu()
        retrieval_embeddings.append(retrieval_embedding)
    retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()

    return retrieval_embedding

def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch = 10):
    if current_epoch < warmup_epoch:
        lr = lr_min + (lr_max-lr_min) * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def info_nce_loss(embeddings1, embeddings2, external_labels, temperature, use_sim=False):
    epsilon=1e-12
    batch_size = embeddings1.shape[0]

    embeddings1 = nn.functional.normalize(embeddings1, dim=1)
    embeddings2 = nn.functional.normalize(embeddings2, dim=1)

    similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / temperature

    if use_sim == True:
        labels = external_labels

    else:
        # 对角线是1
        labels = torch.eye(batch_size).to(similarity_matrix.device)

    img_txt_logits = F.softmax(similarity_matrix, dim=1)
    img_txt_logits = torch.clamp(img_txt_logits, epsilon, 1. - epsilon)
    
    img_txt_loss = torch.sum(-torch.log(torch.sum(labels*img_txt_logits, dim=1))) / img_txt_logits.shape[0]
    loss = img_txt_loss

    return loss


def eval(model, backbone, database_loader, val_loader, map, name, device):
    model.eval()
    if backbone != None:
        backbone.eval()
    with torch.no_grad():
        retrievalB, retrievalL, queryB, queryL = compress(database_loader, val_loader, model, backbone, device)
        result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=map)
        # print(f"compute PR curve and P@top{map} curve")
        # calculate_PR_curve(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, dataset_name=name)
        # calculate_P_at_topK_curve(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=map, dataset_name=name)
    return result


def sim_word(dataset_name, use_aug):
    texts_embedding = np.load("./data_extract/selected_nouns_embedding.npy")
    texts_embedding = texts_embedding / np.linalg.norm(
        texts_embedding, axis=1, keepdims=True
    )
    images_embedding = np.load("./data_extract/" + dataset_name + "_new_image_embedding_train.npy")
    images_embedding = images_embedding / np.linalg.norm(
        images_embedding, axis=1, keepdims=True
    )
    nouns_embedding = combine_nouns(texts_embedding, images_embedding, 0.004)
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )

    if use_aug:
        images_embedding_aug1 = np.load("./data_extract/" + dataset_name + "_new_image_embedding_train_aug.npy")
        images_embedding_aug1 = images_embedding_aug1 / np.linalg.norm(
            images_embedding_aug1, axis=1, keepdims=True
        )

        images_embedding_aug2 = np.load("./data_extract/" + dataset_name + "_new_image_embedding_train_aug2.npy")
        images_embedding_aug2 = images_embedding_aug2 / np.linalg.norm(
            images_embedding_aug2, axis=1, keepdims=True
        )
    else:
        images_embedding = np.load("./data_extract/" + dataset_name + "_new_image_embedding_train.npy")
        images_embedding = images_embedding / np.linalg.norm(
            images_embedding, axis=1, keepdims=True
        )

    # Load query and retrieval data
    images_embedding_query = np.load("./data_extract/" + dataset_name + "_image_embedding_query.npy")
    images_embedding_query = images_embedding_query / np.linalg.norm(
        images_embedding_query, axis=1, keepdims=True
    )
    images_embedding_retrieval = np.load("./data_extract/" + dataset_name + "_image_embedding_retrieval.npy")
    images_embedding_retrieval = images_embedding_retrieval / np.linalg.norm(
        images_embedding_retrieval, axis=1, keepdims=True
    )
    
    labels_query = np.loadtxt("./data_extract/" + dataset_name + "_labels_query.txt")
    labels_retrieval = np.loadtxt("./data_extract/" + dataset_name + "_labels_retrieval.txt")

    query_dataset = TensorDataset(torch.from_numpy(images_embedding_query).float(), torch.from_numpy(labels_query).float())
    retrieval_dataset = TensorDataset(torch.from_numpy(images_embedding_retrieval).float(), torch.from_numpy(labels_retrieval).float())
    if use_aug:
        train_dataset = TensorDataset(torch.from_numpy(images_embedding_aug1).float(), torch.from_numpy(images_embedding_aug2).float(), torch.from_numpy(nouns_embedding).float())
    else:
        train_dataset = TensorDataset(torch.from_numpy(images_embedding).float(), torch.from_numpy(nouns_embedding).float())
        
    return train_dataset, query_dataset, retrieval_dataset


if __name__ == "__main__":
    set_seeds(0)
    for dataset_name in ["CIFAR-10", "nus-wide", "flickr25k", "mscoco"]:
        print(dataset_name)
        for bit in [16, 32, 64]:
            get_time()
            print("hashbit:", bit)
            warmup_epoch = 10
            learning_rate = 1e-4
            lr_min = 1e-5
            batch_size = 512
            num_epochs = 60
            input_img_dim = 512
            device = "cuda:0"
            use_sim = True
            use_aug = True

            if dataset_name == "CIFAR-10":
                map = 1000
                temperature = 0.8
                threshold = 0.97
            elif dataset_name == "nus-wide":
                map = 5000
                temperature = 0.8
                threshold = 0.97
            elif dataset_name == "flickr25k":
                map = 5000
                temperature = 0.8
                threshold = 0.97
            elif dataset_name == "mscoco":
                map = 5000
                temperature = 0.2
                threshold = 0.97
            else:
                raise NotImplementedError

            train_dataset, query_dataset, retrieval_dataset = sim_word(dataset_name, use_aug)

            dataloader_train = DataLoader(
                train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True
            )
            query_loader = DataLoader(
                query_dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False
            )
            retrieval_loader = DataLoader(
                retrieval_dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False
            )

            model = EmbeddingModel(input_img_dim=input_img_dim, input_txt_dim=512, hashbit=bit).cuda()

            optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': learning_rate}
            ])

            result = 0
            for epoch in range(num_epochs):
                # 余弦退火
                current_lr = warmup_cosine(optimizer=optimizer, current_epoch=epoch, max_epoch=num_epochs, lr_min=lr_min, lr_max=learning_rate, warmup_epoch=warmup_epoch)
                model.train()
                for iter, data_tuple in enumerate(dataloader_train):
                    if use_aug:
                        image1, image2, text = data_tuple
                        feature1 = image1.to(device)
                        feature2 = image2.to(device)
                        text = text.to(device)
                        embeddings1, embeddings2 = model(feature1, text)
                        embeddings3, _ = model(feature2, text)
                    else:
                        image, text = data_tuple
                        feature = image.to(device)
                        text = text.to(device)
                        embeddings1, embeddings2 = model(feature, text)
                    
                    text = nn.functional.normalize(text, dim=1)
                    sim_txt = torch.mm(text, text.t())
                    external_labels = (sim_txt > threshold).float()

                    loss1 = info_nce_loss(embeddings1, embeddings2, external_labels, temperature, use_sim=use_sim)
                    
                    if use_aug:
                        Aug_loss = info_nce_loss(embeddings3, embeddings2, external_labels, temperature, use_sim=use_sim) + info_nce_loss(embeddings1, embeddings3, external_labels, temperature, use_sim=use_sim)
                        loss = loss1 + Aug_loss
                    else:
                        loss = loss1
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            cal_map = eval(model, None, retrieval_loader, query_loader, map, dataset_name, device=device)
            print(f"MAP: {cal_map:.3f}")
    