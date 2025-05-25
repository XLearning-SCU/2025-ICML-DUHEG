import torch
import faiss
import clip
import os
from models import CLIPModel
import numpy as np
import pandas as pd
import torch.nn.functional as F


def get_prompt(words, index, SIMPLE_IMAGENET_TEMPLATES, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](word) for word in words]
    text = clip.tokenize(prompt, truncate=True).to(device)
    return text

def nouns_prepare():
    SIMPLE_IMAGENET_TEMPLATES = (
        lambda c: f"itap of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"art of the {c}.",
        lambda c: f"a photo of the small {c}.",
    )# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

    nouns = pd.read_csv("./data/WordNet_Nouns.csv").values
    nouns_num = nouns.shape[0]
    batch_size = 2048
    model = CLIPModel(model_name="ViT-B/16").cuda()
    model.eval()

    for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
        features = []
        print("Inferring text features for index", index)
        for i in range(nouns_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > nouns_num:
                end = nouns_num
            nouns_batch = nouns[start:end]
            with torch.no_grad():
                prompt = get_prompt(nouns_batch[:, 0], index, SIMPLE_IMAGENET_TEMPLATES)
                feature = model.encode_text(prompt)
                features.append(feature.cpu().numpy())
            if i % 50 == 0:
                print(f"[Completed {i * batch_size}/{nouns_num}]")
        features = np.concatenate(features, axis=0)
        print("Feature shape:", features.shape)
        os.makedirs("data_extract", exist_ok=True)
        np.save("./data_extract/nouns_embedding_prompt_" + str(index) + ".npy", features)

    # Multi Prompts
    embeddings = np.zeros((nouns_num, 512))
    for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
        embedding = np.load("./data_extract/nouns_embedding_prompt_" + str(index) + ".npy")
        embeddings += embedding
    embeddings = embeddings / len(SIMPLE_IMAGENET_TEMPLATES)
    np.save("./data_extract/nouns_embedding_ensemble.npy", embeddings)

def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=300, nredo=10)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    print("K-means clustering done.")
    return D.reshape(-1), I.reshape(-1)

def TopK_nouns(preds, nouns_embedding, cluster_num, topK):
    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_centers = torch.zeros((cluster_num, 512), dtype=torch.float16).cuda()
    for k in range(cluster_num):
        nouns_centers[k] = nouns_embedding[preds == k].mean(dim=0)
    nouns_centers = F.normalize(nouns_centers, dim=1)

    similarity = torch.matmul(nouns_centers, nouns_embedding.T)
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    class_pred = torch.argmax(softmax_nouns, dim=0).long()

    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
    for k in range(cluster_num):
        if (class_pred == k).sum() == 0:
            continue
        class_index = torch.where(class_pred == k)[0]
        confidence = softmax_nouns[k, class_index]
        rank = torch.argsort(confidence, descending=True)
        selected_idx[class_index[rank[:topK]]] = True
    selected_idx = selected_idx.cpu().numpy()

    nouns_embedding_selected = nouns_embedding[selected_idx]
    return nouns_embedding_selected


if __name__ == "__main__":
    try:
        nouns_embedding = np.load("./data_extract/nouns_embedding_ensemble.npy")
    except:
        nouns_prepare()
        print("Please rerun the script.")
        exit()
    cluster_num = 800
    topK = 1
    threshold = 0.9
    try:
        nouns_embedding = np.load("./data_extract/filtered_nouns_embedding_ensemble.npy")
        nouns_embedding = (nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)).astype('float32')
    except:
        nouns_embedding = (nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)).astype('float32')

        res = faiss.StandardGpuResources()

        print(f'initial word: {len(nouns_embedding)}')
        all_self = False
        
        while not all_self:
            index = faiss.IndexFlatL2(512)  # 使用L2距离
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(nouns_embedding)
            D, I = gpu_index.search(nouns_embedding, k=2)

            # 删除第一个不是自己的索引
            indices_to_delete = []
            all_self = True
            for i in range(len(I)):
                if I[i, 0] != i:
                    indices_to_delete.append(I[i, 0])
                    all_self = False
            mask = np.ones(nouns_embedding.shape[0], dtype=bool)
            mask[indices_to_delete] = False
            nouns_embedding = nouns_embedding[mask]
            print(f'remaining word: {len(nouns_embedding)}')
        
        filtered_nouns_embedding = nouns_embedding
        np.save("./data_extract/filtered_nouns_embedding_ensemble.npy", filtered_nouns_embedding)
        print("Please rerun the script.")
        exit()

    dis, preds = kmeans(nouns_embedding, cluster_num)
    repst_nouns_embedding = nouns_embedding[dis>threshold]

    clus_nouns_embedding = TopK_nouns(preds, nouns_embedding, cluster_num, topK)
    
    selected_nouns_embedding = np.vstack((repst_nouns_embedding, clus_nouns_embedding.cpu().numpy()))
    np.save(
        "./data_extract/selected_nouns_embedding.npy",
        repst_nouns_embedding
    )
    