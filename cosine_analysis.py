import os
import numpy as np
from matplotlib.pyplot import legend, figure, subplot, hist, title, xlabel, ylabel, tight_layout, show, xlim
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.integrate import simpson

EMB_PATH = r'C:\Users\anshk\Unsupervised_clustering\maxtime_embeddings'

def compute_kde_overlap(data1, data2, bandwidth=0.05):
    kde1 = gaussian_kde(data1, bw_method=bandwidth)
    kde2 = gaussian_kde(data2, bw_method=bandwidth)

    x = np.linspace(0, 1, 1000)
    y1 = kde1(x)
    y2 = kde2(x)

    overlap = np.minimum(y1, y2)
    area = simpson(overlap, x)
    return round(area * 100, 2)
    
categories = {
    'Same_Speaker_1': [],
    'Same_Speaker_2': [],
    'Same_Speaker_mixed_language': [],
    'Different_Speaker_1': [],
    'Different_Speaker_2': [],
    'Different_Speaker_mixed_language': []
}
distances = {
    'Same_Speaker_1': [],
    'Same_Speaker_2': [],
    'Same_Speaker_mixed_language': [],
    'Different_Speaker_1': [],
    'Different_Speaker_2': [],
    'Different_Speaker_mixed_language': []
}
all_files = os.listdir(EMB_PATH)
all_embs = dict()
for i in all_files:
    all_embs[i] = np.load(EMB_PATH + '\\' + i).reshape(1,-1)
    for j in all_files:
        emb1 = i.split('_')
        emb2 = j.split('_')
        if i==j:
            continue
        elif emb1[1] == emb2[1]:
            if emb1[0] == emb2[0]:
                categories[f'Same_Speaker_{emb1[1][-1]}'].append((i,j))
            else:
                categories[f'Different_Speaker_{emb1[1][-1]}'].append((i,j))
        else:
            if emb1[0] == emb2[0]:
                categories[f'Same_Speaker_mixed_language'].append((i,j))
            else:
                categories[f'Different_Speaker_mixed_language'].append((i,j))
for i in categories.keys():
    categories[i] = list({tuple(sorted(pair)) for pair in categories[i]})
#    print(len(categories[i]))
    for e1, e2 in categories[i]:
        distances[i].append(list(cosine_distances(all_embs[e1],all_embs[e2]))[0][0])
    #print(distances)


overlay_pairs = [
    ('Same_Speaker_1', 'Different_Speaker_1'),
    ('Same_Speaker_2', 'Different_Speaker_2'),
    ('Same_Speaker_mixed_language', 'Different_Speaker_mixed_language')
]

figure(figsize=(15, 5))
titles = ['Same vs different speaker - English', 'Same vs different speaker - Hindi', 'Same vs different speaker - Mixed']
for i, (s_key,d_key) in enumerate(overlay_pairs, 1):
    overlap = compute_kde_overlap(distances[s_key], distances[d_key])
    print(overlap)
    subplot(1, 3, i)
#    hist(distances[s_key], bins=30, alpha=0.5, color='red', label = 'Same Speaker')
#    hist(distances[d_key], bins=30, alpha=0.5, color='blue', label = 'Same Speaker')
    sns.kdeplot(distances[s_key], label=f'Same Speaker (Overlap area = {overlap}%)', fill=True)
    sns.kdeplot(distances[d_key], label=f'Different Speaker (Overlap area = {overlap}%)', fill=True)
    
#    sns.kdeplot([], label=f'Overlap = {overlap}%', fill=True)
    xlim(0, 1)
    title(titles[i-1])
    xlabel('Cosine Distance')
    ylabel('Count')
    legend()

tight_layout()
show()
