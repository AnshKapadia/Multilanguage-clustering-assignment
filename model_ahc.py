from eval import acc
from sklearn.cluster import AgglomerativeClustering

def cluster_embeddings(embeddings, n_clusters, clust_metric = 'cosine'):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=clust_metric, linkage='average')
    return clustering.fit_predict(embeddings)

def diarization_pipeline(all_embeddings, filenames, n_speakers):
    labels = cluster_embeddings(all_embeddings, n_clusters=n_speakers)
    #labels = [floor(i/4) for i in range(len(labels))]
    print(f'Agglomerative Heirarchichal Clustering accuracy: {acc(filenames, labels)}')
#    x = sorted(zip(filenames, labels), key=lambda pair: pair[1])
#    for filename, label in x:  print(f"{filename} | Speaker {label}")
