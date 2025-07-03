from statistics import mode
from numpy import mean,std,max,min
from eval import acc
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

def cluster_embeddings(embeddings, n_clusters):
    affinity = cosine_similarity(embeddings)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    return clustering.fit_predict(affinity)


def diarization_pipeline(all_embeddings, filenames, n_speakers):
    accuracies = []
    epochs=1000
    for i in tqdm(range(epochs),desc='Performing Spectral Clustering'):
        labels = cluster_embeddings(all_embeddings, n_clusters=n_speakers)
        accuracies.append(acc(filenames, labels))
    accuracies

    print("Mean:", f"{round(mean(accuracies), 2)}%")
    print("Standard Deviation:", f"{round(std(accuracies), 2)}%")
    print("Max:", f"{round(max(accuracies), 2)}%")
    print("Min:", f"{round(min(accuracies), 2)}%")
    print("Mode:", f"{round(mode(accuracies), 2)}%")
    plt.hist(accuracies, bins=25, edgecolor='black', range=(min(accuracies), max(accuracies)))
    plt.axvline(85, color='red', linestyle='--', label='85%')
    plt.axvline(90, color='green', linestyle='--', label='90%')
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Spectral Clustering Accuracy ({epochs} Runs)")
    plt.legend()
    plt.grid(True)
    plt.show()

