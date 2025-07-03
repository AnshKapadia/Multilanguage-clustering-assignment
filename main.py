from extract_embeddings import get_embeddings
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_dir", default=r'C:\Users\anshk\Unsupervised_clustering', help="Path to directory")
parser.add_argument("--n_speakers", type=int, default=36, help="Number of speakers")
parser.add_argument("--force_emb", type=bool, default=False, help="Whether to force embedding creation")
parser.add_argument("--clust_type", type=str, default='sc', help="Agglomerative or Spectral")
parser.add_argument("--language", type=str, default='both', help="Which languages to consider: hindi/english/both")

args = parser.parse_args()
if str(args.clust_type).startswith(('a', 'A')): 
    from model_ahc import *
else: 
    from model_sc import *
all_embeddings, filenames = get_embeddings(args.project_dir, args.force_emb, args.language)
diarization_pipeline(all_embeddings, filenames, args.n_speakers)

    