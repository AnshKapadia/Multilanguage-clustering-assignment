
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_dir", default=r'C:\Users\anshk\Unsupervised_clustering', help="Path to directory")
parser.add_argument("--n_speakers", type=int, default=36, help="Number of speakers")
parser.add_argument("--force_emb", type=bool, default=False, help="Whether to force embedding creation")
parser.add_argument("--clust_type", type=str, default='sc', help="Agglomerative or Spectral")

args = parser.parse_args()
if str(args.clust_type).startswith(('a', 'A')): 
    from model_ahc import *
else: 
    from model_sc import *

diarization_pipeline(args.project_dir, args.n_speakers, args.force_emb)

    