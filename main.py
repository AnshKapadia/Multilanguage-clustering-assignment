from model_sc import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_dir", default=r'C:\Users\anshk\Unsupervised_clustering', help="Path to directory")
parser.add_argument("--n_speakers", type=int, default=36, help="Number of speakers")
parser.add_argument("--force_emb", type=bool, default=False, help="Whether to force embedding creation")

args = parser.parse_args()
diarization_pipeline(args.project_dir, args.n_speakers, args.force_emb)

    