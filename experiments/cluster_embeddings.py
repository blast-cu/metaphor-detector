
import argparse
import pickle

import numpy as np
from scipy.sparse import csr_matrix

from src.utils import load_env, set_seed, get_logger
from src.experiment_config import ExperimentConfig
from src.clustering.kmeans import KMeansClustering
from src.clustering.weighted_pckmeans import ConstrainedKMeans
from src.clustering.utils.constraint_flat_db import ConstraintFlatDB

def main(args):

    env_vars = load_env()
    seed = env_vars["RANDOM_SEED"]
    set_seed(seed)

    LOGGER = get_logger("cluster_embs")

    task_name = "pckmeans" if args.use_pckmeans else "kmeans"

    config = ExperimentConfig(
        task_type="cluster_embeddings",
        task_name=task_name,
        dataset=args.dataset,
        logger=LOGGER,
        env_vars=env_vars,
        labeled_only=args.labeled_only,
        device=env_vars["DEVICE"],
        random_seed=seed,
        skip_load=True
    )

    LOGGER.info(f"Running {task_name.upper()} clustering with the following parameters:")
    LOGGER.info(f"N_CLUSTERS: {args.k}")
    LOGGER.info(f"EMBEDDING TYPE: {args.embedding_name}")

    input_path = (
        f"{env_vars['RESULTS_DIR']}/gen_embeddings/{args.embedding_model}"
        # f"/{config.dataset_out_name}_{args.embedding_name}_embeddings_features.pkl"
        f"/{config.dataset_out_name}_{args.embedding_name}_embeddings.pkl"

    )
    LOGGER.info(f"Loading embeddings from {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    embeddings = np.array([data[str(i)]["emb"] for i in range(len(data))])

    if args.use_pckmeans:
        LOGGER.info(f"W_CL: {args.w_cl}")
        if args.centroid_percentile is not None:
            LOGGER.info(f"CENTROID_PERCENTILE: {args.centroid_percentile}")
        if args.pairwise_percentile is not None:
            LOGGER.info(f"PAIRWISE_PERCENTILE: {args.pairwise_percentile}")

        feature_key_tag = "_".join(args.feature_key)
        constraints_flat_path = (
            f"{env_vars['RESULTS_DIR']}/constraints/{args.embedding_model}"
            f"/{config.dataset_out_name}_{args.embedding_name}_{feature_key_tag}_flat.db"
        )

        LOGGER.info(f"Loading constraints from {constraints_flat_path}")
        pairs = ConstraintFlatDB(constraints_flat_path).get_all_tuples_as_numpy()
        LOGGER.info(f"Loaded {len(pairs)} cannot-link constraints — building CSR matrix")
        n_samples = len(embeddings)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        constraint_matrix = csr_matrix(
            (np.ones(len(rows), dtype=np.bool_), (rows, cols)),
            shape=(n_samples, n_samples)
        )
        del pairs, rows, cols

        model = ConstrainedKMeans(
            n_clusters=args.k,
            w_cl=args.w_cl,
            max_iter=args.max_iter,
            tol=1e-4,
            early_stopping_tol=5,
            random_state=seed,
            centroid_percentile=args.centroid_percentile,
            pairwise_percentile=args.pairwise_percentile
        )
        model.pckmeans(
            X=embeddings,
            constraint_matrix=constraint_matrix,
            config=config,
            cluster_name=args.embedding_name,
            skip_init=args.skip_init,
            feature_data=data,
            feature_key=args.feature_key
        )
    else:
        model = KMeansClustering(args.k, config, args.embedding_name,
                                 feature_data=data, feature_key=args.feature_key)
        model.kmeans(embeddings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cluster Embeddings')
    parser.add_argument("--dataset", required=True, help="dataset to cluster embeddings for",
                        choices=['tweets_immigration', 'subframes_immigration', 'subframes_guns',
                                 'subframes_abortion', 'lcc-large', 'podcasts'])
    parser.add_argument("--labeled_only", action="store_true")
    parser.add_argument("--embedding_model", default="sbert", help="model used to generate embeddings")
    parser.add_argument("--embedding_name", required=True,
                        help="name tag used in the embeddings filename")
    parser.add_argument("--k", metavar="N_CLUSTERS", default=5, type=int,
                        help="number of clusters")

    # PCKMeans
    parser.add_argument("--use_pckmeans", action="store_true",
                        help="use constrained PCKMeans instead of standard KMeans")
    
    parser.add_argument("--w_cl", default=1.0, type=float,
                        help="weight for cannot-link constraints (pckmeans only)")
    
    parser.add_argument("--feature_key", nargs="+", default=["target_group"],
                        help="feature key(s) used for purity evaluation and cannot-link constraints (pckmeans); cannot-link issued when items differ on any key")
    
    parser.add_argument("--centroid_percentile", default=None, type=float,
                        help="percentile threshold for distance to cluster centers (pckmeans only)")
    
    parser.add_argument("--pairwise_percentile", default=None, type=float,
                        help="percentile threshold for pairwise distances (pckmeans only)")
    
    parser.add_argument("--skip_init", action="store_true",
                        help="initialise centroids with sklearn KMeans instead of constraint-aware KMeans++ (pckmeans only)")
    
    parser.add_argument("--max_iter", default=100, type=int,
                        help="maximum iterations (pckmeans only)")

    args = parser.parse_args()
    main(args)