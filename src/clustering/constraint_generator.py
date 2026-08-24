import argparse
import itertools
import os
import pickle
from typing import Tuple

from tqdm import tqdm

from src.clustering.utils.constraint_flat_db import ConstraintFlatDB
from src.clustering.utils.constraints_graph_db import ConstraintGraphDB
from src.utils import load_env, get_logger


def compute_constraints(
    data: dict,
    feature_key,
    constraints_flat_path: str,
    constraints_graph_path: str,
    batch_size: int = 100000,
    logger=None
) -> Tuple[str, str]:
    """
    Compute cannot-link constraints between items that differ on any of the feature keys
    and write them incrementally to LMDB databases.

    Args:
        data: Embeddings dict keyed by string index; each entry must have a 'features' dict
        feature_key: Feature key string or list of strings; items differing on any key get a cannot-link constraint
        constraints_flat_path: Path for the ConstraintFlatDB (sorted constraint pairs)
        constraints_graph_path: Path for the ConstraintGraphDB (adjacency sets)
        batch_size: Number of constraints to accumulate before flushing to disk

    Returns:
        Paths to the two LMDB databases (flat, graph)
    """
    feature_keys = [feature_key] if isinstance(feature_key, str) else list(feature_key)

    constraints_flat_db = ConstraintFlatDB(db_path=constraints_flat_path)
    constraints_graph_db = ConstraintGraphDB(db_path=constraints_graph_path)

    log = logger.info if logger else print

    n = len(data)
    total_combinations = n * (n - 1) // 2
    log(f"Computing constraints over {total_combinations} pairs (feature keys: {feature_keys})...")

    batch_flat = []
    batch_graph = {}
    total_written = 0

    for k1, k2 in tqdm(itertools.combinations(data.keys(), 2), total=total_combinations):
        feats1 = data[k1].get("features")
        feats2 = data[k2].get("features")

        if feats1 is None or feats2 is None:
            continue

        if any(feats1.get(fk) != feats2.get(fk) for fk in feature_keys):
            i, j = (int(k1), int(k2)) if int(k1) < int(k2) else (int(k2), int(k1))
            batch_flat.append((i, j))
            batch_graph.setdefault(i, set()).add(j)
            batch_graph.setdefault(j, set()).add(i)

            if len(batch_flat) >= batch_size:
                constraints_flat_db.add_tuples(batch_flat)
                constraints_graph_db.update_sets(batch_graph)
                total_written += len(batch_flat)
                log(f"Flushed batch — total written: {total_written}")
                batch_flat = []
                batch_graph = {}

    # Flush remaining
    if batch_flat:
        constraints_flat_db.add_tuples(batch_flat)
        constraints_graph_db.update_sets(batch_graph)
        total_written += len(batch_flat)

    constraints_flat_db.close()
    constraints_graph_db.close()

    log(f"Done. Total constraints written: {total_written}")
    return constraints_flat_path, constraints_graph_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cannot-link constraints from embedding features")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--labeled_only", action="store_true")
    parser.add_argument("--embedding_model", default="sbert", help="Model used to generate embeddings")
    parser.add_argument("--embedding_name", required=True, help="Embedding name tag used in the features filename")
    parser.add_argument("--feature_key", nargs="+", default=["target_group"], help="Feature key(s) to use for constraint generation; cannot-link issued when items differ on any key")
    parser.add_argument("--batch_size", type=int, default=100000)
    args = parser.parse_args()

    logger = get_logger("constraint_generator")
    env_vars = load_env(logger)

    dataset_out_name = args.dataset
    if args.labeled_only:
        dataset_out_name += "_labeled"

    features_path = (
        f"{env_vars['RESULTS_DIR']}/gen_embeddings/{args.embedding_model}"
        f"/{dataset_out_name}_{args.embedding_name}_embeddings_features.pkl"
    )
    logger.info(f"Loading features from {features_path}")
    with open(features_path, "rb") as f:
        data = pickle.load(f)

    out_dir = f"{env_vars['RESULTS_DIR']}/constraints/{args.embedding_model}"
    os.makedirs(out_dir, exist_ok=True)
    feature_key_tag = "_".join(args.feature_key)
    base_name = f"{dataset_out_name}_{args.embedding_name}_{feature_key_tag}"
    constraints_flat_path = f"{out_dir}/{base_name}_flat.db"
    constraints_graph_path = f"{out_dir}/{base_name}_graph.db"

    logger.info(f"Writing constraints to {out_dir}/{base_name}_{{flat,graph}}")
    compute_constraints(
        data=data,
        feature_key=args.feature_key,
        constraints_flat_path=constraints_flat_path,
        constraints_graph_path=constraints_graph_path,
        batch_size=args.batch_size,
        logger=logger
    )