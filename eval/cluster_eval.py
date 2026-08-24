import argparse
import contextlib
import io
import logging
import os
import pickle

import pandas as pd

from src.utils import load_env, get_logger
from src.experiment_config import ExperimentConfig
from src.data_loading import DatasetLoader
from src.lr import RegressionModel
from src.clustering.metrics.cluster_metrics import ClusterMetrics
from src.clustering.metrics.purity import Purity
from experiments.cluster_lr import load_filtered_raw_data


def _parse_train_metrics(stdout_str):
    """Extract train_acc and train_f1 from the tab-separated summary line printed by regression()."""
    for line in stdout_str.splitlines():
        parts = line.split('\t')
        if len(parts) == 4:
            try:
                train_acc, _, train_f1, _ = [float(p) for p in parts]
                return train_acc, train_f1
            except ValueError:
                pass
    return None, None


def _eval_config(clustering_data, metaphors, docs, label, feature_key, seed, logger):
    """Compute calinski, purity, and LR metrics for one (method, k, w_cl) configuration."""

    calinski = ClusterMetrics(clustering_data, logger=logger).compute_calinski_harabasz()

    purity = Purity(metaphors, clustering_data, feature_key, logger=logger)
    purity.compute_purity()
    purity_25 = purity.results['25']['exact_match_purity']
    purity_100 = purity.results['100']['exact_match_purity']

    model = RegressionModel(seed)
    data_all = model.format_dataset(clustering_data, metaphors, docs, None, use_centroid_filtering=False)
    data_all[label] = data_all['doc_id'].apply(lambda x: docs[x][label])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, report = model.regression(data_all.copy(), label)

    train_acc, train_f1 = _parse_train_metrics(buf.getvalue())
    test_acc = round(report['accuracy'] * 100, 2)
    test_f1 = round(report['macro avg']['f1-score'] * 100, 2)

    metrics = {
        'Calinski_Harabasz': round(calinski, 2),
        'LR_Train_Acc': train_acc,
        'LR_Test_Acc': test_acc,
        'LR_Train_F1': train_f1,
        'LR_Test_F1': test_f1,
        'Purity@25': round(purity_25 * 100, 2) if purity_25 is not None else None,
        'Purity@100': round(purity_100 * 100, 2) if purity_100 is not None else None,
    }

    if len(feature_key) > 1:
        for k_str in ['25', '100']:
            fp = purity.results[k_str]['feature_purity'] or {}
            for fk, score in fp.get('key_purities', {}).items():
                metrics[f'FeatPurity@{k_str}_{fk}'] = round(score * 100, 2)
            if 'overall_purity' in fp:
                metrics[f'FeatPurity@{k_str}_overall'] = round(fp['overall_purity'] * 100, 2)

    return metrics


def main(args):
    env_vars = load_env()
    logger = get_logger('cluster_eval', level=logging.INFO)
    seed = env_vars['RANDOM_SEED']

    config = ExperimentConfig(
        task_type='cluster_lr',
        task_name='',
        dataset=args.dataset,
        labeled_only=args.labeled_only,
        logger=logger,
        env_vars=env_vars,
    )
    dataset_out_name = config.dataset_out_name
    results_dir = env_vars['RESULTS_DIR']

    features_path = (
        f"{results_dir}/gen_embeddings/{args.embedding_model}"
        f"/{dataset_out_name}_{args.embedding_name}_embeddings_features.pkl"
    )
    logger.info(f"Loading features from {features_path}")
    with open(features_path, 'rb') as f:
        metaphors = pickle.load(f)

    data_loader = DatasetLoader(config)
    docs = load_filtered_raw_data(
        args.dataset, args.labeled_only, [args.label],
        args.predict_over_met_docs, logger, data_loader
    )

    rows = []
    for k in args.k_list:
        logger.info(f"=== k={k} ===")

        kmeans_path = (
            f"{results_dir}/cluster_embeddings/kmeans"
            f"/{dataset_out_name}/{args.embedding_name}_clusters_{k}.pkl"
        )
        logger.info(f"KMeans: {kmeans_path}")
        with open(kmeans_path, 'rb') as f:
            kmeans_data = pickle.load(f)
        metrics = _eval_config(kmeans_data, metaphors, docs, args.label, args.feature_key, seed, logger)
        rows.append({'Model': 'KMeans', 'N_CLUSTERS': k, 'W_CL': 0, **metrics})

        for w_cl in args.w_cl_list:
            pckmeans_path = (
                f"{results_dir}/cluster_embeddings/pckmeans"
                f"/{dataset_out_name}/{args.embedding_name}_clusters_{k}_w{w_cl}.pkl"
            )
            logger.info(f"PCKMeans w_cl={w_cl}: {pckmeans_path}")
            with open(pckmeans_path, 'rb') as f:
                pckmeans_data = pickle.load(f)
            metrics = _eval_config(pckmeans_data, metaphors, docs, args.label, args.feature_key, seed, logger)
            rows.append({'Model': 'Constrained KMeans', 'N_CLUSTERS': k, 'W_CL': w_cl, **metrics})

        rows.append({})
        rows.append({})

    df = pd.DataFrame(rows)

    out_dir = f"{results_dir}/eval"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{dataset_out_name}_{args.embedding_name}_cluster_eval.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")
    print(df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate clustering configs: calinski, purity, and LR metrics')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--labeled_only', action='store_true')
    parser.add_argument('--embedding_model', default='sbert')
    parser.add_argument('--embedding_name', required=True)
    parser.add_argument('--feature_key', nargs='+', default=['target_group'])
    parser.add_argument('--label', required=True, help='Doc-level label to predict with LR')
    parser.add_argument('--predict_over_met_docs', action='store_true')
    parser.add_argument('--k_list', nargs='+', type=int,
                        default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300])
    parser.add_argument('--w_cl_list', nargs='+', type=float, default=[0.01, 0.05, 0.1])
    args = parser.parse_args()
    main(args)
