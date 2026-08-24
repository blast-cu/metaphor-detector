import argparse
import json
import pickle
import random
import csv
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from src.utils import load_env, get_logger, load_json
from src.experiment_config import ExperimentConfig


class IntrusionDataGenerator:
    def __init__(self, config: ExperimentConfig, logger, jaccard_threshold: float = 0.8):
        """
        Initialize the intrusion data generator.
        
        Args:

            random_seed: Random seed for reproducibility
            jaccard_threshold: Threshold for rejecting similar positive pairs (default: 0.8)
        """
        self.config = config
        self.random_seed = config.random_seed
        self.jaccard_threshold = jaccard_threshold
        self.logger = logger
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # # Load processed chains
        # with open(processed_chains_path, 'rb') as f:
        #     data = pickle.load(f)
        #     self.processed_chains = data['processed_chains']
        #     self.chain_sents = data['chain_sents']
        #     self.chain_group_roles = data['chain_group_roles']
        # self.tweets_embeddings = tweets
        # self.metaphor_expl_emb = metaphor_expl
        
        # print(f"Loaded {len(self.tweets)} tweets and {len(self.metaphor_expl)} metaphor explanations")
        logger.info(f"Jaccard similarity threshold: {self.jaccard_threshold}")
        
        # Verify index mapping
        # self._verify_index_mapping()
    
    def _verify_index_mapping(self):
        """Verify that indices in processed_chains match chain_sents indices."""
        max_idx = max(self.processed_chains.keys())
        if max_idx >= len(self.chain_sents):
            raise ValueError(f"Index mismatch: max processed_chains index {max_idx} >= chain_sents length {len(self.chain_sents)}")
        
        # Spot check a few mappings
        for idx in random.sample(list(self.processed_chains.keys()), min(5, len(self.processed_chains))):
            chain_text_from_dict = self.processed_chains[idx]['chain_text']
            chain_text_from_list = self.chain_sents[idx]
            if chain_text_from_dict != chain_text_from_list:
                raise ValueError(f"Text mismatch at index {idx}")
        
        print(" Index mapping verified successfully")
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts based on word tokens, excluding stop words."""
        # Define common English stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'i', 'you', 'we', 'they', 'them', 'their', 'this', 'these', 'those', 'have', 
            'had', 'been', 'being', 'do', 'does', 'did', 'can', 'could', 'should', 'would',
            'may', 'might', 'must', 'shall', 'not', 'no', 'nor', 'but', 'or', 'yet', 'so',
            'if', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose'
        }
        
        # Tokenize and filter out stop words
        words1 = {word for word in text1.lower().split() if word not in stop_words and len(word) > 1}
        words2 = {word for word in text2.lower().split() if word not in stop_words and len(word) > 1}
        
        # Calculate Jaccard similarity: intersection / union
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_cluster_statistics(self, clustering_data: Dict) -> Dict:
        """Compute basic statistics for each cluster."""

        embeddings = np.array(clustering_data['embeddings']) # numpy array of numpy arrays
        labels = clustering_data['labels']
        centroids = clustering_data['cluster_centers']        
        
        cluster_stats = {}
        
        empty_cluster_count = 0
        for cluster_id in range(clustering_data['number_cluster']):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) < 2:  # Need at least 2 points for positive examples
                empty_cluster_count += 1
                continue
                
            cluster_embeddings = embeddings[cluster_indices]
            centroid = centroids[cluster_id]
            
            # Calculate distances from points to centroid
            distances_to_centroid = euclidean_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()
            
            cluster_stats[cluster_id] = {
                'indices': cluster_indices,
                'size': len(cluster_indices),
                'distances_to_centroid': distances_to_centroid
            }
        print(f"WARNING: {empty_cluster_count} empty clusters")
        return cluster_stats
    
    
    def _sample_positive_examples(self, cluster_stats: Dict, cluster_id: int, clustering_data: Dict, emb_data) -> Tuple[int, int, bool]:
        """Sample two positive examples from a cluster."""
        cluster_indices = cluster_stats[cluster_id]['indices']
        distances = cluster_stats[cluster_id]['distances_to_centroid']
        
        # Sort by distance to centroid and take top k%
        sorted_indices = cluster_indices[np.argsort(distances)]
        top_k_percent = max(1, int(len(sorted_indices) * 0.25))
        candidate_indices = sorted_indices[:top_k_percent]
        
        # Track whether document diversity was satisfied
        doc_diversity_satisfied = True
        
        # Sample two points ensuring they don't have the same doc_id
        max_attempts = 50
        for _ in range(max_attempts):
            if len(candidate_indices) < 2:
                raise NotImplementedError
                # # Fall back to closest two if not enough candidates
                # positive_1_idx = sorted_indices[0]
                # positive_2_idx = sorted_indices[min(1, len(sorted_indices) - 1)]
                # # Check if these have different doc_ids
                # doc_id_1 = self.processed_chains[positive_1_idx]['doc_id']
                # doc_id_2 = self.processed_chains[positive_2_idx]['doc_id']
                # doc_diversity_satisfied = (doc_id_1 != doc_id_2)
                break
                
            # Sample two different indices
            sampled_indices = random.sample(candidate_indices.tolist(), 2)
            positive_1_idx, positive_2_idx = sampled_indices
            
            # # Check if they have different doc_ids
            doc_id_1 = emb_data[str(positive_1_idx)]['id']
            doc_id_2 = emb_data[str(positive_2_idx)]['id']

            
            if doc_id_1 != doc_id_2:
                # Also check Jaccard similarity - reject if too similar
                text_1 = emb_data[str(positive_1_idx)]['text']
                text_2 = emb_data[str(positive_2_idx)]['text']
                jaccard_sim = self._calculate_jaccard_similarity(text_1, text_2)
                
                if jaccard_sim < self.jaccard_threshold:
                    break
        else:
            # If we couldn't find different doc_ids in top 50%, expand search
            found_diverse_pair = False
            for i in range(len(sorted_indices) - 1):
                for j in range(i + 1, len(sorted_indices)):
                    idx1, idx2 = sorted_indices[i], sorted_indices[j]

                    doc_id_1 = emb_data[str(idx1)]['id']
                    doc_id_2 = emb_data[str(idx2)]['id']

                    if doc_id_1 != doc_id_2:
                        # Also check Jaccard similarity in expanded search
                        text_1 = emb_data[str(idx1)]['text']
                        text_2 = emb_data[str(idx2)]['text']
                        jaccard_sim = self._calculate_jaccard_similarity(text_1, text_2)
                        
                        if jaccard_sim < self.jaccard_threshold:
                            positive_1_idx, positive_2_idx = idx1, idx2
                            found_diverse_pair = True
                            break

                else:
                    continue
                break
            
            if not found_diverse_pair:
                # Last resort: take the two closest regardless of doc_id
                positive_1_idx = sorted_indices[0]
                positive_2_idx = sorted_indices[min(1, len(sorted_indices) - 1)]
                doc_diversity_satisfied = False
        
        return positive_1_idx, positive_2_idx, doc_diversity_satisfied
    
    def _sample_intruder_easy(self, clustering_data: Dict, source_cluster_id: int, 
                             cluster_stats: Dict) -> int:
        """Sample an easy intruder from top 25% points in random cluster from top 25% semantically different clusters."""
        centroids = clustering_data['cluster_centers']
        source_centroid = centroids[source_cluster_id]
        
        # Find candidate clusters (excluding source)
        candidate_clusters = [cid for cid in cluster_stats.keys() if cid != source_cluster_id]
        
        # Get distances to all candidate clusters and select top 25% farthest (most different)
        distances = euclidean_distances(source_centroid.reshape(1, -1), centroids[candidate_clusters]).flatten()
        top_25_percent = max(1, int(len(candidate_clusters) * 0.25))
        farthest_cluster_indices = np.argsort(distances)[-top_25_percent:]  # Take the farthest clusters
        different_clusters = [candidate_clusters[i] for i in farthest_cluster_indices]
        
        # Randomly select from the top 25% most different clusters
        target_cluster = random.choice(different_clusters)
        
        # Get top 25% points closest to target cluster's centroid
        target_indices = cluster_stats[target_cluster]['indices']
        target_distances = cluster_stats[target_cluster]['distances_to_centroid']
        
        # Select top 25% closest points to centroid
        top_25_percent_points = max(1, int(len(target_indices) * 0.25))
        closest_point_indices = np.argsort(target_distances)[:top_25_percent_points]
        close_points = target_indices[closest_point_indices]
        
        # Randomly select from top 25% closest points
        intruder_idx = random.choice(close_points)
        
        return intruder_idx
    
    def _sample_intruder_medium(self, clustering_data: Dict, source_cluster_id: int, 
                               cluster_stats: Dict) -> int:
        """Sample a medium intruder from top 25% points in random cluster from top 25% semantically similar clusters."""
        centroids = clustering_data['cluster_centers']
        source_centroid = centroids[source_cluster_id]
        
        # Find candidate clusters (excluding source)
        candidate_clusters = [cid for cid in cluster_stats.keys() if cid != source_cluster_id]
        
        # Get distances to all candidate clusters and select top 25% closest (most similar)
        distances = euclidean_distances(source_centroid.reshape(1, -1), centroids[candidate_clusters]).flatten()
        top_25_percent = max(1, int(len(candidate_clusters) * 0.25))
        closest_cluster_indices = np.argsort(distances)[:top_25_percent]
        similar_clusters = [candidate_clusters[i] for i in closest_cluster_indices]
        
        # Randomly select from the top 25% most similar clusters
        target_cluster = random.choice(similar_clusters)
        
        # Get top 25% points closest to target cluster's centroid
        target_indices = cluster_stats[target_cluster]['indices']
        target_distances = cluster_stats[target_cluster]['distances_to_centroid']
        
        # Select top 25% closest points to centroid
        top_25_percent_points = max(1, int(len(target_indices) * 0.25))
        closest_point_indices = np.argsort(target_distances)[:top_25_percent_points]
        close_points = target_indices[closest_point_indices]
        
        # Randomly select from top 25% closest points
        intruder_idx = random.choice(close_points)
        
        return intruder_idx
    
    def _sample_intruder_hard(self, clustering_data: Dict, source_cluster_id: int, 
                             cluster_stats: Dict) -> int:
        """Sample a hard intruder from top 25% points in the most semantically similar cluster."""
        centroids = clustering_data['cluster_centers']
        source_centroid = centroids[source_cluster_id]
        
        # Find candidate clusters (excluding source)
        candidate_clusters = [cid for cid in cluster_stats.keys() if cid != source_cluster_id]
        
        # Find the semantically closest cluster
        distances = euclidean_distances(source_centroid.reshape(1, -1), centroids[candidate_clusters]).flatten()
        closest_cluster_idx = np.argmin(distances)
        target_cluster = candidate_clusters[closest_cluster_idx]
        
        # Get top 25% points closest to the target cluster's centroid
        target_indices = cluster_stats[target_cluster]['indices']
        target_distances = cluster_stats[target_cluster]['distances_to_centroid']
        
        # Select top 25% closest points to centroid
        top_25_percent_points = max(1, int(len(target_indices) * 0.25))
        closest_point_indices = np.argsort(target_distances)[:top_25_percent_points]
        close_points = target_indices[closest_point_indices]
        
        # Randomly select from top 25% closest points
        intruder_idx = random.choice(close_points)
        
        return intruder_idx
    
    def _sample_intruder(self, clustering_data: Dict, source_cluster_id: int, 
                        cluster_stats: Dict, difficulty: str) -> int:
        """Sample an intruder based on difficulty level."""
        if difficulty == 'easy':
            return self._sample_intruder_easy(clustering_data, source_cluster_id, cluster_stats)
        elif difficulty == 'medium':
            return self._sample_intruder_medium(clustering_data, source_cluster_id, cluster_stats)
        elif difficulty == 'hard':
            return self._sample_intruder_hard(clustering_data, source_cluster_id, cluster_stats)
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")
    
    def generate_intrusion_examples(
            self, clustering_data: Dict, emb_data, met_data, total_triplets: int = 50,
            difficulty_distribution: Tuple[float, float, float] = (0.33, 0.33, 0.34)
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
        """
        Generate intrusion detection examples for two clustering methods with difficulty-based intruder sampling.
        
        Args:
            clustering_data_1: First clustering method results
            clustering_data_2: Second clustering method results  
            total_triplets: Total number of triplets to generate per method
            difficulty_distribution: Tuple of (easy_ratio, medium_ratio, hard_ratio)
            
        Returns:
            Tuple of (examples_dict_by_method, solutions_dict_by_method)
        """
        # Ensure difficulty distribution sums to 1
        easy_ratio, medium_ratio, hard_ratio = difficulty_distribution
        if abs(sum(difficulty_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Difficulty distribution must sum to 1.0, got {sum(difficulty_distribution)}")
        
        # Calculate triplets per difficulty level per method
        easy_per_method = int(total_triplets * easy_ratio)
        medium_per_method = int(total_triplets * medium_ratio)
        hard_per_method = total_triplets - easy_per_method - medium_per_method  # Remainder goes to hard
        
        self.logger.info(f"Generating {total_triplets} triplets per method:")
        self.logger.info(f"  Per difficulty: Easy={easy_per_method}, Medium={medium_per_method}, Hard={hard_per_method}")
        
        examples_by_method = {}
        solutions_by_method = {}
        
        # Process both clustering methods
        method_name = "method 0"

        self.logger.info(f"Processing {method_name}...")
        
        # Initialize lists for this method
        examples_by_method[method_name] = []
        solutions_by_method[method_name] = {}
        example_id = 0
        
        # Compute cluster statistics
        cluster_stats = self._compute_cluster_statistics(clustering_data)
        available_clusters = list(cluster_stats.keys())
        
        if len(available_clusters) == 0:
            self.logger.error(f"No valid clusters available for {method_name}")
            exit(1)

        
        # Generate examples for each difficulty level
        difficulty_counts = {
            'easy': easy_per_method,
            'medium': medium_per_method,
            'hard': hard_per_method
        }
        
        # Track cluster usage counts to maximize utilization
        cluster_usage_counts = {cluster_id: 0 for cluster_id in available_clusters}
        
        # Track document diversity failures
        doc_diversity_failures = 0
        
        for difficulty, count in difficulty_counts.items():
            print(f"  Generating {count} {difficulty} examples...")
            
            for _ in tqdm(range(count)):
                # Create weights favoring less-used clusters
                weights = []
                for cluster_id in available_clusters:
                    usage_count = cluster_usage_counts[cluster_id]
                    # Weight inversely proportional to usage count + 1
                    weight = 1.0 / (usage_count + 1)
                    weights.append(weight)
                
                # Weighted random selection for source cluster
                source_cluster = random.choices(available_clusters, weights=weights, k=1)[0]
                
                # Increment usage count
                cluster_usage_counts[source_cluster] += 1
                
                # Sample positive examples
                pos1_idx, pos2_idx, doc_diversity_satisfied = self._sample_positive_examples(cluster_stats, source_cluster, clustering_data, emb_data)
                
                # Track document diversity failures
                if not doc_diversity_satisfied:
                    doc_diversity_failures += 1
                
                # Sample intruder based on difficulty
                intruder_idx = self._sample_intruder(clustering_data, source_cluster, cluster_stats, difficulty)

                # get source and target info to include in intrusion text
                pos1_tg, pos1_ss = emb_data[str(pos1_idx)]["features"]["target_group"], emb_data[str(pos1_idx)]["features"]["source_image_schema_group"] 

                pos2_tg, pos2_ss = emb_data[str(pos2_idx)]["features"]["target_group"], emb_data[str(pos1_idx)]["features"]["source_image_schema_group"] 

                intr_tg, intr_ss = emb_data[str(pos1_idx)]["features"]["target_group"], emb_data[str(pos1_idx)]["features"]["source_image_schema_group"] 

                # Get sentence texts with character roles
                pos1_text = f"target group: {pos1_tg}, source group: {pos1_ss}, sentence: {emb_data[str(pos1_idx)]['text']}"
                pos2_text = f"target group: {pos2_tg}, source group: {pos2_ss}, sentence: {emb_data[str(pos2_idx)]['text']}"
                intruder_text = f"target group: {intr_tg}, source group: {intr_ss}, sentence: {emb_data[str(intruder_idx)]['text']}"
                
                # Create triplet with random order
                sentences = [pos1_text, pos2_text, intruder_text]
                sentence_roles = ['positive', 'positive', 'intruder']
                
                # Shuffle while tracking intruder position
                combined = list(zip(sentences, sentence_roles))
                random.shuffle(combined)
                shuffled_sentences, shuffled_roles = zip(*combined)
                
                # Find intruder position after shuffle
                intruder_position = shuffled_roles.index('intruder')
                
                # Create example dict
                example = {
                    str(i): sent for i, sent in enumerate(shuffled_sentences)
                }
                examples_by_method[method_name].append(example)
                
                # Store solution
                solutions_by_method[method_name][example_id] = {
                    'answer': intruder_position,
                    'difficulty': difficulty,
                    'method': method_name,
                    'source_cluster': int(source_cluster),
                    'positive_indices': [int(pos1_idx), int(pos2_idx)],
                    'intruder_index': int(intruder_idx)
                }
                
                example_id += 1
            
            # Shuffle the examples to mix difficulty levels
            combined_data = list(zip(examples_by_method[method_name], 
                                   solutions_by_method[method_name].values()))
            random.shuffle(combined_data)
            
            # Reassign shuffled data with new sequential IDs
            examples_by_method[method_name] = [ex for ex, _ in combined_data]
            solutions_by_method[method_name] = {
                i: {**sol, 'original_id': i} for i, (_, sol) in enumerate(combined_data)
            }
            
            print(f"  Generated and shuffled {len(examples_by_method[method_name])} examples for {method_name}")
            unique_clusters_used = len([c for c in cluster_usage_counts if cluster_usage_counts[c] > 0])
            print(f"  Unique clusters used: {unique_clusters_used} out of {len(available_clusters)} available")
            print(f"  Document diversity failures: {doc_diversity_failures} out of {len(examples_by_method[method_name])} examples")
        
        total_examples = sum(len(examples) for examples in examples_by_method.values())
        print(f"Generated {total_examples} total examples across both methods")
        return examples_by_method, solutions_by_method
    
    def save_data_separate(
            self, examples: List[Dict], 
            solutions: Dict, filename: str
    ):
        """Save examples and solutions as separate files for each method."""
        # Create file names with method suffix

        csv_path = f"{self.config.task_dir}/{filename}"
        solutions_path = csv_path.replace(".csv", "_solutions.pkl")
        
        # Save examples as tab-separated CSV with pretty-printed JSON
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['example'])  # Header
            for example in examples:
                # Pretty print JSON with each sentence on a new line
                json_str = json.dumps(example, ensure_ascii=False, indent=2)
                writer.writerow([json_str])
        self.logger.info(f"Saved intrusion data to annotate: {csv_path}")
        
        # Save solutions as pickle
        with open(solutions_path, 'wb') as f:
            pickle.dump(solutions, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(solutions)
        self.logger.info(f"Saved solutions to: {solutions_path}")


def evaluate_predictions(csv_path: str, solutions_path: str, predictions_path: str):
    """
    Evaluate predictions against ground truth.
    
    Args:
        csv_path: Path to examples CSV
        solutions_path: Path to solutions pickle file
        predictions_path: Path to predictions file (one answer per line)
    """
    # Load solutions
    with open(solutions_path, 'rb') as f:
        solutions = pickle.load(f)
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        lines = f.readlines()
    predictions = [int(line.strip()) for line in lines if line.strip()]
    
    if len(predictions) != len(solutions):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(solutions)} examples")
    
    # Calculate overall accuracy
    correct = sum(1 for i, pred in enumerate(predictions) if pred == solutions[i]['answer'])
    total_accuracy = correct / len(predictions) * 100
    
    print(f"Overall Accuracy: {correct}/{len(predictions)} = {total_accuracy:.2f}%")
    
    # Breakdown by difficulty
    difficulty_stats = {}
    for difficulty in ['easy', 'medium', 'hard']:
        difficulty_indices = [i for i, sol in solutions.items() if sol.get('difficulty') == difficulty]
        if difficulty_indices:
            difficulty_correct = sum(1 for i in difficulty_indices if predictions[i] == solutions[i]['answer'])
            difficulty_total = len(difficulty_indices)
            difficulty_accuracy = difficulty_correct / difficulty_total * 100
            difficulty_stats[difficulty] = {
                'correct': difficulty_correct,
                'total': difficulty_total,
                'accuracy': difficulty_accuracy
            }
            print(f"{difficulty.capitalize()} Accuracy: {difficulty_correct}/{difficulty_total} = {difficulty_accuracy:.2f}%")
    
    # Breakdown by method
    method_stats = {}
    for method in ['method_1', 'method_2']:
        method_indices = [i for i, sol in solutions.items() if sol['method'] == method]
        if method_indices:
            method_correct = sum(1 for i in method_indices if predictions[i] == solutions[i]['answer'])
            method_total = len(method_indices)
            method_accuracy = method_correct / method_total * 100
            method_stats[method] = {
                'correct': method_correct,
                'total': method_total,
                'accuracy': method_accuracy
            }
            print(f"{method.replace('_', ' ').title()} Accuracy: {method_correct}/{method_total} = {method_accuracy:.2f}%")
    
    return {
        'overall': {'correct': correct, 'total': len(predictions), 'accuracy': total_accuracy},
        'by_difficulty': difficulty_stats,
        'by_method': method_stats
    }


def main(args):

    env_vars = load_env()
    logger = get_logger("sample_intrusion")  # default info logging level

    # create config
    config = ExperimentConfig(
        task_type="annotation",
        task_name="intrusion",
        dataset=args.dataset,
        logger=logger,
        env_vars=env_vars,
        skip_load=True,
        labeled_only=args.labeled_only,
    )
    
    # LOAD EMBEDDINGS
    # '4328': {'id': '1004009278749868032_0_34_35_1', 'emb': array([ 1.69521943e-02,  3.23939696e-02, -3.35079730e-02, ...])}
    embedding_path = f"{env_vars["RESULTS_DIR"]}/gen_embeddings/{args.embedding_model}/{config.dataset_out_name}_{args.embedding_name}_embeddings_features.pkl"
    with open(embedding_path, 'rb') as f:
        met_emb_data = pickle.load(f)
    logger.info(f"Loaded {len(met_emb_data)} embeddings...")

    # LOAD MET CLASSIFICATION RESULTS
    met_class_path = f"{env_vars["RESULTS_DIR"]}/metaphor_classification/llm/{config.dataset_out_name}_source_verb_target_noun_binary_met_class_qwen.json"
    met_class_res = load_json(met_class_path)["data"]


    # LOAD CLUSTER DATA
    cluster_path = f"{env_vars['RESULTS_DIR']}/cluster_embeddings/{args.cluster_method}/{config.dataset_out_name}/{args.embedding_name}_clusters_{args.k}.pkl"
    if args.cluster_method == "pckmeans":
        cluster_path = cluster_path.replace(".pkl", f"_w{args.w}.pkl")

    with open(cluster_path, "rb") as f:
        met_cluster_data = pickle.load(f)

    assert len(met_emb_data) == len(met_cluster_data['labels']), f"{len(met_emb_data)} != {len(met_cluster_data['labels'])}, met-level embedding/cluster file mismatch"
    
    # Determine counts based on user input (exact counts take precedence)
    if args.exact_counts:
        # Use exact counts provided by user
        easy_count, medium_count, hard_count = args.exact_counts
        total_triplets = easy_count + medium_count + hard_count
        difficulty_distribution = (easy_count / total_triplets, medium_count / total_triplets, hard_count / total_triplets)
        logger.info(f"Using exact counts: Easy={easy_count}, Medium={medium_count}, Hard={hard_count}")
    else:
        # Use ratio-based distribution with auto-detection for even splits
        total_triplets = args.total_triplets
        if total_triplets in [100, 150]:
            # Auto-detect even split for common totals
            even_split = total_triplets // 3
            remainder = total_triplets % 3
            difficulty_distribution = (even_split / total_triplets, even_split / total_triplets, (even_split + remainder) / total_triplets)
            logger.info(f"Auto-detected even split for {total_triplets} examples: {even_split}, {even_split}, {even_split + remainder}")
        else:
            difficulty_distribution = tuple(args.difficulty_split)
            logger.info(f"Using ratio-based distribution with {total_triplets} total triplets: {difficulty_distribution}")


    # Generate intrusion data
    generator = IntrusionDataGenerator(config, logger, jaccard_threshold=args.jaccard_threshold)
    examples, solutions = generator.generate_intrusion_examples(
        met_cluster_data, met_emb_data, met_class_res,
        total_triplets=total_triplets,
        difficulty_distribution=difficulty_distribution
    )
    
    # Save data!
    filename = f"{config.dataset_out_name}_{args.embedding_name}_clusters_{args.cluster_method}_{args.k}.csv"
    if args.cluster_method == "pckmeans":
        filename = filename.replace(".csv", f"_w{args.w}.csv")
    generator.save_data_separate(examples["method 0"], solutions["method 0"], filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate intrusion detection task data")

    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument("--dataset", required=True, help="dataset to cluster embeddings for", choices=['tweets_immigration', 'subframes_immigration', 'subframes_guns', 'subframes_abortion', 'lcc-large', 'podcasts'],)
    parser.add_argument("--labeled_only", action="store_true")
    parser.add_argument("--embedding_model", default="sbert", help="model used to generate embeddings, only sbert implemented for now")
    parser.add_argument("--embedding_name", required=True, help="name to distinguish the embeddings we're working with ([dataset_name]_[embedding_name]_embeddings.pkl")

    # clustering details
    parser.add_argument("--cluster_method", default="pckmeans", help="name to distinguish the embeddings we're working with ([dataset_name]_[embedding_name]_embeddings.pkl")
    parser.add_argument("--k", required=True, help="number of clusters")
    parser.add_argument("--w", required=False, help="weight used for pckmeans clustering (required for pckmeans)")
    

    # parser.add_argument('--output-csv', default='intrusion_examples.tsv', help='Output CSV path')
    # parser.add_argument('--output-solutions', default='intrusion_solutions.pkl', help='Output solutions path')
    
    parser.add_argument('--total-triplets', type=int, default=150,
                       help='Total number of triplets to generate per method (default: 150)')
    
    parser.add_argument('--difficulty-split', nargs=3, type=float, default=[0.33, 0.33, 0.34],
                       help='Difficulty distribution: easy medium hard (must sum to 1.0, default: 0.33 0.33 0.34)')
    
    parser.add_argument('--exact-counts', nargs=3, type=int, metavar=('EASY', 'MEDIUM', 'HARD'),
                       help='Exact number of examples per difficulty: easy medium hard (overrides --difficulty-split and --total-triplets)')
    
    parser.add_argument('--evaluate', action='store_true', help='Path to predictions file for evaluation')
    
    parser.add_argument('--jaccard-threshold', type=float, default=0.6,
                       help='Jaccard similarity threshold for rejecting similar positive pairs (default: 0.8)')
    
    args = parser.parse_args()
    main(args)