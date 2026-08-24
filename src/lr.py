import os
import random
import statistics
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
from numpy import mean, std

from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances

import gc


class RegressionModel:
    def __init__(self, random_seed):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def filter_metaphors_by_centroid_proximity(self, clustering_data, top_k_percent=50.0):
        """
        Filter event metaphors to keep only those in the top k% closest to their cluster centroids.
        
        Args:
            clustering_data: Dictionary containing embeddings, labels, and cluster_centers
            processed_metaphors: Dictionary containing processed metaphor data
            top_k_percent: Percentage of closest metaphors to keep per cluster (default 50%)
            
        Returns:
            Set of metaphor indices to keep
        """
        embeddings = clustering_data['embeddings']
        labels = clustering_data['labels']
        centroids = clustering_data['cluster_centers']
        
        metaphors_to_keep = set()
        
        # For each cluster, find metaphors closest to centroid
        for cluster_id in range(clustering_data['number_cluster']):
            # Get indices of metaphors in this cluster
            cluster_metaphor_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_metaphor_indices) == 0:
                continue
                
            # Get embeddings for metaphors in this cluster
            # cluster_embeddings = embeddings[cluster_metaphor_indices]
            cluster_embeddings = [embeddings[idx] for idx in cluster_metaphor_indices]
            cluster_centroid = centroids[cluster_id].reshape(1, -1)
            
            # Calculate distances to centroid
            distances = euclidean_distances(cluster_embeddings, cluster_centroid).flatten()
            
            # Sort by distance and keep top k%
            sorted_indices = np.argsort(distances)
            num_to_keep = max(1, int(len(cluster_metaphor_indices) * top_k_percent / 100.0))
            closest_indices = sorted_indices[:num_to_keep]
            
            # Add the actual metaphor indices (not relative to cluster)
            metaphors_to_keep.update(cluster_metaphor_indices[closest_indices])
            
        print(f"Filtered from {len(labels)} to {len(metaphors_to_keep)} metaphors ({len(metaphors_to_keep)/len(labels)*100:.1f}%)", flush=True)
        return list(metaphors_to_keep)

    def regression(self, data, label_key, use_centroid_filtering=False, centroid_top_k_percent=50.0, add_target_info=False):
    
        print("Running regression...", flush=True)
        if use_centroid_filtering:
            print(f"# Centroid filtering: top {centroid_top_k_percent}% of metaphors used", flush=True)
        else:
            print(f"# No centroid filtering: all metaphors used", flush=True)
        
        # map classes to labels 
        label_encoder = preprocessing.LabelEncoder()
        data[label_key] = label_encoder.fit_transform(data[label_key])
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        for k, v in label_mapping.items():
            print(v, ' : ', k, flush=True)


        data = data.drop(labels=["doc_id"], axis=1)
        X = data.filter(like='Cluster')
        y = data[label_key]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
        
        print(f"Train set size: {len(X_train)}", flush=True)
        print(f"Test set size: {len(X_test)}", flush=True)
        
        # Print label counts for training set
        train_counts = pd.Series(y_train).value_counts().sort_index()
        print("Training set label counts:", flush=True)
        for label, count in train_counts.items():
            print(f"  Label {label}: {count}", flush=True)
        
        # Print label counts for test set
        test_counts = pd.Series(y_test).value_counts().sort_index()
        print("Test set label counts:", flush=True)
        for label, count in test_counts.items():
            print(f"  Label {label}: {count}", flush=True)

        # Create pipeline to avoid data leakage in cross-validation
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='lbfgs', penalty='l2', C=0.5, max_iter=1000))
        ])

        # define the model evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=self.random_seed)
        # evaluate the model and collect the scores (pipeline handles scaling within each fold)
        accuracy_scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        f1_scores = cross_val_score(pipeline, X_train, y_train, scoring='f1_macro', cv=cv, n_jobs=-1)
        # report the model performance
        print('Mean Train Accuracy: %.2f (%.2f)' % (mean(accuracy_scores) * 100, std(accuracy_scores) * 100), flush=True)
        print('Mean Train F1: %.2f (%.2f)' % (mean(f1_scores) * 100, std(f1_scores) * 100), flush=True)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_accuracy = round(metrics.accuracy_score(y_test.to_numpy(), y_pred) * 100, 2)
        f1_score = round(metrics.f1_score(y_test.to_numpy(), y_pred, average='macro', zero_division=0) * 100, 2)
        classification_report = metrics.classification_report(y_test.to_numpy(), y_pred, zero_division=0, output_dict=True)

        print(f"Test Accuracy: {test_accuracy}", flush=True)
        print(f"F1 Score: {f1_score}", flush=True)
        # print(classification_report, flush=True)

        # Print tab-separated row with all metrics
        train_accuracy = round(mean(accuracy_scores) * 100, 2)
        train_f1 = round(mean(f1_scores) * 100, 2)
        print(f"{train_accuracy}\t{test_accuracy}\t{train_f1}\t{f1_score}", flush=True)

        model = pipeline.named_steps['classifier']
        weights = model.coef_ / pipeline.named_steps['scaler'].scale_

        del pipeline
            
        return weights, classification_report


    def format_dataset(self, clustering_data, metaphors, doc_data, label_key, use_centroid_filtering=False, \
                       centroid_top_k_percent=50.0, filter_labels=False, add_target_info=False):

        print("Creating dataset...", flush=True)

        # Filter metaphors if centroid filtering is enabled
        if use_centroid_filtering:
            print(f"Filtering metaphors to top {centroid_top_k_percent}% closest to centroids...", flush=True)
            metaphors_to_keep = self.filter_metaphors_by_centroid_proximity(
                clustering_data, centroid_top_k_percent
            )
            metaphors_to_keep = [str(m) for m in metaphors_to_keep]
            assert type(metaphors_to_keep[0]) == type(list(metaphors.keys())[0]), f"Type {type(metaphors_to_keep[0])} doesn't match type {type(list(metaphors.keys())[0])}"

            metaphors = {k: v for k, v in metaphors.items() if k in metaphors_to_keep}

        # create a dataframe where each row corresponds to a document
        # there is a column for each cluster where the value is the number of metaphors in the document belonging to that cluster
        # there's also a label column (value we're predicting)
        data_dict = [] # collect document dicts here
        for id, doc_info in tqdm(doc_data.items(), desc="creating df"):
            assert type(id) is str, "wrong id type (should be str) for doc_data"
            doc_row = {"doc_id": id}
            for i in range(clustering_data['number_cluster']):
                key = 'Cluster ' + str(i) # add column for each cluster
                doc_row[key] = 0
    
            if add_target_info:
                unique_targets = list(set([m_info["target_group"] for m_info in metaphors.values()]))
                for target in unique_targets:
                    if target != "":
                        key = f"Target {target}"
                        doc_row[key] = 0

            # now add the metaphors to their respective column clusters
            for met_idx, met_info in metaphors.items():
                met_info_doc_id = met_info["id"].split("_")[0]
                if met_info_doc_id == id: # belongs to document, count it up
                    met_idx = int(met_idx)
                    cluster_idx = clustering_data["labels"][met_idx]
                    doc_row[f"Cluster {cluster_idx}"] += 1

                    if add_target_info:
                        target = met_info["target_group"]
                        if target != "":
                            doc_row[f"Target {target}"] += 1
            
            # add the label we're predicting
            if label_key: 
                doc_row[label_key] = doc_info[label_key]

            data_dict.append(doc_row)

        data = pd.DataFrame.from_dict(data_dict)

            


        return data

    def run_regression(self, clustering_data, metaphors, doc_data, label_list, \
                       k, og_results_dir, w_cl = None, filter_labels=False, add_target_info=False):
        """
        Run regression twice: once on top 25% closest to centroids, then on all metaphors.
        Prints results in tab-separated format: Train Acc, Test Acc, Train F1, Test F1.
        """
        # # Run 1: Top 25% closest to centroids
        # data_filtered = self.format_dataset(
        #     clustering_data, metaphors, doc_data, label_key, use_centroid_filtering=True, 
        #     centroid_top_k_percent=25.0, filter_labels=filter_labels, use_met_score=use_met_score)
        
        # t25_classification_report = self.regression(
        #     data_filtered, 
        #     label_key,
        #     use_centroid_filtering=True, 
        #     centroid_top_k_percent=25.0)

        
        # Run 2: All metaphors
        data_all = self.format_dataset(
            clustering_data, metaphors, doc_data, None, use_centroid_filtering=False, 
            filter_labels=filter_labels, add_target_info=add_target_info
        )

        for label_key in label_list:

            # add label to data before passing to regression
            data_all[label_key] = data_all["doc_id"].apply(lambda x: doc_data[x][label_key])

            weights, classification_report = self.regression(
                data_all, label_key, use_centroid_filtering=False, add_target_info=add_target_info
            )

  
            results_dir = og_results_dir.replace("/LABEL/", f"/{label_key}/")
            os.makedirs(results_dir, exist_ok=True)
            reg_file_name = f"k_{k}.csv"
            if w_cl:
                reg_file_name = reg_file_name.replace(".csv", f"_w{w_cl}.csv")

            reg_df = pd.DataFrame(classification_report).transpose()
            reg_df.to_csv(f"{results_dir}/{reg_file_name}")

            # save weights to pickle
            weight_file_name = reg_file_name.replace(".csv", "_weights.pkl")
            weight_path = f"{results_dir}/{weight_file_name}"
            with open(weight_path, 'wb') as f:
                pickle.dump(weights, f)

            gc.collect()

        
        # # Format results: Top 25% block then All metaphors block (Train Acc, Test Acc, Train F1, Test F1)
        # values = [f"{train_acc_filtered:.2f}", f"{test_acc_filtered:.2f}", f"{train_f1_filtered:.2f}", f"{f1_filtered:.2f}",
        #           f"{train_acc_all:.2f}", f"{test_acc_all:.2f}", f"{train_f1_all:.2f}", f"{f1_all:.2f}"]
        
        # print('\t'.join(values))

        # return classification_report

    def format_combined_features_dataset(self, met_clustering_data, tweet_clustering_data, metaphors, tweets, doc_data, \
                               label_key, use_centroid_filtering=False, centroid_top_k_percent=50.0, filter_labels=False, \
                               add_target_info=False):
        if use_centroid_filtering:
            raise NotImplementedError
        
        data_dict = [] # collect document dicts here
        for id, doc_info in tqdm(doc_data.items(), desc="creating df"):
            
            doc_row = {"doc_id": id}
            for i in range(met_clustering_data['number_cluster']):
                key = 'Met Cluster ' + str(i)
                doc_row[key] = 0
            
            if add_target_info:
                unique_targets = list(set([m_info["target_group"] for m_info in metaphors.values()]))
                for target in unique_targets:
                    if target != "":
                        key = f"Target {target}"
                        doc_row[key] = 0

            
            # add the label we're predicting
            if label_key: 
                doc_row[label_key] = doc_info[label_key]

            # now add the metaphors to their respective column clusters
            met_filter_str = f"{id}"
            for met_idx, met_info in metaphors.items():
                assert type(met_info["id"]) is str, "wrong id type for metaphors"

                if met_info["id"].startswith(met_filter_str): # belongs to document, count it up
                    met_idx = int(met_idx)
                    cluster_idx = met_clustering_data["labels"][met_idx]
                    doc_row[f"Met Cluster {cluster_idx}"] += 1

                    if add_target_info:
                        target = met_info["target_group"]
                        if target != "":
                            doc_row[f"Target {target}"] += 1
            
            for i in range(tweet_clustering_data['number_cluster']):
                key = f'Tweet Cluster {i}' # add column for each cluster
                doc_row[key] = 0

            # now add tweet info
            id_filter_str = str(id)
            for t_idx, t_info in tweets.items():
                if t_info["id"].startswith(id_filter_str): # belongs to document, count it up
                    t_idx = int(t_idx)
                    cluster_idx = tweet_clustering_data["labels"][t_idx]
                    doc_row[f"Tweet Cluster {cluster_idx}"] += 1
                    
            data_dict.append(doc_row)

        data = pd.DataFrame.from_dict(data_dict)
        return data

    
    
    def run_combined_features(self, met_clustering_data, tweet_clustering_data, metaphors, tweets, doc_data, ml_k, tl_k, label_list, \
                       og_results_dir, filter_labels=False, add_target_info=False):
        
        data_all = self.format_combined_features_dataset(
            met_clustering_data, tweet_clustering_data, metaphors, tweets, doc_data, None, use_centroid_filtering=False, 
            filter_labels=filter_labels, add_target_info=add_target_info
        )

        for label_key in label_list:

            # add label to data before passing to regression
            data_all[label_key] = data_all["doc_id"].apply(lambda x: doc_data[x][label_key])

            weights, classification_report = self.regression(
                data_all, label_key, use_centroid_filtering=False, add_target_info=add_target_info
            )

            results_dir = og_results_dir.replace("/LABEL/", f"/{label_key}/")
            os.makedirs(results_dir, exist_ok=True)
            reg_file_name = f"k_t{tl_k}_m{ml_k}.csv"
            reg_df = pd.DataFrame(classification_report).transpose()
            reg_df.to_csv(f"{results_dir}/{reg_file_name}")

            # save weights to pickle
            weight_file_name = reg_file_name.replace(".csv", "_weights.pkl")
            weight_path = f"{results_dir}/{weight_file_name}"
            with open(weight_path, 'wb') as f:
                pickle.dump(weights, f)

            gc.collect()
    