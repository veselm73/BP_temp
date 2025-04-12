# Dependencies
import os
import re
import string
import numpy as np
import pandas as pd
import scipy
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
from sklearn.preprocessing import normalize, LabelEncoder, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.tree import plot_tree
import torch
from transformers import AutoTokenizer, AutoModel
from ucimlrepo import fetch_ucirepo
from gensim.models import Word2Vec
import tspg
import umap



class WeightNormalizer:
    @staticmethod
    def normalize(weights):
        weights_array = np.array(weights)
        min_val = np.min(weights_array)
        max_val = np.max(weights_array)

        if max_val == min_val:
            return np.zeros_like(weights_array)

        return (weights_array - min_val) / (max_val - min_val)


class DataPreprocessor:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="most_frequent")

    def encode_categorical(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le

    def impute(self):
        self.df = pd.DataFrame(
            self.imputer.fit_transform(self.df), columns=self.df.columns
        )

    def get_features_and_target(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y

    def preprocess(self):
        self.encode_categorical()
        self.impute()
        return self.get_features_and_target()


class ModelTrainer:
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=73)

    def train(self, X, y):
        self.model.fit(X, y)

    def get_feature_weights(self, feature_names):
        return pd.DataFrame({
            'Feature': feature_names,
            'Weight': WeightNormalizer.normalize(self.model.feature_importances_)
        })


#Data augemntation sentence creation
#TRAINING ON ALL DATA!!!
class SentenceFormatter:
    def __init__(self, dataframe, descriptive_format=True):

        self.df = dataframe.copy()
        self.descriptive_format = descriptive_format

    def _prettify_column_names(self):

        return [' '.join(col.split('_')) for col in self.df.columns]

    def generate_sentences(self):

        sentences = []
        if self.descriptive_format:
            pretty_columns = self._prettify_column_names()
            for _, row in self.df.iterrows():
                sentence = [f"{col_name} is {val}" for col_name, val in zip(pretty_columns, row)]
                sentences.append(sentence)
        else:
            for _, row in self.df.iterrows():
                sentence = [str(val) for val in row]
                sentences.append(sentence)

        return sentences


#Word2Vec/Bert training and clustering
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class Word2VecModel:
    def __init__(self, sentences, vector_size=36, window=5, min_count=1):
        self.model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)

    def get_embedding(self, sentence, weighted_embeddings=False, feature_weights=None):
        word_embeddings = [self.model.wv[word] for word in sentence if word in self.model.wv]
        if not word_embeddings:
            return np.zeros(self.model.vector_size)

        if weighted_embeddings and feature_weights is not None:
            weights = np.array(feature_weights['Weight'].tolist())
            return np.average(word_embeddings, axis=0, weights=weights)
        return np.mean(word_embeddings, axis=0)

class BertEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, batch_size=16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).half()
        self.batch_size = batch_size

    def get_embedding(self, sentences):
        if isinstance(sentences[0], list):
            sentences = [" ".join(sentence) for sentence in sentences]

        embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

        return np.vstack(embeddings)

class SentenceEmbedder:
    def __init__(self, method="word2vec", **kwargs):
        self.method = method.lower()

        if self.method == "word2vec":
            self.model = Word2VecModel(**kwargs)
        elif self.method == "bert":
            self.model = BertEmbedder(**kwargs)
        else:
            raise ValueError("Unsupported embedding method. Use 'word2vec' or 'bert'.")

    def generate_embeddings(self, sentences, normalize=False, weighted_embeddings=False, feature_weights=None):
        if self.method == "word2vec":
            embeddings = np.array([
                self.model.get_embedding(s, weighted_embeddings=weighted_embeddings, feature_weights=feature_weights)
                for s in sentences
            ])
        elif self.method == "bert":
            embeddings = self.model.get_embedding(sentences)
        else:
            raise ValueError("Unsupported embedding method.")

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms

        return embeddings

class BiasedEmbedder:
    def __init__(self, alpha=0.1, random_state=73):
        self.alpha = alpha
        self.random_state = random_state

    def perpendicular_bias_embeddings(self, embeddings, targets):
        embeddings_biased = np.copy(embeddings)
        unique_classes, targets_encoded = np.unique(targets, return_inverse=True)
        np.random.seed(self.random_state)

        if len(unique_classes) > embeddings.shape[1]:
            raise ValueError("Number of unique classes exceeds embedding dimension, cannot create perpendicular biases.")

        bias_vectors = np.eye(len(unique_classes), embeddings.shape[1])

        for i, class_idx in enumerate(targets_encoded):
            embeddings_biased[i] += self.alpha * bias_vectors[class_idx]

        return embeddings_biased


    def one_hot_bias_embeddings(self, embeddings, targets):
        unique_classes, targets_encoded = np.unique(targets, return_inverse=True)

        embeddings_extended = np.zeros((embeddings.shape[0], embeddings.shape[1] + len(unique_classes)))
        embeddings_extended[:, :embeddings.shape[1]] = embeddings

        for i, class_idx in enumerate(targets_encoded):
            embeddings_extended[i, embeddings.shape[1] + class_idx] = self.alpha

        return embeddings_extended


class ClusteringModel:
    def __init__(self, n_clusters, method="agglomerative", random_state=73, dendrogram_cut=None):
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.dendrogram_cut = dendrogram_cut
        self.linkage_matrix = None

    def fit_predict(self, embeddings):
        clustering_methods = {
            "spherical_kmeans": self._spherical_kmeans_clustering,
            "agglomerative": self._agglomerative_clustering,
            "tspg": self._tspg_clustering
        }

        if self.method in clustering_methods:
            return clustering_methods[self.method](embeddings)

        raise ValueError(f"Unsupported clustering method: {self.method}")

    def _spherical_kmeans_clustering(self, X):
        np.random.seed(self.random_state)
        X = normalize(X, norm='l2', axis=1)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        cluster_centers = X[indices]

        for _ in range(200):
            similarities = X @ cluster_centers.T
            labels = np.argmax(similarities, axis=1)
            new_centroids = np.array([
                normalize(X[labels == j].mean(axis=0).reshape(1, -1))
                if np.any(labels == j) else cluster_centers[j]
                for j in range(self.n_clusters)
            ]).squeeze()

            if np.linalg.norm(new_centroids - cluster_centers) < 1e-4:
                break
            cluster_centers = new_centroids

        return labels

    def _tspg_clustering(self, embeddings):
        self.linkage_matrix = linkage(embeddings, method='average', metric='cosine')
        if self.dendrogram_cut:
            labels = fcluster(self.linkage_matrix, self.dendrogram_cut, criterion='distance') - 1
        elif self.n_clusters:
            labels = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust') - 1
        else:
            labels, _ = tspg.tspg(embeddings, self.n_clusters, distance="cos", num_tsp=min(50, embeddings.shape[0] // 200), dtype="vec")
            labels = np.array(labels) - 1
        return labels

    def _agglomerative_clustering(self, embeddings):
        self.linkage_matrix = linkage(embeddings, method='average', metric='cosine')
        if self.dendrogram_cut:
            labels = fcluster(self.linkage_matrix, self.dendrogram_cut, criterion='distance') - 1
        elif self.n_clusters:
            labels = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust') - 1
        else:
            labels = AgglomerativeClustering(n_clusters=self.n_clusters, metric='cosine', linkage='average').fit_predict(embeddings)
        return labels

    def plot_dendrogram(self, max_d=None, truncate_mode='level', p=30):
        if self.linkage_matrix is None:
            return

        plt.figure(figsize=(24, 6))
        dendrogram(self.linkage_matrix, truncate_mode=truncate_mode, p=p,
                   color_threshold=max_d, distance_sort='ascending', show_leaf_counts=True, no_labels=True)

        ylim = max_d * 1.1 if max_d else np.max(self.linkage_matrix[:, 2]) * 1.1
        plt.ylim(0, ylim)

        plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
        plt.ylabel("Distance", fontsize=14)
        plt.tight_layout()
        plt.show()
        

class DecisionTreeTrainer:
    def __init__(self, features_raw, labels, max_depth, subgroup_size_limit, test_size, random_state=73, print_acc=True):
        self.features_raw = features_raw
        self.labels = labels
        self.max_depth = max_depth
        self.subgroup_size_limit = subgroup_size_limit
        self.test_size = test_size
        self.random_state = random_state
        self.print_acc = print_acc

        self.encoder = OneHotEncoder()
        self.decision_trees = {}
        self.accuracies = {}

        self._prepare_data()
        self._train_trees()

    def _prepare_data(self):
        features_classify = self.features_raw.assign(cluster=self.labels)
        X = features_classify.drop(columns=['cluster'])
        X_encoded = self.encoder.fit_transform(X)
        y = features_classify['cluster']

        self.X_df = pd.DataFrame(X_encoded.toarray(), columns=self.encoder.get_feature_names_out())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_df, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def _train_trees(self):
        for cluster in sorted(self.y_train.unique()):
            y_train_binary = (self.y_train == cluster).astype(int)
            y_test_binary = (self.y_test == cluster).astype(int)

            clf = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.subgroup_size_limit, random_state=self.random_state)
            clf.fit(self.X_train, y_train_binary)

            self.decision_trees[cluster] = clf

            y_pred = clf.predict(self.X_test)
            accuracy = accuracy_score(y_test_binary, y_pred)
            self.accuracies[cluster] = accuracy

            if self.print_acc:
                print(f"Accuracy for Cluster {cluster}: {accuracy:.4f}")
                print(classification_report(y_test_binary, y_pred))

    def get_trees(self):
        return self.decision_trees

    def get_accuracies(self):
        return self.accuracies

    def plot_decision_trees(self, cluster_index=None):
        if cluster_index is not None:
            if cluster_index in self.decision_trees:
                plt.figure(figsize=(12, 8))
                plot_tree(self.decision_trees[cluster_index], feature_names=self.X_df.columns, filled=True, rounded=True)
                plt.title(f"Decision Tree for Cluster {cluster_index}")
                plt.show()
            else:
                print(f"Cluster {cluster_index} not found.")
        else:
            for cluster, tree in self.decision_trees.items():
                plt.figure(figsize=(12, 8))
                plot_tree(tree, feature_names=self.X_df.columns, filled=True, rounded=True)
                plt.title(f"Decision Tree for Cluster {cluster}")
                plt.show()


# Subgroup creation and quality evaluation
def Quality_metric(tp, fp, TP, FP, description_length, description_length_limit, difference_limit, subgroup_size_limit):
    difference = abs(tp / (tp + fp) - (TP / (TP + FP)))
    if difference < difference_limit or description_length > description_length_limit or (tp + fp) < subgroup_size_limit:
        return np.NaN
    return difference / description_length

class ClusterRuleExtractor:
    def __init__(self, data_frame, target_df, cluster_labels, feature_names, decision_trees, trees_acc, tree_accuracy_threshold):
        self.data_frame = data_frame
        self.target_df = target_df
        self.cluster_labels = cluster_labels
        self.feature_names = feature_names
        self.decision_trees = decision_trees
        self.trees_acc = trees_acc
        self.tree_accuracy_threshold = tree_accuracy_threshold

        self.target_column = target_df.columns[0]

        self.cluster_frequencies = {}
        self.cluster_percentages = {}
        self.rules_dict = {}
        self.unextractable_clusters = []
        self.WRAcc_dict = {}

        self.tp_dict = {}
        self.fp_dict = {}
        self.TP_dict = {}
        self.FP_dict = {}

        self._extract_rules_for_clusters()

    def _extract_rules_from_tree(self, tree, node=0, path=None, rule_dict=None):
        if path is None:
            path = []
        if rule_dict is None:
            rule_dict = defaultdict(list)

        left_child = tree.tree_.children_left[node]
        right_child = tree.tree_.children_right[node]
        threshold = tree.tree_.threshold[node]
        feature = tree.tree_.feature[node]
        value = tree.tree_.value[node]

        if left_child == -1 and right_child == -1:
            predicted_class = int(np.argmax(value[0]))
            rule_dict[predicted_class].append(path)
            return rule_dict

        feature_name = self.feature_names[feature]

        if left_child != -1:
            self._extract_rules_from_tree(tree, left_child, path + [f"{feature_name} <= {threshold:.2f}"], rule_dict,)

        if right_child != -1:
            self._extract_rules_from_tree(tree, right_child, path + [f"{feature_name} > {threshold:.2f}"], rule_dict,)

        return rule_dict

    def _extract_rules_for_clusters(self):
        N = len(self.target_df)
        class_counts_total = self.target_df[self.target_column].value_counts().to_dict()
        cluster_class_counts = defaultdict(lambda: defaultdict(int))

        for i, cluster in enumerate(self.cluster_labels):
            label = self.target_df[self.target_column].iloc[i]
            cluster_class_counts[cluster][label] += 1

        for cluster, class_counts in sorted(cluster_class_counts.items()):
            total_count = sum(class_counts.values())
            dominant_class = max(class_counts, key=class_counts.get)
            tp = class_counts[dominant_class]
            fp = total_count - tp

            self.cluster_frequencies[cluster] = total_count
            self.cluster_percentages[cluster] = {cls: round(count / total_count, 2) for cls, count in class_counts.items()}

            TP = class_counts_total[dominant_class]
            FP = N - TP
            self.tp_dict[cluster] = tp
            self.fp_dict[cluster] = fp
            self.TP_dict[cluster] = TP
            self.FP_dict[cluster] = FP

            self.WRAcc_dict[cluster] = ((total_count / N) * (tp / total_count - TP / N))

            if cluster in self.decision_trees:
                accuracy = self.trees_acc.get(cluster, 0)
                if accuracy < self.tree_accuracy_threshold:
                    self.unextractable_clusters.append(cluster)
                    self.rules_dict[cluster] = ["Accuracy below threshold"]
                else:
                    rule_sets = self._extract_rules_from_tree(self.decision_trees[cluster])
                    rules = []

                    for pred_class, paths in rule_sets.items():
                        if pred_class != 1:
                            continue
                        for path in paths:
                            rules.append(" AND ".join(path))

                    self.rules_dict[cluster] = rules
                    if len(rules) == 0:
                        self.rules_dict[cluster] = ["No rules extracted"]
                        self.unextractable_clusters.append(cluster)
            else:
                self.unextractable_clusters.append(cluster)
                self.rules_dict[cluster] = ["No decision tree for this cluster"]

    def print_summary(self):
        print("Cluster Rule Summary:")
        for cluster in sorted(self.cluster_frequencies.keys()):
            support = self.cluster_frequencies[cluster]
            percentages_str = ", ".join(
                f"'{cls}': {perc}" for cls, perc in self.cluster_percentages[cluster].items()
            )
            WRAcc = round(self.WRAcc_dict[cluster], 6)
            print(f"Cluster {cluster}: support = {support}, {percentages_str}, WRAcc = {WRAcc}")

            rules = self.rules_dict.get(cluster, ["No rules extracted"])
            for rule in rules:
                print(f"  - {rule}")

        if self.unextractable_clusters:
            print("Unextractable Clusters:", self.unextractable_clusters)



class SubgroupCreator:
    def __init__(self, X_df, target_df, target_value, target_column, file_path):
        self.X_df = X_df
        self.target_df = target_df
        self.target_value = target_value
        self.target_column = target_column
        self.file_path = file_path
        self.tp_dict = {}
        self.fp_dict = {}
        self.TP_dict = {}
        self.FP_dict = {}
        self.WRAcc_dict = {}
        self.Quality_dict = {}
        self.subgroups = {}
        self.total_coverage = 0
        self.subgroup_descriptions = {}

    def evaluate_rule(self, rule_str, covered_indices_set):
        parts = rule_str.split(" AND ")
        mask = pd.Series(True, index=self.X_df.index)

        description = []

        for part in parts:
            feature, operator, value = part.rsplit(maxsplit=2)
            value = float(value)

            if ">" in operator:
                feature_name = feature.rsplit("_", 1)[0]
                category_value = feature.split("_")[-1]
                description.append(f"{feature_name} = '{category_value}'")
                mask &= self.X_df[feature] > 0.5

            elif "<=" in operator:
                feature_name = feature.rsplit("_", 1)[0]
                category_value = feature.split("_")[-1]
                description.append(f"{feature_name} ≠ '{category_value}'")
                mask &= self.X_df[feature] <= 0.5

        covered_indices = set(self.X_df[mask].index) - covered_indices_set
        covered_indices_set.update(covered_indices)
        covered_targets = self.target_df.loc[list(covered_indices)]

        tp = (covered_targets == self.target_value).sum().item()
        fp = (covered_targets != self.target_value).sum().item()
        TP = (self.target_df == self.target_value).sum().item()
        FP = (self.target_df != self.target_value).sum().item()

        WRAcc = ((tp + fp) / (TP + FP)) * ((tp / (tp + fp)) - (TP / (TP + FP)))

        return tp, fp, TP, FP, WRAcc, covered_indices, description

    def try_merge_rules(self, rules_list):
        from collections import defaultdict

        parsed_rules = []
        for rule in rules_list:
            if "No rules" in rule or "Accuracy below" in rule:
                continue
            conditions = sorted(rule.split(" AND "))
            parsed_rules.append((rule, conditions))

        condition_buckets = defaultdict(list)
        for original_rule, conditions in parsed_rules:
            for i in range(len(conditions)):
                key = tuple(conditions[:i] + conditions[i+1:])
                condition_buckets[key].append((original_rule, conditions))

        merged = set()
        final_rules = []

        for shared_conditions, group in condition_buckets.items():
            if len(group) == 2:
                (_, conds1), (_, conds2) = group
                differing = list(set(conds1) ^ set(conds2))
                if len(differing) == 2:
                    feat1 = differing[0].rsplit(" ", 2)[0]
                    feat2 = differing[1].rsplit(" ", 2)[0]
                    if feat1 == feat2:
                        merged_rule = " AND ".join(shared_conditions)
                        final_rules.append(merged_rule)
                        merged.update([group[0][0], group[1][0]])
            else:
                for rule, _ in group:
                    if rule not in merged:
                        final_rules.append(rule)

        untouched_rules = [r for r, _ in parsed_rules if r not in merged]
        final_rules.extend(untouched_rules)

        return list(set(final_rules))


    def evaluate_all_clusters(self, subgroups):
        with open(self.file_path, "w") as f:
            for cluster, rules_list in subgroups.rules_dict.items():
                if cluster not in subgroups.unextractable_clusters:
                    rules_list = self.try_merge_rules(rules_list)
                    covered_indices_set = set()

                    for idx, rule_str in enumerate(rules_list):
                        if "No rules extracted" in rule_str or "Accuracy below threshold" in rule_str:
                            continue

                        letter_index = string.ascii_lowercase[idx]
                        subgroup_label = f"{cluster}.{letter_index}"

                        tp, fp, TP, FP, WRAcc, covered_indices, description = self.evaluate_rule(rule_str, covered_indices_set)

                        self.tp_dict[subgroup_label] = tp
                        self.fp_dict[subgroup_label] = fp
                        self.TP_dict[subgroup_label] = TP
                        self.FP_dict[subgroup_label] = FP
                        self.WRAcc_dict[subgroup_label] = WRAcc
                        self.subgroups[subgroup_label] = covered_indices
                        self.total_coverage += tp + fp

                        description_str = ", ".join(str(item) for item in description)
                        self.Quality_dict[subgroup_label] = Quality_metric(
                            tp=tp,
                            fp=fp,
                            TP=TP,
                            FP=FP,
                            description_length=len(description),
                            description_length_limit=description_length_limit,
                            difference_limit=difference_limit,
                            subgroup_size_limit=subgroup_size_limit,
                        )
                        self.subgroup_descriptions[subgroup_label] = description_str

                        f.write(
                            f"Description: [{description_str}], Target: {target_column} = '{self.target_value}' ; "
                            f"QuMe = {float(self.Quality_dict[subgroup_label]) if self.Quality_dict[subgroup_label] is not None else 0:.8f} ; "
                            f"WRAcc = {float(WRAcc) if WRAcc is not None else 0:.8f} ; "
                            f"tp = {tp} ; fp = {fp} ; TP = {TP} ; FP = {FP}\n"
                        )
            f.write(f"Total coverage: {self.total_coverage / len(self.target_df)}\n")


# Clusters visualization
def plot_sentence_embeddings(embeddings, cluster_labels, clustering_method=None):
    plt.figure(figsize=(12, 8))

    reducer = umap.UMAP(n_components=2, random_state=73, n_neighbors=200, min_dist=0.1, metric='cosine')
    embeddings_2d = reducer.fit_transform(np.array(embeddings))

    cluster_labels = np.array(cluster_labels)

    cmap = plt.get_cmap('hsv', len(set(cluster_labels)))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap=cmap, alpha=0.5, s=2)

    plt.colorbar(scatter, label='Cluster Labels')
    plt.title(f"Sentence Embeddings Visualization ({clustering_method})")
    plt.show()

#BSD vs SDUEBA comparisson
def parse_and_compute_quality(file_path, description_length_limit, difference_limit, subgroup_size_limit):
    output_file = "results_BSD_.txt"

    with open(file_path, 'r') as file, open(output_file, 'w') as output:
        for line in file:
            match = re.search(r"(Description: \[.*?\]), Target: (.*?) ; Quality Measure WRAcc = ([\d.-]+) ; tp = (\d+) ; fp = (\d+) ; TP = (\d+) ; FP = (\d+)", line)

            if match:
                description = match.group(1)
                target = match.group(2)
                WRAcc = float(match.group(3))
                tp = int(match.group(4))
                fp = int(match.group(5))
                TP = int(match.group(6))
                FP = int(match.group(7))

                description_text = description[13:-1]
                description_length = description_text.count(',') + 1 if description_text else 1

                quality_score = Quality_metric(tp, fp, TP, FP, description_length, description_length_limit, difference_limit, subgroup_size_limit)

                output_line = f"{description}, Target: {target} ; QuMe = {quality_score:.8f} ; WRAcc = {WRAcc:.8f} ; tp = {tp} ; fp = {fp} ; TP = {TP} ; FP = {FP}\n"
                output.write(output_line)


def compute_coverage(filename, quality_measure, k, dataset, print_subgroups=False):
    with open(filename, 'r') as file:
        lines = file.readlines()

    subgroups = []
    instance_coverage = {}

    for line in lines:
        match = re.search(rf"{quality_measure}\s*=\s*([-+]?\d*\.?\d+)\s*;.*tp\s*=\s*(\d+)\s*;\s*fp\s*=\s*(\d+)", line)
        if match:
            quality = float(match.group(1))
            tp = int(match.group(2))
            fp = int(match.group(3))

            desc_match = re.search(r"Description: \[(.*?)\]", line)
            if desc_match:
                description = desc_match.group(1).split(", ")
            else:
                description = []

            mask = pd.Series(True, index=dataset.index)
            for condition in description:
                try:
                    feature, value = condition.split(" = ")
                    value = value.strip("'")

                    if feature not in dataset.columns:
                        continue

                    mask &= dataset[feature] == value
                except ValueError:
                    continue

            covered_instances = set(dataset[mask].index)

            for instance in covered_instances:
                instance_coverage[instance] = instance_coverage.get(instance, 0) + 1

            subgroups.append((quality, covered_instances, description, line.strip()))

    subgroups.sort(reverse=True, key=lambda x: x[0])
    top_k_subgroups = subgroups[:k]

    covered_instances = set()
    filtered_instance_coverage = {}

    for _, instance_ids, _, _ in top_k_subgroups:
        covered_instances.update(instance_ids)
        for instance in instance_ids:
            filtered_instance_coverage[instance] = filtered_instance_coverage.get(instance, 0) + 1

    coverage = len(covered_instances) / len(dataset) if len(dataset) else 0
    overlapping_instances = sum(1 for count in filtered_instance_coverage.values() if count > 1)
    overlap_ratio = overlapping_instances / len(covered_instances) if covered_instances else 0

    if print_subgroups:
        print("Top-k subgroups:")
        for _, _, _, subgroup in top_k_subgroups:
            print(subgroup)

    return coverage, overlap_ratio