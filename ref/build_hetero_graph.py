import os
import gc
import json
import numpy as np
import torch
import networkx as nx
from typing import Dict, Tuple, Optional, List, Union, Any
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Features, Sequence, Value
from sklearn.metrics import pairwise_distances
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm
from argparse import ArgumentParser
from utils.sample_k_shot import sample_k_shot
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# --- Constants ---
DEFAULT_K_SHOT = 8                                  # 3-16 shot
DEFAULT_DATASET_NAME = "politifact"                 # politifact, gossipcop
DEFAULT_EMBEDDING_TYPE = "deberta"                  # Default embedding for news nodes (bert, distilbert, roberta, deberta, bigbird)
# --- Edge Policies Parameters ---
DEFAULT_EDGE_POLICY = "knn_test_isolated"           # For news-news edges (label_aware_knn, knn, knn_test_isolated)
DEFAULT_K_NEIGHBORS = 5                             # For knn edge policy
# --- Unlabeled Node Sampling Parameters ---
DEFAULT_SAMPLE_UNLABELED_FACTOR = 5                 # for unlabeled node sampling (train_unlabeld_nodes = num_classes * k_shot * sample_unlabeled_factor)
DEFAULT_MULTI_VIEW = 0                              # for multi-view edge policy
DEFAULT_INTERACTION_EDGE_MODE = "edge_attr"         # for interaction edge policy (edge_attr, edge_type): how news_node and interaction_node are connected
# --- Graphs and Plots Directories ---
DEFAULT_SEED = 42
DEFAULT_DATASET_CACHE_DIR = "dataset"
DEFAULT_GRAPH_DIR = "graphs_hetero"
DEFAULT_BATCH_SIZE = 50

# --- Utility Functions ---
def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


# --- HeteroGraphBuilder Class ---
class HeteroGraphBuilder:
    """
    Builds heterogeneous graph datasets ('news', 'interaction')
    for few-shot fake news detection.
    """

    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = DEFAULT_EMBEDDING_TYPE,
        edge_policy: str = DEFAULT_EDGE_POLICY,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        partial_unlabeled: bool = False,
        sample_unlabeled_factor: int = DEFAULT_SAMPLE_UNLABELED_FACTOR,
        pseudo_label: bool = False,
        pseudo_label_cache_path: str = None,
        multi_view: int = DEFAULT_MULTI_VIEW,
        enable_dissimilar: bool = False,
        ensure_test_labeled_neighbor: bool = False,
        interaction_embedding_field: str = "interaction_embeddings_list",
        interaction_tone_field: str = "interaction_tones_list",
        interaction_edge_mode: str = DEFAULT_INTERACTION_EDGE_MODE,
        dataset_cache_dir: str = DEFAULT_DATASET_CACHE_DIR,
        seed: int = DEFAULT_SEED,
        output_dir: str = DEFAULT_GRAPH_DIR,
        no_interactions: bool = False,
    ):
        """Initialize the HeteroGraphBuilder."""
        self.dataset_name = dataset_name.lower()
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.text_embedding_field = f"{embedding_type}_embeddings"
        self.interaction_embedding_field = interaction_embedding_field
        self.interaction_tone_field = interaction_tone_field
        self.interaction_edge_mode = interaction_edge_mode
        self.no_interactions = no_interactions
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        ## Sampling
        self.partial_unlabeled = partial_unlabeled
        self.sample_unlabeled_factor = sample_unlabeled_factor
        self.pseudo_label = pseudo_label
        if pseudo_label_cache_path:
            self.pseudo_label_cache_path = pseudo_label_cache_path
        else:
            self.pseudo_label_cache_path = f"utils/pseudo_label_cache_{self.dataset_name}.json"
        self.multi_view = multi_view
        self.enable_dissimilar = enable_dissimilar
        self.ensure_test_labeled_neighbor = ensure_test_labeled_neighbor

        if self.edge_policy == "label_aware_knn":
            self.pseudo_label = True
        
        if self.pseudo_label:
            self.partial_unlabeled = True
        
        self.dataset_cache_dir = dataset_cache_dir
        self.seed = seed

        # Setup and create directories
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Set device
        np.random.seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # Initialize state
        self.dataset = None     # Load from dataset
        self.graph_metrics = {} # Store analysis results
        # Selected Indices
        self.train_labeled_indices = None
        self.train_unlabeled_indices = None
        self.test_indices = None

        self.pseudo_selected_indices = np.array([])
        self.pseudo_selected_labels = np.array([])
        self.pseudo_selected_confidences = np.array([])
    
        self.news_orig_to_new_idx = None # Mapping for mask creation (global2local)
        self.tone2id = {}

    def _tone2id(self, tone):
        if tone not in self.tone2id:
            self.tone2id[tone] = len(self.tone2id)
        return self.tone2id[tone]

    def load_dataset(self) -> None:
        """Load dataset and perform initial checks."""
        print(f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings...")
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"
        # download from huggingface and cache to local path
        local_hf_dir = os.path.join(self.dataset_cache_dir, f"{self.dataset_name}_hf")
        if os.path.exists(local_hf_dir):
            print(f"Loading dataset from local path: {local_hf_dir}")
            dataset = load_from_disk(local_hf_dir)
        else:
            print(f"Loading dataset from huggingface: {hf_dataset_name}")
            dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists")
            dataset.save_to_disk(local_hf_dir)

        # dataset: DatasetDict
        self.dataset = {
            "train": dataset["train"], 
            "test": dataset["test"]
        }
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]

        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        unique_labels = set(self.train_data['label']) | set(self.test_data['label']) # 0: real, 1: fake
        self.num_classes = len(unique_labels)   # 2
        self.total_labeled_size = self.k_shot * self.num_classes    # for k-shot learning

        print(f"\nOriginal dataset size: Train={self.train_size}, Test={self.test_size}")
        print(f"  Detected Labels: {unique_labels} ({self.num_classes} classes)")
        print(f"  Train labeled set: {self.k_shot}-shot * {self.num_classes} classes = {self.total_labeled_size} total labeled nodes")

    def build_hetero_graph(self, test_batch_indices=None) -> Optional[HeteroData]:
        """
        Build a heterogeneous graph including both nodes and edges.
        Pipeline:
        1. Build empty graph
            - train_labeled_nodes (train_labeled_mask from training set)
            - train_unlabeled_nodes (train_unlabeled_mask from training set)
            - test_nodes (test_mask from test set)
        2. Build edges (based on edge policy)
        3. Update graph data with edges
        """
        print("\nStarting Heterogeneous Graph Construction...")

        data = HeteroData()

        # --- Select News Nodes (k-shot train labeled nodes, train unlabeled nodes, test nodes) ---
        # 1. Sample k-shot labeled nodes from train set (with cache)
        print(f"  ===== Sampling k-shot train labeled nodes from train set =====")
        train_labeled_indices_cache_path = f"utils/{self.dataset_name}_{self.k_shot}_shot_train_labeled_indices_{self.seed}.json"
        if os.path.exists(train_labeled_indices_cache_path):
            with open(train_labeled_indices_cache_path, "r") as f:
                train_labeled_indices = json.load(f)
            train_labeled_indices = np.array(train_labeled_indices)
            print(f"  Loaded k-shot indices from cache: {train_labeled_indices_cache_path}")
        else:
            train_labeled_indices, _ = sample_k_shot(self.train_data, self.k_shot, self.seed)
            train_labeled_indices = np.array(train_labeled_indices)
            with open(train_labeled_indices_cache_path, "w") as f:
                json.dump(train_labeled_indices.tolist(), f)
            print(f"Saved k-shot indices to cache: {train_labeled_indices_cache_path}")
        self.train_labeled_indices = train_labeled_indices
        print(f"  Selected {len(train_labeled_indices)} train labeled nodes: {train_labeled_indices} ...")

        # 2. Get train_unlabeled_nodes (all train nodes not in train_labeled_indices)
        ## Preselect all the train_unlabeled_nodes from all train nodes
        print(f"  ===== Sampling train unlabeled nodes from train set =====")
        all_train_indices = np.arange(len(self.train_data))
        train_unlabeled_indices = np.setdiff1d(all_train_indices, train_labeled_indices, assume_unique=True)
        # print(f"  All train nodes: {all_train_indices}")
        # print(f"  Preselected {len(train_unlabeled_indices)} unlabeled nodes: {train_unlabeled_indices} ...")

        # --- Sample train_unlabeled_nodes if required ---
        if self.partial_unlabeled:
            num_to_sample = min(self.num_classes * self.k_shot * self.sample_unlabeled_factor, len(train_unlabeled_indices))
            print(f"  Sampling {num_to_sample}({self.num_classes}*{self.k_shot}*{self.sample_unlabeled_factor}) train_unlabeled_nodes (num_classes={self.num_classes}, k={self.k_shot}, factor={self.sample_unlabeled_factor}) from {len(train_unlabeled_indices)} available.")
            if self.pseudo_label:
                print("  Using pseudo-label based sampling (sorted by confidence)...")
                try:
                    with open(self.pseudo_label_cache_path, "r") as f:
                        pseudo_data = json.load(f)
                    # Build a map: index -> (pseudo_label, confidence)
                    pseudo_label_map = {int(item["index"]): (int(item["pseudo_label"]), float(item.get("score", 1.0))) for item in pseudo_data}
                    # Group all train_unlabeled_indices by pseudo label, and sort by confidence
                    pseudo_label_groups = {label: [] for label in range(self.num_classes)}
                    for idx in train_unlabeled_indices:
                        if int(idx) in pseudo_label_map:
                            label, conf = pseudo_label_map[int(idx)]
                            pseudo_label_groups[label].append((idx, conf))
                    # Sort each group by confidence (descending)
                    for label in pseudo_label_groups:
                        pseudo_label_groups[label].sort(key=lambda x: -x[1])
                    # Sample top-N from each group
                    samples_per_class = num_to_sample // self.num_classes
                    sampled_indices = []
                    sampled_labels = []
                    sampled_confidences = []
                    for label in range(self.num_classes):
                        group = pseudo_label_groups[label]
                        n_samples = min(samples_per_class, len(group))
                        selected = group[:n_samples]
                        for idx, conf in selected:
                            sampled_indices.append(idx)
                            sampled_labels.append(label)
                            sampled_confidences.append(conf)
                    # If not enough, randomly sample from the remaining
                    if len(sampled_indices) < num_to_sample:
                        remaining = num_to_sample - len(sampled_indices)
                        remaining_indices = list(set(train_unlabeled_indices) - set(sampled_indices))
                        if remaining_indices:
                            additional = np.random.choice(remaining_indices, size=min(remaining, len(remaining_indices)), replace=False)
                            for idx in additional:
                                # Fallback: assign label/confidence if available, else -1/1.0
                                if int(idx) in pseudo_label_map:
                                    label, conf = pseudo_label_map[int(idx)]
                                else:
                                    label, conf = -1, 1.0
                                sampled_indices.append(idx)
                                sampled_labels.append(label)
                                sampled_confidences.append(conf)
                    train_unlabeled_indices = np.array(sampled_indices)
                    # Store for downstream label-aware KNN edge construction
                    self.pseudo_selected_indices = np.array(sampled_indices)
                    self.pseudo_selected_labels = np.array(sampled_labels)
                    self.pseudo_selected_confidences = np.array(sampled_confidences)
                    print(f"  Sampled {len(train_unlabeled_indices)} unlabeled nodes using pseudo labels (sorted by confidence)")
                except Exception as e:
                    print(f"Warning: Error during pseudo-label sampling: {e}. Falling back to random sampling.")
                    train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
                    # Fallback: store empty arrays for downstream
                    self.pseudo_selected_indices = np.array([])
                    self.pseudo_selected_labels = np.array([])
                    self.pseudo_selected_confidences = np.array([])
            else:
                train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
        
        self.train_unlabeled_indices = train_unlabeled_indices
        print(f"  Selected {len(train_unlabeled_indices)} train unlabeled nodes: {train_unlabeled_indices[:2*self.k_shot]} ...")

        # 3. Get all test node indices
        print(f"  ===== Sampling test nodes from test set =====")
        if test_batch_indices is not None:
            test_indices = np.array(test_batch_indices)
        else:
            test_indices = np.arange(len(self.test_data))
        self.test_indices = test_indices
        print(f"  Selected {len(test_indices)} test nodes: {test_indices[:2*self.k_shot]} ...")

        print(f"  ===== Building news nodes =====")
        # 4. Extract features and labels for each group
        train_labeled_emb = np.array(self.train_data.select(train_labeled_indices.tolist())[self.text_embedding_field])
        train_labeled_label = np.array(self.train_data.select(train_labeled_indices.tolist())["label"])
        train_unlabeled_emb = np.array(self.train_data.select(train_unlabeled_indices.tolist())[self.text_embedding_field])
        train_unlabeled_label = np.array(self.train_data.select(train_unlabeled_indices.tolist())["label"])
        test_emb = np.array(self.test_data.select(test_indices.tolist())[self.text_embedding_field])
        test_label = np.array(self.test_data.select(test_indices.tolist())["label"])

        # 5. Concatenate all nodes
        x = torch.tensor(np.concatenate([train_labeled_emb, train_unlabeled_emb, test_emb]), dtype=torch.float)
        y = torch.tensor(np.concatenate([train_labeled_label, train_unlabeled_label, test_label]), dtype=torch.long)

        # 6. Build masks
        num_train_labeled = len(train_labeled_indices)      # train labeled nodes
        num_train_unlabeled = len(train_unlabeled_indices)  # train unlabeled nodes
        num_test = len(test_indices)                        # test nodes
        num_nodes = num_train_labeled + num_train_unlabeled + num_test  # all nodes

        train_labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_unlabeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_labeled_mask[:num_train_labeled] = True
        train_unlabeled_mask[num_train_labeled:num_train_labeled+num_train_unlabeled] = True
        test_mask[num_train_labeled+num_train_unlabeled:] = True

        data['news'].x = x
        data['news'].y = y
        data['news'].num_nodes = num_nodes
        data['news'].train_labeled_mask = train_labeled_mask
        data['news'].train_unlabeled_mask = train_unlabeled_mask
        data['news'].test_mask = test_mask
        
        print(f"  Total news nodes: {num_nodes}")
        print(f"    - 'news' features shape: {data['news'].x.shape}")
        print(f"    - 'news' labels shape: {data['news'].y.shape}")
        print(f"    - 'news' num_nodes: {data['news'].num_nodes}")
        print(f"    - 'news' train_labeled_nodes: {data['news'].train_labeled_mask.sum()}")
        print(f"    - 'news' train_unlabeled_nodes: {data['news'].train_unlabeled_mask.sum()}")
        print(f"    - 'news' test_nodes: {data['news'].test_mask.sum()}")
        print(f"    - 'news' masks created: Train Labeled={train_labeled_mask.sum()}, Train Unlabeled={train_unlabeled_mask.sum()}, Test={test_mask.sum()}")

        # --- Prepare 'interaction' Node Features ---
        if not self.no_interactions:
            print(f"  ===== Building interaction nodes for news nodes ('news' - 'has_interaction' - 'interaction') =====")
            num_interactions_per_news = 20
            num_interaction_nodes = num_nodes * num_interactions_per_news
            print(f"  Preparing 'interaction' nodes for {num_nodes} news nodes...")
            print(f"  Each news node has {num_interactions_per_news} 'interaction' nodes")
            print(f"  Total 'interaction' nodes: {num_interaction_nodes}")
            if self.interaction_edge_mode == "edge_attr":
                self._add_interaction_edges_with_attr(data, train_labeled_indices, train_unlabeled_indices, test_indices, num_nodes, num_interactions_per_news, num_interaction_nodes)
            elif self.interaction_edge_mode == "edge_type":
                self._add_interaction_edges_by_type(data, train_labeled_indices, train_unlabeled_indices, test_indices, num_nodes, num_interactions_per_news, num_interaction_nodes)
            else:
                raise ValueError(f"Unknown interaction_edge_mode: {self.interaction_edge_mode}")

        # --- Create Edges for 'news' - 'news' nodes---
        print(f"  ===== Building edges between 'news' and 'news' ('news' - 'similar_to' - 'news') =====")
        news_embeddings = data['news'].x.cpu().numpy()

        if self.edge_policy == "knn":
            sim_edge_index, sim_edge_attr, _, _ = self._build_knn_edges(news_embeddings, self.k_neighbors)
            # === Build 'news' - 'similar_to' - 'news' edges ===
            data['news', 'similar_to', 'news'].edge_index = sim_edge_index
            if sim_edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
            print(f"    - Created {sim_edge_index.shape[1]} 'news <-> news' similar edges.")
        elif self.edge_policy == "knn_test_isolated":
            # Use test isolation with mutual KNN for high-quality connections
            self._build_safe_test_isolated_edges(data, news_embeddings)
        elif self.edge_policy == "label_aware_knn":
            # Use the pre-sampled pseudo_selected_indices, labels, and confidences for label-aware KNN
            pseudo_indices = getattr(self, 'pseudo_selected_indices', np.array([]))
            pseudo_labels = getattr(self, 'pseudo_selected_labels', np.array([]))
            pseudo_confidence = getattr(self, 'pseudo_selected_confidences', np.array([]))
            num_train_labeled = len(train_labeled_indices)
            num_train_unlabeled = len(pseudo_indices)
            num_test = len(test_indices)
            num_nodes = num_train_labeled + num_train_unlabeled + num_test
            # Local indices for each group
            train_labeled_idx_local = np.arange(num_train_labeled)
            train_unlabeled_idx_local = np.arange(num_train_labeled, num_train_labeled + num_train_unlabeled)
            test_idx_local = np.arange(num_train_labeled + num_train_unlabeled, num_nodes)
            edge_src = []
            edge_dst = []
            edge_attr = []
            # --- Train labeled nodes: only connect to train_unlabeled nodes with the same pseudo label ---
            for label in range(self.num_classes):
                labeled_mask = train_labeled_label == label
                pseudo_mask = pseudo_labels == label
                labeled_idx = train_labeled_idx_local[labeled_mask]
                pseudo_idx = train_unlabeled_idx_local[pseudo_mask]
                if len(pseudo_idx) == 0 or len(labeled_idx) == 0:
                    continue
                labeled_emb = news_embeddings[labeled_idx]
                pseudo_emb = news_embeddings[pseudo_idx]
                sim = cosine_similarity(labeled_emb, pseudo_emb)
                for i, src in enumerate(labeled_idx):
                    k = min(self.k_neighbors, sim.shape[1])
                    if k == 0:
                        continue
                    topk = np.argpartition(-sim[i], k-1)[:k]
                    for j in topk:
                        dst = pseudo_idx[j]
                        edge_src.append(src)
                        edge_dst.append(dst)
                        edge_attr.append(sim[i, j])
            # --- Test nodes: connect to all other nodes (train_labeled, train_unlabeled, test) using KNN ---
            if num_test > 0:
                all_idx = np.arange(num_nodes)
                for i, src in enumerate(test_idx_local):
                    test_emb = news_embeddings[src].reshape(1, -1)
                    sim = cosine_similarity(test_emb, news_embeddings)[0]
                    # Exclude self
                    sim[src] = -np.inf
                    k = min(self.k_neighbors, num_nodes - 1)
                    if k <= 0:
                        continue
                    topk = np.argpartition(-sim, k-1)[:k]
                    for dst in topk:
                        edge_src.append(src)
                        edge_dst.append(dst)
                        edge_attr.append(sim[dst])
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) if edge_attr else None
            data['news', 'similar_to', 'news'].edge_index = edge_index
            if edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = edge_attr
            print(f"    - Created {edge_index.shape[1]} 'news -> news' label-aware similar edges.")
            # --- Add low-level KNN (k=2) for all nodes as a separate edge type ---
            low_k = 2
            low_edge_index, low_edge_attr, _, _ = self._build_knn_edges(news_embeddings, low_k)
            data['news', 'low_level_knn_to', 'news'].edge_index = low_edge_index
            if low_edge_attr is not None:
                data['news', 'low_level_knn_to', 'news'].edge_attr = low_edge_attr
            print(f"    - Created {low_edge_index.shape[1]} 'news <-> news' low-level KNN edges (k={low_k}).")

        # --- Ensure Test Nodes Have a Labeled Neighbor (Robust Undirected Check) ---
        if self.ensure_test_labeled_neighbor:
            print(f"  Attempting to ensure each test node has at least one 'similar_to' neighbor from train_labeled set (robust undirected check)...")
            train_labeled_idx_local = np.arange(num_train_labeled)
            test_idx_local = np.arange(num_train_labeled + num_train_unlabeled, num_nodes)
            edge_index = data['news', 'similar_to', 'news'].edge_index
            edge_attr = data['news', 'similar_to', 'news'].edge_attr if hasattr(data['news', 'similar_to', 'news'], 'edge_attr') else None
            # symmetrize for checking only
            edge_index_check = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            edge_set = set((src.item(), dst.item()) for src, dst in edge_index_check.t())
            test_to_labeled_added = 0
            new_edges = []
            new_attrs = []
            for test_local in test_idx_local:
                has_labeled = any(
                    (test_local, labeled_local) in edge_set or (labeled_local, test_local) in edge_set
                    for labeled_local in train_labeled_idx_local
                )
                if not has_labeled:
                    test_emb = news_embeddings[test_local].reshape(1, -1)
                    labeled_emb = news_embeddings[train_labeled_idx_local]
                    sim = cosine_similarity(test_emb, labeled_emb)[0]
                    best_idx = np.argmax(sim)
                    best_labeled_local = train_labeled_idx_local[best_idx]
                    # add only one direction for now
                    new_edges.append([test_local, best_labeled_local])
                    if edge_attr is not None:
                        sim_val = sim[best_idx]
                        new_attrs.append([sim_val])
                    test_to_labeled_added += 1
            if new_edges:
                new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
                edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)
                if edge_attr is not None:
                    new_attrs_tensor = torch.tensor(new_attrs, dtype=torch.float)
                    edge_attr = torch.cat([edge_attr, new_attrs_tensor], dim=0)
                # FIX: Assign back to the graph data structure
                data['news', 'similar_to', 'news'].edge_index = edge_index
                if edge_attr is not None:
                    data['news', 'similar_to', 'news'].edge_attr = edge_attr
            print(f"    - Patched {test_to_labeled_added} test nodes with at least one labeled neighbor.")

        # --- Multi-view edge construction ---
        if self.multi_view > 1:
            emb = data['news'].x
            dim = emb.shape[1]
            print(f"    - Multi-view embedding dim: {dim}")
            assert dim % self.multi_view == 0, f"Embedding dim {dim} not divisible by multi_view {self.multi_view}"
            sub_dim = dim // self.multi_view
            
            # Get test mask for test isolation
            test_mask = data['news'].test_mask.cpu().numpy()
            train_mask = ~test_mask
            
            for v in range(self.multi_view):
                sub_emb = emb[:, v*sub_dim:(v+1)*sub_dim].cpu().numpy()
                
                if self.edge_policy == "knn_test_isolated":
                    # Get global indices
                    train_global_idx = np.where(train_mask)[0]
                    test_global_idx = np.where(test_mask)[0]
                    
                    # Split embeddings based on global indices
                    train_emb = sub_emb[train_global_idx]
                    test_emb = sub_emb[test_global_idx]
                    
                    # Initialize edges for this view
                    edge_src, edge_dst = [], []
                    attr_list = []
                    
                    # 1. Build edges between train nodes
                    if len(train_emb) > 0:
                        train_sim_idx_local, train_sim_attr, _, _ = self._build_knn_edges(train_emb, self.k_neighbors)
                        if train_sim_idx_local.shape[1] > 0:
                            # Map local indices back to global indices
                            edge_src.extend(train_global_idx[train_sim_idx_local[0]])
                            edge_dst.extend(train_global_idx[train_sim_idx_local[1]])
                            if train_sim_attr is not None:
                                attr_list.extend(train_sim_attr.squeeze().tolist())

                    # 2. Build edges from test nodes to train nodes
                    if len(test_emb) > 0 and len(train_emb) > 0:
                        test_sim_idx_local, test_sim_attr, _, _ = self._build_knn_edges(test_emb, self.k_neighbors, target_embeddings=train_emb)
                        if test_sim_idx_local.shape[1] > 0:
                            # Map local indices back to global indices
                            edge_src.extend(test_global_idx[test_sim_idx_local[0]]) # source is test set
                            edge_dst.extend(train_global_idx[test_sim_idx_local[1]]) # destination is train set
                            if test_sim_attr is not None:
                                attr_list.extend(test_sim_attr.squeeze().tolist())
                    
                    # Create final edge indices for this view
                    sim_idx = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                    sim_attr = torch.tensor(attr_list, dtype=torch.float).unsqueeze(1) if attr_list else None
                else:
                    # Original behavior for other edge policies
                    sim_idx, sim_attr, _, _ = self._build_knn_edges(sub_emb, self.k_neighbors)
                
                # Make undirected
                sim_idx = torch.cat([sim_idx, sim_idx[[1,0],:]], dim=1)
                if sim_attr is not None:
                    sim_attr = torch.cat([sim_attr, sim_attr], dim=0)
                
                sim_type = ("news", f"similar_to_sub{v+1}", "news")
                data[sim_type].edge_index = sim_idx
                if sim_attr is not None:
                    data[sim_type].edge_attr = sim_attr
                print(f"    - Created {sim_idx.shape[1]} 'news <-> news' similar_sub{v+1} edges.")

        # --- Post-process: Add Dissimilar Edges (Universal Logic) ---
        if self.enable_dissimilar:
            print(f"  ===== Building dissimilar edges ('news' - 'dissimilar_to' - 'news') =====")
            self._add_dissimilar_edges_universal(data, news_embeddings)

        # --- Post-process: Add Multi-view Dissimilar Edges ---
        if self.multi_view > 1 and self.enable_dissimilar:
            print(f"  ===== Building multi-view dissimilar edges =====")
            emb = data['news'].x
            dim = emb.shape[1]
            sub_dim = dim // self.multi_view
            
            for v in range(self.multi_view):
                sub_emb = emb[:, v*sub_dim:(v+1)*sub_dim].cpu().numpy()
                
                if self.edge_policy == "knn_test_isolated":
                    # Build multi-view dissimilar edges respecting test isolation
                    self._build_safe_multiview_dissimilar_edges(data, sub_emb, v+1)
                else:
                    # Use _build_mutual_knn_edges to get proper dissimilar edges
                    _, _, dis_idx, dis_attr = self._build_mutual_knn_edges(sub_emb, self.k_neighbors)
                    
                    if dis_idx is not None and dis_idx.shape[1] > 0:
                        # Make dissimilar edges undirected if not already
                        reverse_edges = dis_idx[[1, 0], :]
                        existing_edges = set(map(tuple, dis_idx.T.tolist()))
                        reverse_edge_tuples = set(map(tuple, reverse_edges.T.tolist()))
                        
                        if not reverse_edge_tuples.issubset(existing_edges):
                            dis_idx = torch.cat([dis_idx, reverse_edges], dim=1)
                            if dis_attr is not None:
                                dis_attr = torch.cat([dis_attr, dis_attr], dim=0)
                        
                        dis_type = ("news", f"dissimilar_to_sub{v+1}", "news")
                        data[dis_type].edge_index = dis_idx
                        if dis_attr is not None:
                            data[dis_type].edge_attr = dis_attr
                        print(f"    - Created {dis_idx.shape[1]} 'news <-> news' dissimilar_sub{v+1} edges.")
                    else:
                        print(f"    - Warning: No dissimilar edges created for sub-view {v+1}")

        # --- Final symmetrize and unique ---
        ## Get edge_index and edge_attr from 'news' - 'similar_to' - 'news' edge type
        edge_index = data['news', 'similar_to', 'news'].edge_index
        edge_attr = data['news', 'similar_to', 'news'].edge_attr if hasattr(data['news', 'similar_to', 'news'], 'edge_attr') else None

        ## symmetrize the edge_index and edge_attr
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        ## reassign edge_index and edge_attr to the graph data structure
        data['news', 'similar_to', 'news'].edge_index = edge_index
        if edge_attr is not None:
            data['news', 'similar_to', 'news'].edge_attr = edge_attr
        
        print("Heterogeneous graph construction complete.")

        return data


    def _build_knn_edges(self, embeddings: np.ndarray, k: int, target_embeddings: np.ndarray = None, is_test: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build KNN edges for 'news'-'news' edges.
        Args:
            embeddings: numpy array of shape (num_nodes, embedding_dim)
            k: number of nearest neighbors to consider
            target_embeddings: optional numpy array of shape (num_target_nodes, embedding_dim)
                             if provided, only find neighbors from target_embeddings
            is_test: whether these are test nodes (if True, can't connect to other test nodes)
        Returns:
            sim_edge_index: torch tensor of shape (2, num_similar_edges)
            sim_edge_attr: torch tensor of shape (num_similar_edges, 1)
            dis_edge_index: torch tensor of shape (2, num_dissimilar_edges)
            dis_edge_attr: torch tensor of shape (num_dissimilar_edges, 1)
        """
        print(f"    Building KNN graph (k={k}) for 'news'-'news' edges...")

        num_nodes = embeddings.shape[0]
        if target_embeddings is not None:
            num_target = target_embeddings.shape[0]
            k = min(k, num_target)  # Adjust k if it's too large
        else:
            k = min(k, num_nodes - 1)  # Adjust k if it's too large

        if k <= 0:
            return (torch.zeros((2,0), dtype=torch.long), None, torch.zeros((2,0), dtype=torch.long), None)

        # Calculate pairwise distances
        try:
            if target_embeddings is not None:
                distances = pairwise_distances(embeddings, target_embeddings, metric="cosine", n_jobs=-1)
            else:
                distances = pairwise_distances(embeddings, metric="cosine", n_jobs=-1)
        except Exception as e:
            print(f"      Error calculating pairwise distances: {e}. Using single core.")
            if target_embeddings is not None:
                distances = pairwise_distances(embeddings, target_embeddings, metric="cosine")
            else:
                distances = pairwise_distances(embeddings, metric="cosine")

        # Initialize lists for similar edges
        sim_rows, sim_cols, sim_data = [], [], []
        
        # For each node, find k nearest neighbors
        for i in tqdm(range(num_nodes), desc=f"    Finding {k} nearest neighbors", leave=False, ncols=100):
            dist_i = distances[i].copy()
            
            # For test nodes, prevent connections to other test nodes
            if is_test and target_embeddings is not None:
                dist_i[num_target:] = np.inf
                
            # Find k nearest neighbors
            nearest_indices = np.argpartition(dist_i, k)[:k]
            valid_nearest = nearest_indices[np.isfinite(dist_i[nearest_indices])]
            
            for j in valid_nearest:
                sim_rows.append(i)
                sim_cols.append(j)
                sim = 1.0 - distances[i, j]
                sim_data.append(sim)

        # Create similar edge tensors
        if not sim_rows:
            sim_edge_index = torch.zeros((2, 0), dtype=torch.long)
            sim_edge_attr = None
        else:
            sim_edge_index = torch.tensor(np.vstack((sim_rows, sim_cols)), dtype=torch.long)
            sim_edge_attr = torch.tensor(sim_data, dtype=torch.float).unsqueeze(1)

        print(f"    - Created {sim_edge_index.shape[1]} similar edges.")
        return sim_edge_index, sim_edge_attr, None, None

    def _build_safe_test_isolated_edges(self, data: HeteroData, news_embeddings: np.ndarray) -> None:
        """
        Build edges with test isolation but ensuring no test nodes become isolated.
        Strategy:
        1. Train nodes: use mutual KNN (high quality)
        2. Test nodes: use one-way KNN to train nodes (guaranteed connection)
        3. No test-test connections (test isolated)
        """
        print(f"    Building safe test-isolated edges...")
        
        train_labeled_mask = data['news'].train_labeled_mask.cpu().numpy()
        train_unlabeled_mask = data['news'].train_unlabeled_mask.cpu().numpy()
        test_mask = data['news'].test_mask.cpu().numpy()
        
        # Get indices for each set
        train_labeled_indices = np.where(train_labeled_mask)[0]
        train_unlabeled_indices = np.where(train_unlabeled_mask)[0] 
        test_indices = np.where(test_mask)[0]
        train_all_indices = np.concatenate([train_labeled_indices, train_unlabeled_indices])
        
        all_sim_rows, all_sim_cols, all_sim_data = [], [], []
        
        # 1. Train-Train connections: Use mutual KNN (bidirectional, high quality)
        if len(train_all_indices) > 1:
            train_embeddings = news_embeddings[train_all_indices]
            train_sim_idx, train_sim_attr, _, _ = self._build_mutual_knn_edges(train_embeddings, self.k_neighbors)
            
            if train_sim_idx is not None and train_sim_idx.shape[1] > 0:
                # Map local indices back to global indices
                global_sim_idx = train_all_indices[train_sim_idx.cpu().numpy()]
                all_sim_rows.extend(global_sim_idx[0])
                all_sim_cols.extend(global_sim_idx[1])
                all_sim_data.extend([1.0] * len(global_sim_idx[0]))
                print(f"      - Created {train_sim_idx.shape[1]} train-train mutual KNN edges")
        
        # 2. Test-Train connections: Use one-way KNN (guaranteed connections)
        if len(test_indices) > 0 and len(train_all_indices) > 0:
            test_embeddings = news_embeddings[test_indices]
            train_embeddings = news_embeddings[train_all_indices]
            
            # Calculate test-to-train distances
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(test_embeddings, train_embeddings, metric="cosine", n_jobs=-1)
            
            # For each test node, connect to K nearest train nodes
            k_actual = min(self.k_neighbors, len(train_all_indices))
            for i, test_idx in enumerate(test_indices):
                nearest_train_indices = np.argpartition(distances[i], k_actual)[:k_actual]
                for j in nearest_train_indices:
                    train_idx = train_all_indices[j]
                    # Add bidirectional edges (test<->train)
                    all_sim_rows.extend([test_idx, train_idx])
                    all_sim_cols.extend([train_idx, test_idx])
                    all_sim_data.extend([distances[i, j], distances[i, j]])
            
            print(f"      - Created {len(test_indices) * k_actual * 2} test-train bidirectional edges")
        
        # Create final edge tensors
        if all_sim_rows:
            sim_edge_index = torch.tensor(np.vstack((all_sim_rows, all_sim_cols)), dtype=torch.long)
            sim_edge_attr = torch.tensor(all_sim_data, dtype=torch.float).unsqueeze(1)
        else:
            sim_edge_index = torch.zeros((2, 0), dtype=torch.long)
            sim_edge_attr = None
        
        # Assign to graph
        data['news', 'similar_to', 'news'].edge_index = sim_edge_index
        if sim_edge_attr is not None:
            data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
        
        print(f"    - Total similar edges created: {sim_edge_index.shape[1]}")
        return

    def _build_mutual_knn_edges(self, embeddings: np.ndarray, k: int):
        """
        Build mutual KNN (similar) and mutual farthest (dissimilar) edges for the given embeddings.
        Returns: (sim_edge_index, sim_edge_attr, dis_edge_index, dis_edge_attr)
        """
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1:
            return (torch.zeros((2,0), dtype=torch.long), None, torch.zeros((2,0), dtype=torch.long), None)
        k = min(k, num_nodes - 1)
        if k <= 0:
            return (torch.zeros((2,0), dtype=torch.long), None, torch.zeros((2,0), dtype=torch.long), None)
        print(f"    Building MUTUAL KNN graph (k={k}) for 'news'-'news' edges...")
        try:
            distances = pairwise_distances(embeddings, metric="cosine", n_jobs=-1)
        except Exception as e:
            print(f"      Error calculating pairwise distances: {e}. Using single core.")
            distances = pairwise_distances(embeddings, metric="cosine")

        # --- Mutual KNN (similar) edges ---
        rows, cols, data = [], [], []
        all_neighbors = {}
        for i in tqdm(range(num_nodes), desc=f"      Finding {k} nearest neighbors (pass 1)", leave=False, ncols=100):
            dist_i = distances[i].copy()
            dist_i[i] = np.inf
            indices = np.argpartition(dist_i, k)[:k]
            valid_indices = indices[np.isfinite(dist_i[indices])]
            all_neighbors[i] = set(valid_indices)
        for i in tqdm(range(num_nodes), desc=f"      Checking {k} mutual neighbors (pass 2)", leave=False, ncols=100):
            if i not in all_neighbors: continue
            for j in all_neighbors[i]:
                if j not in all_neighbors: continue
                if i in all_neighbors[j]:
                    rows.append(i)
                    cols.append(j)
                    sim = 1.0 - distances[i, j]
                    data.append(max(0.0, sim)) # Ensure non-negative
        if not rows:
            print("      Warning: No mutual KNN edges found.")
            sim_edge_index = torch.zeros((2,0), dtype=torch.long)
            sim_edge_attr = None
        else:
            sim_edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
            sim_edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        # --- Mutual farthest (dissimilar) edges ---
        dis_rows, dis_cols, dis_data = [], [], []
        farthest_neighbors = {}
        for i in tqdm(range(num_nodes), desc=f"      Finding {k} farthest neighbors (pass 1)", leave=False, ncols=100):
            dist_i = distances[i].copy()
            dist_i[i] = -np.inf
            farthest_indices = np.argpartition(-dist_i, k)[:k]
            valid_indices = farthest_indices[np.isfinite(dist_i[farthest_indices])]
            farthest_neighbors[i] = set(valid_indices)
        for i in tqdm(range(num_nodes), desc=f"      Checking {k} mutual farthest neighbors (pass 2)", leave=False, ncols=100):
            if i not in farthest_neighbors: continue
            for j in farthest_neighbors[i]:
                if j not in farthest_neighbors: continue
                if i in farthest_neighbors[j]:
                    dis_rows.append(i)
                    dis_cols.append(j)
                    sim = 1.0 - distances[i, j]
                    dis_data.append(-sim) # Negative for dissimilar edge
        if not dis_rows:
            dis_edge_index = torch.zeros((2,0), dtype=torch.long)
            dis_edge_attr = None
        else:
            dis_edge_index = torch.tensor(np.vstack((dis_rows, dis_cols)), dtype=torch.long)
            dis_edge_attr = torch.tensor(dis_data, dtype=torch.float).unsqueeze(1)

        print(f"    - Created {sim_edge_index.shape[1]} mutual KNN edges, {dis_edge_index.shape[1]} mutual farthest edges.")
        return sim_edge_index, sim_edge_attr, dis_edge_index, dis_edge_attr

    def _add_dissimilar_edges_universal(self, data: HeteroData, news_embeddings: np.ndarray) -> None:
        """
        Universal method to add dissimilar edges for any edge policy.
        This method creates K-farthest neighbor edges (dissimilar edges).
        Respects test isolation if the edge policy requires it.
        """
        print(f"    Building universal dissimilar edges (k={self.k_neighbors})...")
        
        if self.edge_policy == "knn_test_isolated":
            # Build dissimilar edges with test isolation
            self._build_safe_dissimilar_edges(data, news_embeddings)
        else:
            # Use _build_mutual_knn_edges to get both similar and dissimilar edges
            # We only need the dissimilar edges here
            _, _, dis_edge_index, dis_edge_attr = self._build_mutual_knn_edges(news_embeddings, self.k_neighbors)
            
            if dis_edge_index is not None and dis_edge_index.shape[1] > 0:
                # Make dissimilar edges undirected (if not already)
                if dis_edge_index.shape[1] > 0:
                    # Check if edges are already undirected by looking for reverse edges
                    reverse_edges = dis_edge_index[[1, 0], :]  # Swap src and dst
                    # Convert to set of tuples for efficient lookup
                    existing_edges = set(map(tuple, dis_edge_index.T.tolist()))
                    reverse_edge_tuples = set(map(tuple, reverse_edges.T.tolist()))
                    
                    # If not all reverse edges exist, make it undirected
                    if not reverse_edge_tuples.issubset(existing_edges):
                        dis_edge_index = torch.cat([dis_edge_index, reverse_edges], dim=1)
                        if dis_edge_attr is not None:
                            dis_edge_attr = torch.cat([dis_edge_attr, dis_edge_attr], dim=0)
                
                # Assign to graph
                data['news', 'dissimilar_to', 'news'].edge_index = dis_edge_index
                if dis_edge_attr is not None:
                    data['news', 'dissimilar_to', 'news'].edge_attr = dis_edge_attr
                
                print(f"    - Created {dis_edge_index.shape[1]} 'news <-> news' universal dissimilar edges.")
            else:
                print(f"    - Warning: No dissimilar edges created (empty edge set).")

    def _build_safe_dissimilar_edges(self, data: HeteroData, news_embeddings: np.ndarray) -> None:
        """
        Build dissimilar edges while respecting test isolation.
        Strategy:
        1. Train-Train dissimilar: use mutual farthest neighbors
        2. Test-Train dissimilar: use one-way farthest neighbors 
        3. No Test-Test dissimilar edges (respects test isolation)
        """
        print(f"    Building safe dissimilar edges (respecting test isolation)...")
        
        train_labeled_mask = data['news'].train_labeled_mask.cpu().numpy()
        train_unlabeled_mask = data['news'].train_unlabeled_mask.cpu().numpy()
        test_mask = data['news'].test_mask.cpu().numpy()
        
        # Get indices for each set
        train_labeled_indices = np.where(train_labeled_mask)[0]
        train_unlabeled_indices = np.where(train_unlabeled_mask)[0] 
        test_indices = np.where(test_mask)[0]
        train_all_indices = np.concatenate([train_labeled_indices, train_unlabeled_indices])
        
        all_dis_rows, all_dis_cols, all_dis_data = [], [], []
        
        # 1. Train-Train dissimilar connections: Use mutual farthest neighbors
        if len(train_all_indices) > 1:
            train_embeddings = news_embeddings[train_all_indices]
            _, _, train_dis_idx, train_dis_attr = self._build_mutual_knn_edges(train_embeddings, self.k_neighbors)
            
            if train_dis_idx is not None and train_dis_idx.shape[1] > 0:
                # Map local indices back to global indices
                global_dis_idx = train_all_indices[train_dis_idx.cpu().numpy()]
                all_dis_rows.extend(global_dis_idx[0])
                all_dis_cols.extend(global_dis_idx[1])
                all_dis_data.extend([1.0] * len(global_dis_idx[0]))
                print(f"      - Created {train_dis_idx.shape[1]} train-train mutual dissimilar edges")
        
        # 2. Test-Train dissimilar connections: Use one-way farthest neighbors
        if len(test_indices) > 0 and len(train_all_indices) > 0:
            test_embeddings = news_embeddings[test_indices]
            train_embeddings = news_embeddings[train_all_indices]
            
            # Calculate test-to-train distances
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(test_embeddings, train_embeddings, metric="cosine", n_jobs=-1)
            
            # For each test node, connect to K farthest train nodes
            k_actual = min(self.k_neighbors, len(train_all_indices))
            for i, test_idx in enumerate(test_indices):
                farthest_train_indices = np.argpartition(-distances[i], k_actual)[:k_actual]  # K farthest
                for j in farthest_train_indices:
                    train_idx = train_all_indices[j]
                    # Add bidirectional dissimilar edges (test<->train)
                    all_dis_rows.extend([test_idx, train_idx])
                    all_dis_cols.extend([train_idx, test_idx])
                    all_dis_data.extend([distances[i, j], distances[i, j]])
            
            print(f"      - Created {len(test_indices) * k_actual * 2} test-train bidirectional dissimilar edges")
        
        # Create final dissimilar edge tensors
        if all_dis_rows:
            dis_edge_index = torch.tensor(np.vstack((all_dis_rows, all_dis_cols)), dtype=torch.long)
            dis_edge_attr = torch.tensor(all_dis_data, dtype=torch.float).unsqueeze(1)
        else:
            dis_edge_index = torch.zeros((2, 0), dtype=torch.long)
            dis_edge_attr = None
        
        # Assign to graph
        data['news', 'dissimilar_to', 'news'].edge_index = dis_edge_index
        if dis_edge_attr is not None:
            data['news', 'dissimilar_to', 'news'].edge_attr = dis_edge_attr
        
        print(f"    - Total dissimilar edges created: {dis_edge_index.shape[1]}")

    def _build_safe_multiview_dissimilar_edges(self, data: HeteroData, sub_embeddings: np.ndarray, view_id: int) -> None:
        """
        Build multi-view dissimilar edges while respecting test isolation.
        """
        train_labeled_mask = data['news'].train_labeled_mask.cpu().numpy()
        train_unlabeled_mask = data['news'].train_unlabeled_mask.cpu().numpy()
        test_mask = data['news'].test_mask.cpu().numpy()
        
        # Get indices for each set
        train_labeled_indices = np.where(train_labeled_mask)[0]
        train_unlabeled_indices = np.where(train_unlabeled_mask)[0] 
        test_indices = np.where(test_mask)[0]
        train_all_indices = np.concatenate([train_labeled_indices, train_unlabeled_indices])
        
        all_dis_rows, all_dis_cols, all_dis_data = [], [], []
        
        # 1. Train-Train dissimilar connections: Use mutual farthest neighbors
        if len(train_all_indices) > 1:
            train_sub_emb = sub_embeddings[train_all_indices]
            _, _, train_dis_idx, train_dis_attr = self._build_mutual_knn_edges(train_sub_emb, self.k_neighbors)
            
            if train_dis_idx is not None and train_dis_idx.shape[1] > 0:
                # Map local indices back to global indices
                global_dis_idx = train_all_indices[train_dis_idx.cpu().numpy()]
                all_dis_rows.extend(global_dis_idx[0])
                all_dis_cols.extend(global_dis_idx[1])
                all_dis_data.extend([1.0] * len(global_dis_idx[0]))
        
        # 2. Test-Train dissimilar connections: Use one-way farthest neighbors
        if len(test_indices) > 0 and len(train_all_indices) > 0:
            test_sub_emb = sub_embeddings[test_indices]
            train_sub_emb = sub_embeddings[train_all_indices]
            
            # Calculate test-to-train distances
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(test_sub_emb, train_sub_emb, metric="cosine", n_jobs=-1)
            
            # For each test node, connect to K farthest train nodes
            k_actual = min(self.k_neighbors, len(train_all_indices))
            for i, test_idx in enumerate(test_indices):
                farthest_train_indices = np.argpartition(-distances[i], k_actual)[:k_actual]  # K farthest
                for j in farthest_train_indices:
                    train_idx = train_all_indices[j]
                    # Add bidirectional dissimilar edges (test<->train)
                    all_dis_rows.extend([test_idx, train_idx])
                    all_dis_cols.extend([train_idx, test_idx])
                    all_dis_data.extend([distances[i, j], distances[i, j]])
        
        # Create final dissimilar edge tensors
        if all_dis_rows:
            dis_idx = torch.tensor(np.vstack((all_dis_rows, all_dis_cols)), dtype=torch.long)
            dis_attr = torch.tensor(all_dis_data, dtype=torch.float).unsqueeze(1)
        else:
            dis_idx = torch.zeros((2, 0), dtype=torch.long)
            dis_attr = None
        
        # Assign to graph
        dis_type = ("news", f"dissimilar_to_sub{view_id}", "news")
        data[dis_type].edge_index = dis_idx
        if dis_attr is not None:
            data[dis_type].edge_attr = dis_attr
        
        print(f"    - Created {dis_idx.shape[1]} 'news <-> news' safe dissimilar_sub{view_id} edges.")

    def _add_interaction_edges_by_type(self, data, train_labeled_indices, train_unlabeled_indices, test_indices, num_nodes, num_interactions_per_news, num_interaction_nodes):
        all_interaction_embeddings = []
        all_tones_set = set()
        edge_indices = dict()
        reverse_edge_indices = dict()
        
        # Pre-fetch train data embeddings and tones for all train indices at once
        train_indices_in_batch = np.concatenate([train_labeled_indices, train_unlabeled_indices])
        if len(train_indices_in_batch) > 0:
            train_embeddings_batch = self.train_data.select(train_indices_in_batch.tolist())
            train_interactions_batch = train_embeddings_batch[self.interaction_embedding_field]
            train_tones_batch = train_embeddings_batch[self.interaction_tone_field]
        
        # Pre-fetch test data embeddings and tones for current test batch at once
        if len(test_indices) > 0:
            test_embeddings_batch = self.test_data.select(test_indices.tolist())
            test_interactions_batch = test_embeddings_batch[self.interaction_embedding_field]
            test_tones_batch = test_embeddings_batch[self.interaction_tone_field]
        
        pbar_interact = tqdm(total=num_nodes, desc="    Extracting Interaction Embeddings", ncols=100)
        interaction_global_idx = 0
        
        # Process train nodes first
        for idx in train_indices_in_batch:
            train_batch_idx = np.where(train_indices_in_batch == idx)[0][0]
            embeddings_list = train_interactions_batch[train_batch_idx]
            tones_list = train_tones_batch[train_batch_idx]
            
            node_interactions = np.array(embeddings_list)
            all_interaction_embeddings.append(node_interactions)
            for i, tone in enumerate(tones_list):
                tone_key = tone.strip().lower().replace(' ', '_')
                all_tones_set.add(tone_key)
                if tone_key not in edge_indices:
                    edge_indices[tone_key] = [[], []]
                    reverse_edge_indices[tone_key] = [[], []]
                edge_indices[tone_key][0].append(train_batch_idx)
                edge_indices[tone_key][1].append(interaction_global_idx)
                reverse_edge_indices[tone_key][0].append(interaction_global_idx)
                reverse_edge_indices[tone_key][1].append(train_batch_idx)
                interaction_global_idx += 1
            pbar_interact.update(1)
            
        # Process test nodes
        for idx in test_indices:
            test_batch_idx = np.where(test_indices == idx)[0][0]
            embeddings_list = test_interactions_batch[test_batch_idx]
            tones_list = test_tones_batch[test_batch_idx]
            
            node_interactions = np.array(embeddings_list)
            all_interaction_embeddings.append(node_interactions)
            for i, tone in enumerate(tones_list):
                tone_key = tone.strip().lower().replace(' ', '_')
                all_tones_set.add(tone_key)
                if tone_key not in edge_indices:
                    edge_indices[tone_key] = [[], []]
                    reverse_edge_indices[tone_key] = [[], []]
                edge_indices[tone_key][0].append(len(train_indices_in_batch) + test_batch_idx)
                edge_indices[tone_key][1].append(interaction_global_idx)
                reverse_edge_indices[tone_key][0].append(interaction_global_idx)
                reverse_edge_indices[tone_key][1].append(len(train_indices_in_batch) + test_batch_idx)
                interaction_global_idx += 1
            pbar_interact.update(1)
            
        pbar_interact.close()
        final_interaction_features = np.vstack(all_interaction_embeddings)
        data['interaction'].x = torch.tensor(final_interaction_features, dtype=torch.float)
        data['interaction'].num_nodes = data['interaction'].x.shape[0]
        if data['interaction'].num_nodes != num_interaction_nodes:
            print(f"Warning: Interaction node count mismatch! Expected {num_interaction_nodes}, Got {data['interaction'].num_nodes}")
            
        # Create edge indices for each tone type
        for tone_key in sorted(all_tones_set):
            edge_type = ('news', f'has_{tone_key}_interaction', 'interaction')
            rev_edge_type = ('interaction', f'rev_has_{tone_key}_interaction', 'news')
            if edge_indices[tone_key][0]:
                data[edge_type].edge_index = torch.tensor(edge_indices[tone_key], dtype=torch.long)
            if reverse_edge_indices[tone_key][0]:
                data[rev_edge_type].edge_index = torch.tensor(reverse_edge_indices[tone_key], dtype=torch.long)

    def _add_interaction_edges_with_attr(self, data, train_labeled_indices, train_unlabeled_indices, test_indices, num_nodes, num_interactions_per_news, num_interaction_nodes):
        all_interaction_embeddings = []
        all_interaction_tones = []
        
        # Pre-fetch train data embeddings and tones for all train indices at once
        train_indices_in_batch = np.concatenate([train_labeled_indices, train_unlabeled_indices])
        # print(f"train_labeled_indices: {train_labeled_indices}")
        # print(f"train_unlabeled_indices: {train_unlabeled_indices}")
        # print(f"train_indices_in_batch: {train_indices_in_batch}")
        if len(train_indices_in_batch) > 0:
            train_embeddings_batch = self.train_data.select(train_indices_in_batch.tolist())
            train_interactions_batch = train_embeddings_batch[self.interaction_embedding_field]
            train_tones_batch = train_embeddings_batch[self.interaction_tone_field]
        
        # Pre-fetch test data embeddings and tones for current test batch at once
        if len(test_indices) > 0:
            test_embeddings_batch = self.test_data.select(test_indices.tolist())
            test_interactions_batch = test_embeddings_batch[self.interaction_embedding_field]
            test_tones_batch = test_embeddings_batch[self.interaction_tone_field]
        
        pbar_interact = tqdm(total=num_nodes, desc="    Extracting Interaction Embeddings", ncols=100)
        interaction_global_idx = 0
        
        # Process train nodes first
        for idx in train_indices_in_batch:
            train_batch_idx = np.where(train_indices_in_batch == idx)[0][0]
            embeddings_list = train_interactions_batch[train_batch_idx]
            tones_list = train_tones_batch[train_batch_idx]
            
            node_interactions = np.array(embeddings_list)
            all_interaction_embeddings.append(node_interactions)
            for i, tone in enumerate(tones_list):
                tone_key = tone.strip().lower().replace(' ', '_')
                all_interaction_tones.append(self._tone2id(tone_key))
                interaction_global_idx += 1
            pbar_interact.update(1)
            
        # Process test nodes
        for idx in test_indices:
            test_batch_idx = np.where(test_indices == idx)[0][0]
            embeddings_list = test_interactions_batch[test_batch_idx]
            tones_list = test_tones_batch[test_batch_idx]
            
            node_interactions = np.array(embeddings_list)
            all_interaction_embeddings.append(node_interactions)
            for i, tone in enumerate(tones_list):
                tone_key = tone.strip().lower().replace(' ', '_')
                all_interaction_tones.append(self._tone2id(tone_key))
                interaction_global_idx += 1
            pbar_interact.update(1)
            
        pbar_interact.close()
        final_interaction_features = np.vstack(all_interaction_embeddings)
        data['interaction'].x = torch.tensor(final_interaction_features, dtype=torch.float)
        data['interaction'].num_nodes = data['interaction'].x.shape[0]
        if data['interaction'].num_nodes != num_interaction_nodes:
            print(f"Warning: Interaction node count mismatch! Expected {num_interaction_nodes}, Got {data['interaction'].num_nodes}")
            
        # Create edge indices using simple arange
        news_has_interaction_src = torch.arange(num_nodes).repeat_interleave(num_interactions_per_news)
        interaction_has_interaction_tgt = torch.arange(num_interaction_nodes)
        data['news', 'has_interaction', 'interaction'].edge_index = torch.stack([news_has_interaction_src, interaction_has_interaction_tgt], dim=0)
        data['interaction', 'rev_has_interaction', 'news'].edge_index = torch.stack([interaction_has_interaction_tgt, news_has_interaction_src], dim=0)
        data['news', 'has_interaction', 'interaction'].edge_attr = torch.tensor(all_interaction_tones, dtype=torch.long)
        data['interaction', 'rev_has_interaction', 'news'].edge_attr = torch.tensor(all_interaction_tones, dtype=torch.long)

    def analyze_hetero_graph(self, hetero_graph: HeteroData) -> None:
        """Detailed analysis for heterogeneous graph, similar to build_graph.py but adapted for hetero."""
        
        print("\n" + "=" * 60)
        print("     Heterogeneous Graph Analysis (Detailed)")
        print("=" * 60)

        self.graph_metrics = {}  # Reset metrics

        # --- Node Type Stats ---
        print("\n--- Node Types ---")
        total_nodes = 0
        node_type_info = {}
        for node_type in hetero_graph.node_types:
            n = hetero_graph[node_type].num_nodes
            total_nodes += n
            print(f"Node Type: '{node_type}'")
            print(f"  - Num Nodes: {n}")
            if hasattr(hetero_graph[node_type], 'x') and hetero_graph[node_type].x is not None:
                print(f"  - Features Dim: {hetero_graph[node_type].x.shape[1]}")
                node_type_info[node_type] = {"num_nodes": n, "feature_dim": hetero_graph[node_type].x.shape[1]}
            
            # For 'news', print label and mask info
            if node_type == 'news':
                if hasattr(hetero_graph[node_type], 'y') and hetero_graph[node_type].y is not None:
                    y = hetero_graph[node_type].y.cpu().numpy()
                    print(f"  - Labels Shape: {y.shape}")
                    unique, counts = np.unique(y, return_counts=True)
                    label_dist = {int(k): int(v) for k, v in zip(unique, counts)}
                    print(f"  - Label Distribution: {label_dist}")
                    node_type_info[node_type]["label_dist"] = label_dist
                for mask in ['train_labeled_mask', 'train_unlabeled_mask', 'test_mask']:
                    if hasattr(hetero_graph[node_type], mask) and hetero_graph[node_type][mask] is not None:
                        count = hetero_graph[node_type][mask].sum().item()
                        print(f"  - {mask}: {count} nodes ({count/n*100:.1f}% of '{node_type}')")
                        node_type_info[node_type][mask] = count
        
        print(f"Total Nodes (all types): {total_nodes}")
        self.graph_metrics['node_type_info'] = node_type_info
        self.graph_metrics['nodes_total'] = total_nodes

        # --- Edge Type Stats ---
        print("\n--- Edge Types ---")
        total_edges = 0
        edge_type_info = {}
        for edge_type in hetero_graph.edge_types:
            num_edges = hetero_graph[edge_type].num_edges
            total_edges += num_edges
            edge_type_str = " -> ".join(edge_type) if isinstance(edge_type, tuple) else edge_type
            print(f"[*] Edge Type: {edge_type_str}")
            print(f"  - Num Edges: {num_edges}")
            if hasattr(hetero_graph[edge_type], 'edge_attr') and hetero_graph[edge_type].edge_attr is not None:
                edge_attr = hetero_graph[edge_type].edge_attr
                try:
                    shape = tuple(edge_attr.shape)
                    if len(shape) == 1: # news - interaction (tone_id)
                        print(f"  - Attributes Dim: {shape[0]}") # (num_nodes, )    # each has_interaction edge has a tone_id
                        edge_type_info[edge_type_str] = {"num_edges": num_edges, "attr_dim": shape[0]}
                    else:
                        print(f"  - Attributes Dim: {shape}")   # (num_nodes, [similarity]) # each news-news edge has a similarity score (in 1 dimension vector)
                        edge_type_info[edge_type_str] = {"num_edges": num_edges, "attr_dim": shape[1]}
                except Exception as e:
                    print(f"  - Attributes Dim: Error getting shape - {e}")
                    print(f"  - Attributes: {edge_attr}")
            else:
                print("  - Attributes: None")
                edge_type_info[edge_type_str] = {"num_edges": num_edges, "attr_dim": None}
        print(f"Total Edges (all types): {total_edges}")
        self.graph_metrics['edge_type_info'] = edge_type_info
        self.graph_metrics['edges_total'] = total_edges

        # --- News-News Edge Analysis ---
        news_similar_edge_type = ('news', 'similar_to', 'news')
        news_dissimilar_edge_type = ('news', 'dissimilar_to', 'news')
        news_label_aware_similar_edge_type = ('news', 'label_aware_similar_to', 'news')
        news_low_level_knn_edge_type = ('news', 'low_level_knn_to', 'news')
        num_news_nodes = hetero_graph['news'].num_nodes

        if news_similar_edge_type in hetero_graph.edge_types:
            print("\n--- Analysis for 'news'-'similar_to'-'news' Edges ---")
            nn_edge_index = hetero_graph[news_similar_edge_type].edge_index
            num_nn_edges = hetero_graph[news_similar_edge_type].num_edges
            print(f"  - Num Edges: {num_nn_edges}")
            degrees_nn = torch.zeros(num_news_nodes, dtype=torch.long, device=nn_edge_index.device)
            degrees_nn.scatter_add_(0, nn_edge_index[0], torch.ones_like(nn_edge_index[0]))
            degrees_nn.scatter_add_(0, nn_edge_index[1], torch.ones_like(nn_edge_index[1]))
            avg_degree_nn = degrees_nn.float().mean().item() / 2.0
            print(f"  - Avg Degree (undirected): {avg_degree_nn:.2f}")
            self.graph_metrics['avg_degree_news_similar_to'] = avg_degree_nn
            num_isolated = int((degrees_nn == 0).sum().item())
            print(f"  - Isolated News Nodes: {num_isolated} ({num_isolated/num_news_nodes*100:.1f}%)")
            self.graph_metrics['news_similar_isolated'] = num_isolated

        if news_dissimilar_edge_type in hetero_graph.edge_types:
            print("\n--- Analysis for 'news'-'dissimilar_to'-'news' Edges ---")
            nn_edge_index = hetero_graph[news_dissimilar_edge_type].edge_index
            num_nn_edges = hetero_graph[news_dissimilar_edge_type].num_edges
            print(f"  - Num Edges: {num_nn_edges}")
            degrees_nn = torch.zeros(num_news_nodes, dtype=torch.long, device=nn_edge_index.device)
            degrees_nn.scatter_add_(0, nn_edge_index[0], torch.ones_like(nn_edge_index[0]))
            degrees_nn.scatter_add_(0, nn_edge_index[1], torch.ones_like(nn_edge_index[1]))
            avg_degree_nn = degrees_nn.float().mean().item() / 2.0
            print(f"  - Avg Degree (undirected): {avg_degree_nn:.2f}")
            self.graph_metrics['avg_degree_news_dissimilar_to'] = avg_degree_nn
            num_isolated = int((degrees_nn == 0).sum().item())
            print(f"  - Isolated News Nodes: {num_isolated} ({num_isolated/num_news_nodes*100:.1f}%)")
            self.graph_metrics['news_dissimilar_isolated'] = num_isolated

        if news_label_aware_similar_edge_type in hetero_graph.edge_types:
            print("\n--- Analysis for 'news'-'label_aware_similar_to'-'news' Edges ---")
            nn_edge_index = hetero_graph[news_label_aware_similar_edge_type].edge_index
            num_nn_edges = hetero_graph[news_label_aware_similar_edge_type].num_edges
            print(f"  - Num Edges: {num_nn_edges}")
            degrees_nn = torch.zeros(num_news_nodes, dtype=torch.long, device=nn_edge_index.device)
            degrees_nn.scatter_add_(0, nn_edge_index[0], torch.ones_like(nn_edge_index[0]))
            degrees_nn.scatter_add_(0, nn_edge_index[1], torch.ones_like(nn_edge_index[1]))
            avg_degree_nn = degrees_nn.float().mean().item() / 2.0
            print(f"  - Avg Degree (undirected): {avg_degree_nn:.2f}")
            self.graph_metrics['avg_degree_news_label_aware_similar_to'] = avg_degree_nn
            num_isolated = int((degrees_nn == 0).sum().item())
            print(f"  - Isolated News Nodes: {num_isolated} ({num_isolated/num_news_nodes*100:.1f}%)")
            self.graph_metrics['news_label_aware_similar_isolated'] = num_isolated

        # --- Analysis for ALL news-news Edges (merged) ---
        if any(edge_type in hetero_graph.edge_types for edge_type in [news_similar_edge_type, news_dissimilar_edge_type, news_label_aware_similar_edge_type, news_low_level_knn_edge_type]):
            print("\n--- Analysis for ALL news-news Edges (merged) ---")
            G_all = nx.Graph()
            G_all.add_nodes_from(range(num_news_nodes))
            # similar edges
            if news_similar_edge_type in hetero_graph.edge_types:
                sim_idx = hetero_graph[news_similar_edge_type].edge_index.cpu().numpy()
                G_all.add_edges_from(sim_idx.T)
            # dissimilar edges
            if news_dissimilar_edge_type in hetero_graph.edge_types:
                dis_idx = hetero_graph[news_dissimilar_edge_type].edge_index.cpu().numpy()
                G_all.add_edges_from(dis_idx.T)
            # label-aware similar edges
            if news_label_aware_similar_edge_type in hetero_graph.edge_types:
                label_aware_sim_idx = hetero_graph[news_label_aware_similar_edge_type].edge_index.cpu().numpy()
                G_all.add_edges_from(label_aware_sim_idx.T)
            # low-level KNN edges
            if news_low_level_knn_edge_type in hetero_graph.edge_types:
                low_knn_idx = hetero_graph[news_low_level_knn_edge_type].edge_index.cpu().numpy()
                G_all.add_edges_from(low_knn_idx.T)
            print(f"  Nodes: {G_all.number_of_nodes()} Edges: {G_all.number_of_edges()}")
            if G_all.number_of_nodes() > 0:
                print(f"  Density: {nx.density(G_all):.4f}")
                print(f"  Avg Clustering: {nx.average_clustering(G_all):.4f}")
                print(f"  Connected Components: {nx.number_connected_components(G_all)}")

        print("=" * 60)
        print("      End of Heterogeneous Graph Analysis")
        print("=" * 60 + "\n")

    def save_graph(self, hetero_graph: HeteroData, batch_id=None) -> Optional[str]:
        """Save the HeteroData graph and analysis results."""

        # --- Generate graph name ---
        # Add sampling info to filename if sampling was used
        suffix = []
        if self.no_interactions:
            suffix.append("no_interactions")
        if self.ensure_test_labeled_neighbor:
            suffix.append("ensure_test_labeled_neighbor")
        if self.pseudo_label:
            suffix.append("pseudo")
        if self.partial_unlabeled:
            suffix.append("partial")
            suffix.append(f"sample_unlabeled_factor_{self.sample_unlabeled_factor}")
        if self.enable_dissimilar:
            suffix.append("dissimilar")
        if self.multi_view > 1:
            suffix.append(f"multiview_{self.multi_view}")
        sampling_suffix = f"{'_'.join(suffix)}" if suffix else ""

        # Include text embedding type and edge types in name
        graph_name = f"{self.k_shot}_shot_{self.embedding_type}_hetero_{self.edge_policy}_{self.k_neighbors}_{sampling_suffix}"
        scenario_dir = os.path.join(self.output_dir, graph_name)
        os.makedirs(scenario_dir, exist_ok=True)
        if batch_id is not None:
            graph_file = f"graph_batch{batch_id}.pt"
            metrics_file = f"graph_batch{batch_id}_metrics.json"
            indices_file = f"graph_batch{batch_id}_indices.json"
        else:
            graph_file = f"graph.pt"
            metrics_file = f"graph_metrics.json"
            indices_file = f"graph_indices.json"

        graph_path = os.path.join(scenario_dir, graph_file)
        metrics_path = os.path.join(scenario_dir, metrics_file)
        indices_path = os.path.join(scenario_dir, indices_file)
        # --- End filename generation ---

        # Save graph data
        cpu_graph_data = hetero_graph.cpu()
        torch.save(cpu_graph_data, graph_path)

        # Save graph metrics (simplified for hetero)
        def default_serializer(obj): # Helper for JSON serialization
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, torch.Tensor): return obj.tolist()
            try: return json.JSONEncoder().encode(obj)
            except TypeError: return str(obj)

        try:
            with open(metrics_path, "w") as f:
                json.dump(self.graph_metrics, f, indent=2, default=default_serializer)
            print(f"  - Graph analysis metrics saved to {metrics_path}")
        except Exception as e:
            print(f"  Error saving metrics JSON: {e}")

        # Save selected indices info
        indices_data = {
            "k_shot": int(self.k_shot),
            "seed": int(self.seed),
            "partial_unlabeled": self.partial_unlabeled,
            "embedding_type": self.embedding_type,
            "news_news_edge_policy": self.edge_policy,
        }

        if self.train_labeled_indices is not None:
            indices_data["train_labeled_indices"] = [int(i) for i in self.train_labeled_indices]
            # Add label distribution if possible
            try:
                train_labels = self.train_data['label']
                label_dist = {}
                for idx in self.train_labeled_indices:
                    label = train_labels[int(idx)]
                    label_dist[label] = label_dist.get(label, 0) + 1
                indices_data["train_labeled_label_distribution"] = {int(k): int(v) for k, v in label_dist.items()}
            except Exception as e_label: print(f"Warning: Could not get label distribution for indices: {e_label}")

        if self.partial_unlabeled and self.train_unlabeled_indices is not None:
            indices_data["sample_unlabeled_factor"] = int(self.sample_unlabeled_factor)
            indices_data["train_unlabeled_indices"] = [int(i) for i in self.train_unlabeled_indices]
            # Add label distribution if possible
            try:
                train_labels = self.train_data['label']
                true_label_dist = {}
                for idx in self.train_unlabeled_indices:
                    label = train_labels[int(idx)]
                    true_label_dist[label] = true_label_dist.get(label, 0) + 1
                indices_data["train_unlabeled_true_label_distribution"] = {int(k): int(v) for k, v in true_label_dist.items()}
                
                # Add pseudo label distribution if using pseudo label sampling
                if self.pseudo_label:
                    try:
                        with open(self.pseudo_label_cache_path, "r") as f:
                            pseudo_data = json.load(f)
                        pseudo_label_map = {int(item["index"]): int(item["pseudo_label"]) for item in pseudo_data}
                        # Overall pseudo label cache distribution
                        all_pseudo_labels = list(pseudo_label_map.values())
                        indices_data["pseudo_label_cache_distribution"] = dict(Counter(all_pseudo_labels))
                        # Distribution of sampled pseudo labels
                        sampled_pseudo_labels = [pseudo_label_map[idx] for idx in self.train_unlabeled_indices if idx in pseudo_label_map]
                        indices_data["train_unlabeled_pseudo_label_distribution"] = dict(Counter(sampled_pseudo_labels))
                    except Exception as e:
                        print(f"Warning: Could not compute pseudo-label stats for indices.json: {e}")
            except Exception as e_label: print(f"Warning: Could not get label distribution for indices: {e_label}")

        with open(indices_path, "w") as f:
            json.dump(indices_data, f, indent=2)
        print(f"  - Selected indices info saved to {indices_path}")
        print(f"  - Graph saved to {graph_path}")

        return graph_path

    def run_pipeline(self) -> Optional[HeteroData]:
        """Run the complete graph building pipeline."""
        self.load_dataset()
        hetero_graph = self.build_hetero_graph()
        # print("  - hetero_graph.x.shape:", hetero_graph['news'].x.shape)
        # print("  - hetero_graph.train_labeled_mask.shape:", hetero_graph['news'].train_labeled_mask.shape)
        # print("  - hetero_graph.train_unlabeled_mask.shape:", hetero_graph['news'].train_unlabeled_mask.shape)
        # print("  - hetero_graph.test_mask.shape:", hetero_graph['news'].test_mask.shape)
        self.analyze_hetero_graph(hetero_graph)
        self.save_graph(hetero_graph)
        return hetero_graph


# --- Argument Parser ---
def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Build a HETEROGENEOUS graph ('news', 'interaction') for few-shot fake news detection")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME, choices=["politifact", "gossipcop"], help=f"HuggingFace Dataset (default: {DEFAULT_DATASET_NAME})")
    parser.add_argument("--k_shot", type=int, default=DEFAULT_K_SHOT, choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], help=f"Number of labeled samples per class (3-16) (default: {DEFAULT_K_SHOT})")

    # Node Feature Args
    parser.add_argument("--embedding_type", type=str, default=DEFAULT_EMBEDDING_TYPE, choices=["bert", "roberta", "distilbert", "bigbird", "deberta"], help=f"Embedding type for 'news' nodes (default: {DEFAULT_EMBEDDING_TYPE})")

    # Edge Policy Args (for 'news'-'similar_to'-'news' edges)
    parser.add_argument("--edge_policy", type=str, default=DEFAULT_EDGE_POLICY, choices=["knn", "label_aware_knn", "knn_test_isolated"], help="Edge policy for 'news'-'news' similarity edges")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help=f"K for (Mutual) KNN policy (default: {DEFAULT_K_NEIGHBORS})")
    parser.add_argument("--ensure_test_labeled_neighbor", action="store_true", help="Ensure every test node is connected to at least one train_labeled node (anchor edge).")

    # Sampling Args
    parser.add_argument("--partial_unlabeled", action="store_true", help="Use only a partial subset of unlabeled nodes. Suffix: partial")
    parser.add_argument("--sample_unlabeled_factor", type=int, default=DEFAULT_SAMPLE_UNLABELED_FACTOR, help="Factor M to sample M*2*k unlabeled training 'news' nodes (default: 10). Used if --partial_unlabeled.")
    parser.add_argument("--pseudo_label", action="store_true", help="Enable pseudo label factor. Suffix: pseudo")
    parser.add_argument("--pseudo_label_cache_path", type=str, default=None, help="Path to pseudo-label cache (json). Default: utils/pseudo_label_cache_<dataset>.json")
    parser.add_argument("--enable_dissimilar", action="store_true", help="Enable dissimilar edge construction. Suffix: dissimilar")
    parser.add_argument("--multi_view", type=int, default=DEFAULT_MULTI_VIEW, help=f"Number of sub-embeddings (views) to split news embeddings into (default: {DEFAULT_MULTI_VIEW})")

    # Interaction Edge Args
    parser.add_argument("--no_interactions", action="store_true", help="Build a graph without any 'interaction' nodes or edges.")
    parser.add_argument("--interaction_embedding_field", type=str, default="interaction_embeddings_list", help="Field for interaction embeddings")
    parser.add_argument("--interaction_tone_field", type=str, default="interaction_tones_list", help="Field for interaction tones")
    parser.add_argument("--interaction_edge_mode", type=str, default=DEFAULT_INTERACTION_EDGE_MODE, choices=["edge_attr", "edge_type"], help="How to encode interaction tone: as edge type (edge_type) or as edge_attr (edge_attr)")
    
    # Output & Settings Args
    parser.add_argument("--output_dir", type=str, default=DEFAULT_GRAPH_DIR, help=f"Directory to save graphs (default: {DEFAULT_GRAPH_DIR})")
    parser.add_argument("--dataset_cache_dir", type=str, default=DEFAULT_DATASET_CACHE_DIR, help=f"Directory to cache datasets (default: {DEFAULT_DATASET_CACHE_DIR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for graph building (default: {DEFAULT_BATCH_SIZE})")

    return parser.parse_args()


# --- Main Execution ---
def main() -> None:
    """Main function to run the heterogeneous graph building pipeline."""
    args = parse_arguments()
    set_seed(args.seed)
    
    if args.edge_policy == "label_aware_knn":
        args.pseudo_label = True
        
    if args.pseudo_label:
        args.partial_unlabeled = True
        
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print("   Heterogeneous Fake News Graph Building Pipeline")
    print("=" * 60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"K-Shot:           {args.k_shot}")
    print(f"News Embeddings:  {args.embedding_type}")
    print("-" * 20 + " News-News Edges " + "-" * 20)
    print(f"Policy:           {args.edge_policy}")
    print(f"K neighbors:      {args.k_neighbors}")
    print(f"Multi-view:       {args.multi_view}")
    print(f"Ensure Test Labeled Neighbor: {args.ensure_test_labeled_neighbor}")
    print("-" * 20 + " News Node Sampling " + "-" * 20)
    print(f"Partial Unlabeled: {args.partial_unlabeled}")
    if args.partial_unlabeled: 
        print(f"Sample Factor(M): {args.sample_unlabeled_factor} (target 2*k-shot*M nodes)")
        print(f"Pseudo-label Sampling: {args.pseudo_label}")
        if args.pseudo_label:
            print(f"Pseudo-label Cache: {args.pseudo_label_cache_path or f'utils/pseudo_label_cache_{args.dataset_name}.json'}")
    else: 
        print(f"Sample Factor(M): N/A (using all unlabeled train news nodes)")
    print("-" * 20 + " Output & Settings " + "-" * 20)
    print(f"Output directory: {args.output_dir}")
    print(f"Seed:             {args.seed}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available(): print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")

    # Instantiate and run the builder
    builder = HeteroGraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        partial_unlabeled=args.partial_unlabeled if hasattr(args, 'partial_unlabeled') else False,
        sample_unlabeled_factor=args.sample_unlabeled_factor,
        pseudo_label=args.pseudo_label,
        pseudo_label_cache_path=args.pseudo_label_cache_path,
        multi_view=args.multi_view,
        enable_dissimilar=args.enable_dissimilar if hasattr(args, 'enable_dissimilar') else False,
        ensure_test_labeled_neighbor=args.ensure_test_labeled_neighbor if hasattr(args, 'ensure_test_labeled_neighbor') else False,
        interaction_embedding_field=args.interaction_embedding_field,
        interaction_tone_field=args.interaction_tone_field,
        interaction_edge_mode=args.interaction_edge_mode,
        dataset_cache_dir=args.dataset_cache_dir,
        seed=args.seed,
        output_dir=args.output_dir,
        no_interactions=args.no_interactions,
    )

    hetero_graph = builder.run_pipeline()

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print(" Heterogeneous Graph Building Complete")
    print("=" * 60)
    print(f"  Total Nodes:                             {hetero_graph['news'].num_nodes + hetero_graph['interaction'].num_nodes if not args.no_interactions else hetero_graph['news'].num_nodes}")
    print(f"    - News Nodes:                          {hetero_graph['news'].num_nodes}")
    print(f"    - Interact Nodes:                      {hetero_graph['interaction'].num_nodes if not args.no_interactions else 0}")
    print(f"  Total News-News Edges:                   {hetero_graph['news', 'similar_to', 'news'].num_edges + hetero_graph['news', 'dissimilar_to', 'news'].num_edges}")
    print(f"    - News<-similar->News:                 {hetero_graph['news', 'similar_to', 'news'].num_edges}")
    print(f"    - News<-dissimilar->News:              {hetero_graph['news', 'dissimilar_to', 'news'].num_edges}")
    print(f"  Total News-Interact Edges:               {hetero_graph['news', 'has_interaction', 'interaction'].num_edges + hetero_graph['interaction', 'rev_has_interaction', 'news'].num_edges if not args.no_interactions else 0}")
    print(f"    - News<-has_interaction->Interact:     {hetero_graph['news', 'has_interaction', 'interaction'].num_edges if not args.no_interactions else 0}")
    print(f"    - Interact<-rev_has_interaction->News: {hetero_graph['interaction', 'rev_has_interaction', 'news'].num_edges if not args.no_interactions else 0}")
    print("\nNext Steps:")
    print(f"  1. Review the saved graph '.pt' file, metrics '.json' file, and indices '.json' file.")
    print(f"  2. Train a GNN model, e.g.:")
    print(f"  python train_hetero_graph.py --graph_path {os.path.join(builder.output_dir, 'scenario', '<graph_file_name>.pt')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()