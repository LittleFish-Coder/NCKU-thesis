# filepath: /home/littlefish/fake-news-detection/train_hetero_graph.py
import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import Adam
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, HANConv, Linear, SAGEConv, GATv2Conv, to_hetero, RGCNConv

# Constants
DEFAULT_MODEL = "HAN"   # HGT, HAN, HANv2, SAGE, GATv2
DEFAULT_LOSS_FN = "ce" # ce, focal
DEFAULT_EPOCHS = 300
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_DROPOUT = 0.3
DEFAULT_HEADS = 4 # Number of attention heads for HGT/HAN
DEFAULT_HGT_LAYERS = 1
DEFAULT_HAN_LAYERS = 1  # Number of layers for HAN
DEFAULT_PATIENCE = 30
DEFAULT_SEED = 42
DEFAULT_TARGET_NODE_TYPE = "news" # Target node type for classification
RESULTS_DIR = "results_hetero" # Separate results for hetero models
PLOTS_DIR = "plots_hetero"

# Constants for early stopping
EPSILON = 1e-3  # Minimum difference to consider as improvement
OVERFIT_THRESHOLD = 0.3  # Stop training if validation loss drops below this

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class KLDistillationLoss(nn.Module):
    """Knowledge Distillation Loss using KL Divergence"""
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, targets):
        # KL divergence between student and teacher
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(student_logits, targets)
        
        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss


class ContrastiveLoss(nn.Module):
    """Improved Contrastive Loss for few-shot graph representation learning"""
    def __init__(self, temperature: float = 0.07, margin: float = 0.5, max_samples: int = 64):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.max_samples = max_samples  # Limit samples for numerical stability
    
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        
        # Return zero loss for trivial cases
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Sample subset if batch is too large (for numerical stability)
        if batch_size > self.max_samples:
            indices = torch.randperm(batch_size, device=embeddings.device)[:self.max_samples]
            embeddings = embeddings[indices]
            labels = labels[indices]
            batch_size = self.max_samples
        
        # Normalize embeddings to unit sphere
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Create masks for positive and negative pairs
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        positives_mask = labels_eq.float()
        negatives_mask = (~labels_eq).float()
        
        # Remove self-similarity (diagonal)
        eye_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        positives_mask[eye_mask] = 0
        negatives_mask[eye_mask] = 0
        
        # Check if we have any positive pairs
        if positives_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # For numerical stability, subtract max before exp
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum of positive similarities for each anchor
        pos_exp_sim = (exp_sim * positives_mask).sum(dim=1)
        
        # Sum of all non-self similarities for each anchor  
        all_exp_sim = (exp_sim * (positives_mask + negatives_mask)).sum(dim=1)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        pos_exp_sim = torch.clamp(pos_exp_sim, min=epsilon)
        all_exp_sim = torch.clamp(all_exp_sim, min=epsilon)
        
        # InfoNCE loss: -log(sum(pos_sim) / sum(all_sim))
        loss = -torch.log(pos_exp_sim / all_exp_sim)
        
        # Only use anchors that have positive pairs
        has_positives = (positives_mask.sum(dim=1) > 0).float()
        loss = loss * has_positives
        
        # Return mean loss over valid anchors
        num_valid = has_positives.sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class RobustFewShotLoss(nn.Module):
    """Robust loss function specifically designed for few-shot learning scenarios"""
    def __init__(self, temperature: float = 0.1, label_smoothing: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logits, targets, embeddings=None):
        # Primary cross-entropy loss with label smoothing
        ce_loss = self.ce_loss(logits, targets)
        
        # Confidence penalty to prevent overconfident predictions in few-shot
        softmax_probs = F.softmax(logits, dim=1)
        max_probs = torch.max(softmax_probs, dim=1)[0]
        confidence_penalty = torch.mean(max_probs)  # Penalize high confidence
        
        # Entropy regularization to encourage diverse predictions
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8), dim=1)
        entropy_reg = -torch.mean(entropy)  # Negative because we want to maximize entropy
        
        # Combine losses with careful weighting for few-shot
        total_loss = ce_loss + 0.1 * confidence_penalty + 0.05 * entropy_reg
        
        return total_loss


class EnhancedLoss(nn.Module):
    """Enhanced loss combining multiple loss functions for few-shot learning"""
    def __init__(self, loss_type: str = "enhanced", focal_gamma: float = 1.5, label_smoothing: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "ce":
            self.primary_loss = nn.CrossEntropyLoss()
        elif loss_type == "ce_smooth":
            self.primary_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss_type == "focal":
            self.primary_loss = FocalLoss(gamma=focal_gamma)
        elif loss_type == "robust":
            self.primary_loss = RobustFewShotLoss(label_smoothing=label_smoothing)
        elif loss_type == "enhanced":
            # Better hyperparameters for few-shot learning
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.focal_loss = FocalLoss(alpha=0.75, gamma=focal_gamma)  # Less aggressive focal loss
            self.contrastive_loss = ContrastiveLoss(temperature=0.07)  # Lower temperature for sharper similarities
            
            # Adaptive weights that change during training
            self.register_buffer('training_step', torch.tensor(0))
            
            # Better loss weights for few-shot scenarios
            self.ce_weight = 0.70      # Primary classification loss
            self.focal_weight = 0.20   # Reduced focal weight
            self.contrastive_weight = 0.10  # Contrastive for representation
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, logits, targets, embeddings=None):
        if self.loss_type == "enhanced":
            # Increment training step for adaptive weighting
            if self.training:
                self.training_step += 1
            
            # Compute individual losses
            ce_loss = self.ce_loss(logits, targets)
            focal_loss = self.focal_loss(logits, targets)
            
            # Base loss combination
            total_loss = self.ce_weight * ce_loss + self.focal_weight * focal_loss
            
            # Add contrastive loss if embeddings provided
            if embeddings is not None and embeddings.size(0) > 1:
                cont_loss = self.contrastive_loss(embeddings, targets)
                
                # Adaptive contrastive weight (reduce over time for stability)
                adaptive_cont_weight = self.contrastive_weight
                if self.training_step > 50:  # After warmup period
                    adaptive_cont_weight *= 0.5
                
                total_loss += adaptive_cont_weight * cont_loss
            
            return total_loss
        else:
            return self.primary_loss(logits, targets)

# --- Model Definitions ---
class HGTModel(nn.Module):
    """Heterogeneous Graph Transformer (HGT) Model with Residual and LayerNorm"""
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int,
                 num_layers: int, heads: int, target_node_type: str, dropout_rate: float):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout_rate = dropout_rate

        self.lins = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for node_type in data.node_types:
            self.lins[node_type] = Linear(data[node_type].num_features, hidden_channels)
            self.norms[node_type] = nn.LayerNorm(hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), heads)
            self.convs.append(conv)

        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Initial transformation
        for node_type, x in x_dict.items():
            x = self.lins[node_type](x)
            x = self.norms[node_type](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x_dict[node_type] = x

        # HGT convolutions with residual and LayerNorm
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x = x_dict[node_type] + x_dict_new[node_type]
                x = self.norms[node_type](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                x_dict[node_type] = x

        return self.out_lin(x_dict[self.target_node_type])


class HANModel(nn.Module):
    """Heterogeneous Attentional Network (HAN) Model with Configurable Layers"""
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int,
                 heads: int, target_node_type: str, dropout_rate: float, num_layers: int = 1):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Multiple HAN layers for 1-hop vs 2-hop research
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer can infer input channels
                conv = HANConv(in_channels=-1, out_channels=hidden_channels, 
                             metadata=data.metadata(), heads=heads, dropout=dropout_rate)
            else:
                # Subsequent layers need explicit input channels
                conv = HANConv(in_channels=hidden_channels, out_channels=hidden_channels, 
                             metadata=data.metadata(), heads=heads, dropout=dropout_rate)
            self.convs.append(conv)
        
        # Output linear layer for the target node type
        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Apply multiple HAN layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and dropout after each layer (except the last)
            if i < len(self.convs) - 1:
                for node_type in x_dict:
                    x_dict[node_type] = F.elu(x_dict[node_type])
                    x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout_rate, training=self.training)

        # Get features for the target node type
        target_node_features = x_dict[self.target_node_type]
        
        # Apply final activation and dropout
        target_node_features = F.elu(target_node_features)
        target_node_features = F.dropout(target_node_features, p=self.dropout_rate, training=self.training)
        
        return self.out_lin(target_node_features)


class HANv2Model(nn.Module):
    """Enhanced HAN Model with multiple layers, residual connections, and normalization"""
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int,
                 heads: int, target_node_type: str, dropout_rate: float, num_layers: int = 2):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Initial feature transformation
        self.lins = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for node_type in data.node_types:
            self.lins[node_type] = Linear(data[node_type].num_features, hidden_channels)
            self.norms[node_type] = nn.LayerNorm(hidden_channels)

        # Multiple HAN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HANConv(in_channels=hidden_channels, out_channels=hidden_channels,
                          metadata=data.metadata(), heads=heads, dropout=dropout_rate)
            self.convs.append(conv)
        
        # Output layer
        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Initial transformation
        for node_type, x in x_dict.items():
            x = self.lins[node_type](x)
            x = self.norms[node_type](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x_dict[node_type] = x

        # Multiple HAN layers with residual connections
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x = x_dict[node_type] + x_dict_new[node_type]  # Residual
                x = self.norms[node_type](x)  # LayerNorm
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                x_dict[node_type] = x

        return self.out_lin(x_dict[self.target_node_type])


class HeteroGATModel(nn.Module):
    """Simple Heterogeneous GAT Model"""
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int,
                 heads: int, target_node_type: str, dropout_rate: float):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout_rate = dropout_rate

        # Initial feature transformation for each node type
        self.lins = nn.ModuleDict()
        for node_type in data.node_types:
            self.lins[node_type] = Linear(data[node_type].num_features, hidden_channels)

        # GAT layers for each edge type
        self.convs = nn.ModuleDict()
        for edge_type in data.edge_types:
            edge_key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
            self.convs[edge_key] = GATv2Conv(
                in_channels=(-1, -1),  # Let GATv2Conv infer input dimensions
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout_rate
            )

        # Final output layer
        self.out_lin = Linear(hidden_channels * heads, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Initial feature transformation
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lins[node_type](x)
            x_dict[node_type] = F.elu(x_dict[node_type])
            x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout_rate, training=self.training)

        # Process each edge type separately
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            edge_key = f"{src_type}_{rel_type}_{dst_type}"
            x_dict[dst_type] = self.convs[edge_key](
                (x_dict[src_type], x_dict[dst_type]),
                edge_index
            )
            x_dict[dst_type] = F.elu(x_dict[dst_type])
            x_dict[dst_type] = F.dropout(x_dict[dst_type], p=self.dropout_rate, training=self.training)

        # Get features for target node type
        x = x_dict[self.target_node_type]
        return self.out_lin(x)


# --- Utility Functions ---

def load_hetero_graph(path: str, device: torch.device, target_node_type: str) -> HeteroData:
    """Load HeteroData graph and move it to the specified device."""
    try:
        data = torch.load(path, map_location=torch.device('cpu'), weights_only=False) # Load to CPU first
        # print(f"HeteroData loaded from {path}")
    except Exception as e:
        print(f"Error loading HeteroData: {e}")
        raise ValueError(f"Could not load HeteroData from {path}") from e

    if not isinstance(data, HeteroData):
        raise TypeError(f"Loaded data is not a HeteroData object (got {type(data)}).")

    # Validate target node type and its attributes
    if target_node_type not in data.node_types:
        raise ValueError(f"Target node type '{target_node_type}' not found in graph. Available: {data.node_types}")

    # Use new mask names: train_labeled_mask, train_unlabeled_mask, test_mask
    required_attrs = ['x', 'y', 'train_labeled_mask', 'train_unlabeled_mask', 'test_mask']
    for attr in required_attrs:
        if not hasattr(data[target_node_type], attr):
            raise AttributeError(f"Target node type '{target_node_type}' is missing required attribute: {attr}")

    # Move graph data to the target device
    data = data.to(device)
    # print(f"HeteroData moved to {device}")

    # Ensure masks are boolean type for the target node
    data[target_node_type].train_labeled_mask = data[target_node_type].train_labeled_mask.bool()
    data[target_node_type].train_unlabeled_mask = data[target_node_type].train_unlabeled_mask.bool()
    data[target_node_type].test_mask = data[target_node_type].test_mask.bool()

    # Check if train_labeled_mask has any True values for training supervision
    if data[target_node_type].train_labeled_mask.sum() == 0:
        raise ValueError(f"Cannot train: No nodes available in train_labeled_mask for target node '{target_node_type}'.")

    return data

def get_model(model_name: str, data: HeteroData, args: ArgumentParser) -> nn.Module:
    """Initialize the Heterogeneous GNN model."""
    num_classes = data[args.target_node_type].y.max().item() + 1
    print(f"Number of classes for target node '{args.target_node_type}': {num_classes}")
    
    if model_name == "HeteroGAT":
        print("Using HeteroGAT (simple heterogeneous GAT)")
        model = HeteroGATModel(
            data=data,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            heads=args.heads,
            target_node_type=args.target_node_type,
            dropout_rate=args.dropout_rate
        )
    elif model_name == "HGT":
        print("Using HGT (with residual + LayerNorm)")
        model = HGTModel(
            data=data,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.hgt_layers,
            heads=args.heads,
            target_node_type=args.target_node_type,
            dropout_rate=args.dropout_rate
        )
    elif model_name == "HAN":
        print(f"Using HAN (meta-path attention, {args.han_layers} layers)")
        model = HANModel(
            data=data,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            heads=args.heads,
            target_node_type=args.target_node_type,
            dropout_rate=args.dropout_rate,
            num_layers=args.han_layers
        )
    elif model_name == "HANv2":
        print("Using HANv2 (enhanced HAN with multiple layers, residual connections, and normalization)")
        model = HANv2Model(
            data=data,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            heads=args.heads,
            target_node_type=args.target_node_type,
            dropout_rate=args.dropout_rate,
            num_layers=args.han_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    return model

def train_epoch(model: nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, criterion: nn.Module, target_node_type: str) -> tuple[float, float]:
    """Perform a single training epoch for HeteroData."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data[target_node_type].train_labeled_mask
    if isinstance(out, dict):
        out_target = out[target_node_type]
    else:
        out_target = out
    
    # Handle enhanced loss with embeddings if available
    if isinstance(criterion, EnhancedLoss) and criterion.loss_type == "enhanced":
        # For contrastive loss, we need the embeddings (features before final classification)
        embeddings = out_target[mask]  # Use the pre-classification features
        loss = criterion(out_target[mask], data[target_node_type].y[mask], embeddings)
    else:
        loss = criterion(out_target[mask], data[target_node_type].y[mask])
    
    loss.backward()
    optimizer.step()
    pred = out_target[mask].argmax(dim=1)
    correct = (pred == data[target_node_type].y[mask]).sum().item()
    acc = correct / mask.sum().item() if mask.sum().item() > 0 else 0
    f1 = f1_score(data[target_node_type].y[mask].cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)
    return loss.item(), acc, f1

@torch.no_grad()
def evaluate(model: nn.Module, data: HeteroData, eval_mask_name: str, criterion: nn.Module, target_node_type: str) -> tuple[float, float, float]:
    """Evaluate the model on a given data mask for HeteroData."""
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data[target_node_type][eval_mask_name]
        if isinstance(out, dict):
            out_target = out[target_node_type]
        else:
            out_target = out
        loss = criterion(out_target[mask], data[target_node_type].y[mask])
        pred = out_target[mask].argmax(dim=1)
        correct = (pred == data[target_node_type].y[mask]).sum().item()
        acc = correct / mask.sum().item() if mask.sum().item() > 0 else 0
        y_true = data[target_node_type].y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return loss.item(), acc, f1

def train(model: nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, criterion: nn.Module, args: ArgumentParser, output_dir: str, model_name_fs: str=None) -> dict:
    """Train the model with validation and early stopping."""
    print("\n--- Starting Heterogeneous Training ---")
    start_time = time.time()

    train_losses, train_accs, train_f1s = [], [], []
    val_losses, val_accs, val_f1s = [], [], []
    
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = -1

    model_save_path = os.path.join(output_dir, f"{model_name_fs}_best.pt" if model_name_fs else "graph_best.pt")

    for epoch in range(args.n_epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, data, optimizer, criterion, args.target_node_type)
        val_loss, val_acc, val_f1 = evaluate(model, data, 'train_labeled_mask', criterion, args.target_node_type)
        test_loss, test_acc, test_f1 = evaluate(model, data, 'test_mask', criterion, args.target_node_type)
 
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch: {epoch+1:03d}/{args.n_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")


        # Model selection based on validation loss (more stable for few-shot)
        if val_loss + EPSILON < best_val_loss:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved (Loss: {best_val_loss:.4f}, F1: {best_val_f1:.4f}) Patience: {patience_counter}/{args.patience}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping: patience exceeded OR overfitting detected
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered: patience ({args.patience}) exceeded after {epoch + 1} epochs.")
            break
        elif val_loss < OVERFIT_THRESHOLD:
            print(f"\nEarly stopping triggered: potential overfitting detected (val_loss={val_loss:.4f} < {OVERFIT_THRESHOLD}) after {epoch + 1} epochs.")
            break

    train_time = time.time() - start_time
    print(f"--- Heterogeneous Training Finished in {train_time:.2f} seconds ---")
    if best_epoch != -1:
        print(f"Best model from epoch {best_epoch} saved to {model_save_path}")
    else:
        print("No best model saved (training might have stopped early or validation F1 did not improve).")

    history = {
        "train_loss": train_losses, "train_acc": train_accs, "train_f1": train_f1s,
        "val_loss": val_losses, "val_acc": val_accs, "val_f1": val_f1s,
        "best_epoch": best_epoch, "train_time": train_time,
        "best_val_f1": best_val_f1
    }
    return history

def final_evaluation(model: nn.Module, data: HeteroData, model_path: str, target_node_type: str) -> dict:
    print("\n--- Final Heterogeneous Evaluation on Test Set ---")
    try:
        model.load_state_dict(torch.load(model_path, map_location=data[target_node_type].x.device, weights_only=False))
        print(f"Loaded best model weights from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Best model file not found at {model_path}. Evaluating with current model state (if any).")
    except Exception as e:
        print(f"Warning: Could not load best model weights from {model_path}. Evaluating with last state. Error: {e}")

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data[target_node_type].test_mask
        if isinstance(out, dict):
            out_target = out[target_node_type]
        else:
            out_target = out
        pred = out_target[mask].argmax(dim=1)
        y_true = data[target_node_type].y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist()
    }
    print(f"Target Node: '{target_node_type}'")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix (for target node '{target_node_type}'):\n{metrics['confusion_matrix']}")
    print("--- Heterogeneous Evaluation Finished ---")
    return metrics

def save_results(history: dict, final_metrics: dict, args: ArgumentParser, output_dir: str, model_name_fs: str) -> None:
    """Save training history, final metrics, and plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to safely get last value from list
    def get_last_value(key):
        return history.get(key, [])[-1] if history.get(key) else None

    results_data = {
        "args": vars(args),
        "model_name": model_name_fs,
        "training_history": {
            "final_train_loss": get_last_value("train_loss"),
            "final_train_acc": get_last_value("train_acc"),
            "final_train_f1": get_last_value("train_f1"),
            "final_val_loss": get_last_value("val_loss"),
            "final_val_acc": get_last_value("val_acc"),
            "final_val_f1": get_last_value("val_f1"),
            "best_val_f1": history.get("best_val_f1"),
            "best_epoch": history.get("best_epoch"),
            "total_epochs_run": len(history.get("train_loss", [])),
            "training_time_seconds": history.get("train_time"),
        },
        "final_test_metrics_on_target_node": final_metrics
    }
    results_path = os.path.join(output_dir, f"metrics.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results JSON: {e}")

    plot_path = os.path.join(output_dir, f"training_curves.png")
    try:
        epochs_ran = range(1, len(history.get('train_loss', [])) + 1)
        if not epochs_ran:
            print("No training history to plot.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Training Curves - {model_name_fs}")

        # Plot only available metrics
        if 'train_acc' in history and 'val_acc' in history:
            axes[0].plot(epochs_ran, history['train_acc'], label='Train Accuracy', marker='.')
            axes[0].plot(epochs_ran, history['val_acc'], label='Validation Accuracy', marker='.')
            axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].set_title('Accuracy')
            axes[0].legend(); axes[0].grid(True)

        if 'train_loss' in history and 'val_loss' in history:
            axes[1].plot(epochs_ran, history['train_loss'], label='Train Loss', marker='.')
            axes[1].plot(epochs_ran, history['val_loss'], label='Validation Loss', marker='.')
            axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].set_title('Loss')
            axes[1].legend(); axes[1].grid(True)

        if 'val_f1' in history:
            axes[2].plot(epochs_ran, history['val_f1'], label='Validation F1 Score', marker='.', color='green')
            axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('F1 Score'); axes[2].set_title('Validation F1')
            axes[2].legend(); axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close()
        print(f"Training plots saved to {plot_path}")
    except Exception as e:
         print(f"Error saving plots: {e}")


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Train Heterogeneous Graph Neural Networks for Fake News Detection")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the preprocessed HeteroData graph (.pt file)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=["HGT", "HAN", "HANv2", "RGCN", "HeteroGAT"], help="Heterogeneous GNN model type")
    parser.add_argument("--loss_fn", type=str, default=DEFAULT_LOSS_FN, choices=["ce", "focal", "enhanced", "ce_smooth", "robust"], help="Loss function")
    parser.add_argument("--target_node_type", type=str, default=DEFAULT_TARGET_NODE_TYPE, help="Target node type for classification")
    parser.add_argument("--dropout_rate", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument("--hidden_channels", type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Number of hidden units in GNN layers")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS, help="Number of attention heads for HGT/HAN/GATv2 models")
    parser.add_argument("--hgt_layers", type=int, default=DEFAULT_HGT_LAYERS, help="Number of layers for HGT model")
    parser.add_argument("--han_layers", type=int, default=DEFAULT_HAN_LAYERS, help="Number of layers for HANv2 model")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay (L2 regularization)")
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output_dir_base", type=str, default=RESULTS_DIR, help="Base directory to save results and plots")
    parser.add_argument("--compare_methods", action="store_true", help="Compare few-shot cross-validation with original training")
    parser.add_argument("--comprehensive_evaluation", action="store_true", help="Comprehensive evaluation of the model")
    parser.add_argument("--bootstrap_only", action="store_true", help="Only perform bootstrap validation")
    parser.add_argument("--bootstrap_n_bootstraps", type=int, default=15, help="Number of bootstraps for bootstrap validation")
    parser.add_argument("--analyze_confidence", action="store_true", help="Analyze confidence of the model")

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache(); gc.collect()

    data = load_hetero_graph(args.graph_path, device, args.target_node_type)

    try: # Construct a meaningful scenario name from graph_path
        # the input graph_path is like: graphs_hetero/politifact/8_shot_roberta_hetero_knn_5_smpf10_multiview_0/graph.pt
        parts = args.graph_path.split(os.sep)
        scenario_filename = parts[-2]
        dataset_name = parts[-3]
    except IndexError:
        scenario_filename = "unknown_scenario"
        dataset_name = "unknown_dataset"
    
    # Full model name for filesystem/logging (includes model type, dataset, scenario)
    model_name_fs = f"{args.model}_{dataset_name}_{scenario_filename}"
    # Output directory: base_dir/model_type/dataset_name/scenario_filename/
    output_dir = os.path.join(args.output_dir_base, args.model, dataset_name, scenario_filename)
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- HeteroGraph Configuration ---")
    print(f"Model:           {args.model}")
    print(f"Target Node:     {args.target_node_type}")
    print(f"Graph Path:      {args.graph_path}")
    print(f"Dataset:         {dataset_name}")
    print(f"Scenario:        {scenario_filename}")
    print(f"Output Dir:      {output_dir}")
    
    print("Arguments:")
    for k, v in vars(args).items(): print(f"  {k:<18}: {v}")
    
    print("\n--- HeteroData Info ---")
    for node_type in data.node_types:
        print(f"  Node type: '{node_type}'")
        print(f"    - Num nodes: {data[node_type].num_nodes}")
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            print(f"    - Features: {data[node_type].x.shape[1]}")
        if node_type == args.target_node_type:
            print(f"    - Train labeled mask sum: {data[node_type].train_labeled_mask.sum().item()}")
            print(f"    - Train unlabeled mask sum: {data[node_type].train_unlabeled_mask.sum().item()}")
            print(f"    - Test mask sum: {data[node_type].test_mask.sum().item()}")
            if hasattr(data[node_type],'y') and data[node_type].y is not None:
                 print(f"    - Num classes: {data[node_type].y.max().item() + 1}")

    print("  Edge types and counts:")
    for edge_type in data.edge_types:
        print(f"    - {edge_type}: {data[edge_type].num_edges} edges")
    
    model = get_model(args.model, data, args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Enhanced loss function with label smoothing for few-shot learning
    if args.loss_fn == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Prevents overconfident predictions
        print(f"Using CrossEntropyLoss with label_smoothing=0.1 for few-shot robustness")
    elif args.loss_fn == "focal":
        criterion = FocalLoss()
        print(f"Using FocalLoss for imbalanced data handling")
    elif args.loss_fn == "enhanced":
        criterion = EnhancedLoss(loss_type="enhanced", focal_gamma=1.5, label_smoothing=0.1)
        print(f"Using Enhanced Loss (CE + Focal + Contrastive) with improved hyperparameters")
    elif args.loss_fn == "ce_smooth":
        criterion = EnhancedLoss(loss_type="ce_smooth", label_smoothing=0.1)
        print(f"Using CrossEntropyLoss with enhanced label smoothing")
    elif args.loss_fn == "robust":
        criterion = EnhancedLoss(loss_type="robust", label_smoothing=0.05)
        print(f"Using Robust Few-Shot Loss with confidence penalty and entropy regularization")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print(f"Unknown loss function '{args.loss_fn}', defaulting to CrossEntropyLoss with label smoothing")

    print("\n--- Model Architecture ---")
    print(model)
    print("-------------------------")

    training_history = train(model, data, optimizer, criterion, args, output_dir)
    
    model_path = os.path.join(output_dir, f"graph_best.pt")
    final_metrics = final_evaluation(model, data, model_path, args.target_node_type)
    
    save_results(training_history, final_metrics, args, output_dir, model_name_fs)

    print("\n--- Heterogeneous Pipeline Complete ---")
    print(f"Results, plots, and best model saved in: {output_dir}")
    print("-------------------------------------\n")

if __name__ == "__main__":
    main()
