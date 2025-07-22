import os
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pickle
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import warnings


def read_pickle(path):
    """Helper function to read pickle files"""
    from utils import CustomData
    with open(path, 'rb') as f:
        return pickle.load(f)


class EnhancedDrugPairDataset(Dataset):
    """
   Drug pair dataset optimized for Graph Neural Process models.
   Suitable for preprocessed positive and negative sample data.
   """

    def __init__(self, dataframe, drug_graphs_dict, is_train=True,
                 min_nodes=3, min_edges=2, augment_data=False):
        self.dataframe = dataframe
        self.drug_graphs = drug_graphs_dict
        self.is_train = is_train
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        self.augment_data = augment_data

        # Reset statistics structure
        self.stats = {
            'total_rows': len(dataframe),
            'valid_samples': 0,
            'invalid_graphs': 0,
            'missing_drugs': 0,
            'positive_samples': 0,
            'negative_samples': 0
        }

        self.samples = self._create_samples()
        self._print_dataset_stats()

    def _validate_graph(self, graph_data, drug_id="Unknown"):
        """Validate the integrity and quality of the graph data"""
        if not isinstance(graph_data, Data):
            return False, "Not a PyG Data object"

        if graph_data.x is None or graph_data.x.size(0) < self.min_nodes:
            return False, f"Insufficient nodes: {graph_data.x.size(0) if graph_data.x is not None else 0}"

        if graph_data.edge_index is None or graph_data.edge_index.size(1) < self.min_edges:
            return False, f"Insufficient edges: {graph_data.edge_index.size(1) if graph_data.edge_index is not None else 0}"

        # Check the validity of edge indices
        if graph_data.edge_index.max() >= graph_data.x.size(0):
            return False, "Invalid edge indices"

        return True, "Valid"

    def _ensure_graph_attributes(self, graph_data, drug_id="Unknown"):
        """Ensure the graph data has necessary attributes, optimized for the GNP model"""
        if not isinstance(graph_data, Data):
            # Create a minimal valid graph
            warnings.warn(f"Drug {drug_id} is not a valid PyG Data object. Creating minimal graph.")
            graph_data = Data(
                x=torch.randn(self.min_nodes, 70),  # default node feature dimension
                edge_index=torch.randint(0, self.min_nodes, (2, self.min_edges)),
                edge_attr=torch.randn(self.min_edges, 6)  # default edge feature dimension
            )

        # Ensure basic attributes exist
        if graph_data.x is None:
            graph_data.x = torch.randn(max(self.min_nodes, 5), 70)

        if graph_data.edge_index is None or graph_data.edge_index.size(1) == 0:
            num_nodes = graph_data.x.size(0)
            # Create a simple connected graph
            edges = []
            for i in range(min(num_nodes - 1, self.min_edges)):
                edges.append([i, i + 1])
                edges.append([i + 1, i])  # bidirectional edges
            if edges:
                graph_data.edge_index = torch.tensor(edges, dtype=torch.long).t()
            else:
                graph_data.edge_index = torch.empty((2, 0), dtype=torch.long)

        if graph_data.edge_attr is None:
            num_edges = graph_data.edge_index.size(1)
            graph_data.edge_attr = torch.randn(num_edges, 6)

        # Add line graph edge index for DMPNN (if it doesn't exist)
        if not hasattr(graph_data, 'line_graph_edge_index') or graph_data.line_graph_edge_index is None:
            graph_data.line_graph_edge_index = self._create_line_graph_edges(graph_data.edge_index)

        # Add edge_index_batch attribute for DMPNN (if it doesn't exist)
        if not hasattr(graph_data, 'edge_index_batch') or graph_data.edge_index_batch is None:
            num_edges = graph_data.edge_index.size(1)
            graph_data.edge_index_batch = torch.zeros(num_edges, dtype=torch.long)

        # Data augmentation (only during training)
        if self.is_train and self.augment_data:
            graph_data = self._augment_graph(graph_data)

        return graph_data

    def _create_line_graph_edges(self, edge_index):
        """Create edge index for the line graph"""
        if edge_index.size(1) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        # Simplified line graph construction: adjacent edges are connected
        num_edges = edge_index.size(1)
        line_edges = []

        for i in range(num_edges):
            for j in range(i + 1, num_edges):
                # If two edges share a node, they are connected in the line graph
                edge_i = edge_index[:, i]
                edge_j = edge_index[:, j]
                if len(set(edge_i.tolist()) & set(edge_j.tolist())) > 0:
                    line_edges.append([i, j])
                    line_edges.append([j, i])

        if line_edges:
            return torch.tensor(line_edges, dtype=torch.long).t()
        else:
            return torch.empty((2, 0), dtype=torch.long)

    def _augment_graph(self, graph_data):
        """Graph data augmentation to improve the generalization ability of GNP"""
        # Add slight noise to node features
        if graph_data.x is not None and torch.rand(1) < 0.3:  # 30% probability
            noise = torch.randn_like(graph_data.x) * 0.01
            graph_data.x = graph_data.x + noise

        # Add noise to edge features
        if graph_data.edge_attr is not None and torch.rand(1) < 0.3:
            noise = torch.randn_like(graph_data.edge_attr) * 0.01
            graph_data.edge_attr = graph_data.edge_attr + noise

        return graph_data

    def _create_samples(self):
        """Create a list of samples, directly using preprocessed positive and negative samples"""
        processed_samples = []
        for index, row in self.dataframe.iterrows():
            try:
                drug1_id = row['Drug1_ID']
                drug2_id = row['Drug2_ID']

                # Check if the drug exists
                if drug1_id not in self.drug_graphs or drug2_id not in self.drug_graphs:
                    self.stats['missing_drugs'] += 1
                    continue

                # Get original graph data
                graph1_orig = self.drug_graphs[drug1_id]
                graph2_orig = self.drug_graphs[drug2_id]

                # Validate graph validity
                valid1, msg1 = self._validate_graph(graph1_orig, drug1_id)
                valid2, msg2 = self._validate_graph(graph2_orig, drug2_id)

                if not (valid1 and valid2):
                    self.stats['invalid_graphs'] += 1
                    if not valid1:
                        warnings.warn(f"Invalid graph for drug {drug1_id}: {msg1}")
                    if not valid2:
                        warnings.warn(f"Invalid graph for drug {drug2_id}: {msg2}")
                    continue

                # Get relation type and interaction label
                relation_id = int(row.get('relation_type', 0))
                has_interaction = int(row.get('Y', 0))

                # Ensure the graph has all necessary attributes
                try:
                    graph1 = self._ensure_graph_attributes(graph1_orig.clone(), drug1_id)
                    graph2 = self._ensure_graph_attributes(graph2_orig.clone(), drug2_id)
                except Exception as e:
                    warnings.warn(f"Error ensuring graph attributes: {e}")
                    self.stats['invalid_graphs'] += 1
                    continue

                # Add to the sample list
                processed_samples.append((graph1, graph2, relation_id, has_interaction))
                self.stats['valid_samples'] += 1

                # Count positive and negative samples
                if has_interaction == 1:
                    self.stats['positive_samples'] += 1
                else:
                    self.stats['negative_samples'] += 1

            except Exception as e:
                warnings.warn(f"Error processing row {index}: {e}")
                continue

        return processed_samples

    def _print_dataset_stats(self):
        """Print dataset statistics"""
        print(f"\n=== Dataset Statistics ({'Training Set' if self.is_train else 'Validation/Test Set'}) ===")
        print(f"Original number of rows: {self.stats['total_rows']}")
        print(f"Number of valid samples: {self.stats['valid_samples']}")
        print(f"Missing drug graphs: {self.stats['missing_drugs']}")
        print(f"Invalid graph data: {self.stats['invalid_graphs']}")
        print(f"Final number of samples: {len(self.samples)}")

        # Print positive and negative sample statistics
        pos_count = self.stats['positive_samples']
        neg_count = self.stats['negative_samples']
        total_count = pos_count + neg_count
        if total_count > 0:
            print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
            print(f"Positive-negative ratio: {pos_count / total_count:.3f} : {neg_count / total_count:.3f}")
        else:
            print("No valid samples to calculate statistics.")

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.samples)

    def __getitem__(self, index):
        """Get a sample at the specified index"""
        head_graph, tail_graph, relation_id, label = self.samples[index]

        # Clone graph data to avoid modifying original data
        try:
            head_graph_clone = head_graph.clone()
            tail_graph_clone = tail_graph.clone()
        except Exception as e:
            warnings.warn(f"Error cloning graph at index {index}: {e}")
            # Create a default graph as a fallback
            default_graph = Data(
                x=torch.randn(self.min_nodes, 70),
                edge_index=torch.randint(0, self.min_nodes, (2, self.min_edges * 2)),
                edge_attr=torch.randn(self.min_edges * 2, 6),
                line_graph_edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_index_batch=torch.zeros(self.min_edges * 2, dtype=torch.long)
            )
            head_graph_clone = default_graph
            tail_graph_clone = default_graph

        return (head_graph_clone,
                tail_graph_clone,
                torch.tensor(relation_id, dtype=torch.long),
                torch.tensor(label, dtype=torch.float))


def enhanced_collate_fn(batch):
    """
   Collate function that fixes edge_index_batch processing.
   Optimized for preprocessed positive and negative sample data.
   """
    try:
        # Filter out None values
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            raise ValueError("Batch contains no valid items")

        head_graphs = [item[0] for item in valid_batch]
        tail_graphs = [item[1] for item in valid_batch]
        relations = torch.stack([item[2] for item in valid_batch])
        labels = torch.stack([item[3] for item in valid_batch])

        # Validate and fix the edge_index_batch dimension for each graph
        for i, graph in enumerate(head_graphs + tail_graphs):
            graph_type = "head" if i < len(head_graphs) else "tail"
            graph_idx = i if i < len(head_graphs) else i - len(head_graphs)

            # Ensure edge_index attribute exists
            if not hasattr(graph, 'edge_index'):
                warnings.warn(f"Missing edge_index in {graph_type} graph {graph_idx}, creating empty one")
                graph.edge_index = torch.empty((2, 0), dtype=torch.long)

            # Ensure edge_index_batch attribute exists
            if not hasattr(graph, 'edge_index_batch'):
                warnings.warn(f"Missing edge_index_batch in {graph_type} graph {graph_idx}, creating empty one")
                graph.edge_index_batch = torch.zeros(0, dtype=torch.long)

            # Ensure line_graph_edge_index attribute exists
            if not hasattr(graph, 'line_graph_edge_index'):
                warnings.warn(f"Missing line_graph_edge_index in {graph_type} graph {graph_idx}, creating empty one")
                graph.line_graph_edge_index = torch.empty((2, 0), dtype=torch.long)

            # Fix edge_index_batch dimension
            if graph.edge_index.nelement() > 0:
                expected_size = graph.edge_index.size(1)
                actual_size = graph.edge_index_batch.size(0)
                if expected_size != actual_size:
                    graph.edge_index_batch = torch.zeros(expected_size, dtype=torch.long)

        # Create batched data
        try:
            batched_head_graphs = Batch.from_data_list(head_graphs)
            batched_tail_graphs = Batch.from_data_list(tail_graphs)
        except Exception as e:
            warnings.warn(f"Error batching graphs: {e}")
            # Create a default graph as a fallback
            empty_graph = Data(
                x=torch.empty(0, 70),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 6),
                line_graph_edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_index_batch=torch.empty(0, dtype=torch.long)
            )
            return (Batch.from_data_list([empty_graph]),
                    Batch.from_data_list([empty_graph]),
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.float))

        return batched_head_graphs, batched_tail_graphs, relations, labels

    except Exception as e:
        print(f"Error in enhanced_collate_fn: {e}")
        import traceback
        traceback.print_exc()

        # Return an empty batch to avoid interrupting training
        empty_graph = Data(
            x=torch.empty(0, 70),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 6),
            line_graph_edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_index_batch=torch.empty(0, dtype=torch.long)
        )
        return (Batch.from_data_list([empty_graph]),
                Batch.from_data_list([empty_graph]),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float))


class EnhancedDrugDataLoader(torch.utils.data.DataLoader):
    """Enhanced data loader that directly uses preprocessed positive and negative samples"""

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0,
                 pin_memory=True, drop_last=False, **kwargs):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            collate_fn=enhanced_collate_fn,
            **kwargs
        )


def enhanced_split_train_valid(data_df, fold_id, val_ratio=0.15, random_state=42,
                               stratify_col='Y', min_samples_per_class=2):
    """
   Enhanced function to split data into training and validation sets.
   """
    if data_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Check for a predefined 'fold' column
    if 'fold' in data_df.columns:
        print(f"Using the predefined 'fold' column for splitting (fold_id={fold_id})")
        val_df = data_df[data_df['fold'] == fold_id].copy()
        train_df = data_df[data_df['fold'] != fold_id].copy()

        if val_df.empty:
            print(f"Warning: Validation set for fold {fold_id} is empty, falling back to random split")
            # Fallback to random split
        else:
            return train_df, val_df

    # Check if stratified splitting is possible
    can_stratify = (stratify_col in data_df.columns and
                    len(data_df[stratify_col].unique()) > 1 and
                    len(data_df) >= min_samples_per_class * len(data_df[stratify_col].unique()))

    if can_stratify:
        # Check the number of samples in each class
        class_counts = data_df[stratify_col].value_counts()
        min_class_count = class_counts.min()

        if min_class_count < min_samples_per_class:
            print(
                f"Warning: Minimum class sample count ({min_class_count}) is less than the required minimum ({min_samples_per_class})")
            can_stratify = False

    if can_stratify and len(data_df) > int(1 / val_ratio):
        print(f"Performing stratified split (val_ratio={val_ratio}, random_state={fold_id})")
        try:
            train_df, val_df = train_test_split(
                data_df,
                test_size=val_ratio,
                random_state=fold_id,
                stratify=data_df[stratify_col]
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}, using simple random split")
            train_df, val_df = train_test_split(
                data_df,
                test_size=val_ratio,
                random_state=fold_id
            )
    elif len(data_df) > 1:
        print(f"Performing simple random split (val_ratio={val_ratio}, random_state={fold_id})")
        train_df, val_df = train_test_split(
            data_df,
            test_size=val_ratio,
            random_state=fold_id
        )
    else:
        print("Not enough data to split. Using all data for the training set")
        train_df = data_df.copy()
        val_df = pd.DataFrame(columns=data_df.columns)

    return train_df, val_df


def load_ddi_dataset(root, batch_size, subset_size=None, val_ratio=0.15,
                     augment_data=False, min_nodes=3, min_edges=2,
                     # --- New and old parameters coexist, with default values provided ---
                     fold=0,
                     train_file_template='processed_train_fold{}.csv',
                     test_file_template='processed_test_fold{}.csv',
                     train_filename=None,
                     test_filename=None):
    """
   Final revised DDI dataset loading function.
   - Can handle different filename templates.
   - Can automatically adapt to different column names ('head'/'tail' vs 'Drug1_ID'/'Drug2_ID').
   - Can load data by directly specifying filenames (train_filename) or via fold+template.
   """
    print(f"\n=== Loading DDI Dataset ===")
    print(f"Data root directory: {root}")
    print(f"Batch size: {batch_size}, Fold: {fold}")
    print(f"Subset size: {subset_size}, Validation ratio: {val_ratio}")
    print(f"Data augmentation: {augment_data}, Min nodes: {min_nodes}, Min edges: {min_edges}")

    # ==================== Core modification is here ====================
    # Prioritize using directly specified filenames (train_filename, test_filename)
    if train_filename and test_filename:
        print("Mode: Using directly specified filenames.")
        train_path = os.path.join(root, train_filename)
        test_path = os.path.join(root, test_filename)
    # If not specified directly, fall back to using fold and template
    else:
        print("Mode: Constructing filenames using fold and template.")
        train_path = os.path.join(root, train_file_template.format(fold))
        test_path = os.path.join(root, test_file_template.format(fold))
    # =========================================================

    # 1. Load drug graph data (this logic is unchanged)
    drug_graph_path = os.path.join(root, 'drug_data.pkl')
    if not os.path.exists(drug_graph_path):
        raise FileNotFoundError(f"Drug graph data not found: {drug_graph_path}")
    print(f"Loading drug graph data: {drug_graph_path}")
    drug_graphs_dict = read_pickle(drug_graph_path)
    print(f"Number of drug graphs: {len(drug_graphs_dict)}")
    valid_drug_graphs = {k: v for k, v in drug_graphs_dict.items() if
                         isinstance(v, Data) and v.x is not None and v.edge_index is not None}
    print(f"Number of valid drug graphs: {len(valid_drug_graphs)}")

    # 2. Load training and test CSV files
    print(f"Attempting to load training file: {train_path}")
    print(f"Attempting to load test file: {test_path}")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Required data files not found. Please confirm that {train_path} and {test_path} exist.")

    df_train_full = pd.read_csv(train_path)
    df_test_full = pd.read_csv(test_path)
    print(f"Successfully loaded files. Training data rows: {len(df_train_full)}, Test data rows: {len(df_test_full)}")

    # 3. Unify column names! This is the key fix (this logic is unchanged)
    column_map = {
        'head': 'Drug1_ID',
        'tail': 'Drug2_ID',
        'rel': 'relation_type',
        'label': 'Y'
    }
    df_train_full.rename(columns=column_map, inplace=True)
    df_test_full.rename(columns=column_map, inplace=True)
    print("Renamed 'head/tail/rel/label' columns to 'Drug1_ID/Drug2_ID/relation_type/Y'")

    # --- All subsequent logic is based on the unified column names, no further changes needed ---
    # (this logic is unchanged)
    if 'Y' not in df_train_full.columns: df_train_full['Y'] = 1
    if 'relation_type' not in df_train_full.columns: df_train_full['relation_type'] = 0
    if 'Y' not in df_test_full.columns: df_test_full['Y'] = 1
    if 'relation_type' not in df_test_full.columns: df_test_full['relation_type'] = 0

    print("Filtering invalid drug pairs...")
    df_train_filtered = df_train_full[
        df_train_full['Drug1_ID'].isin(valid_drug_graphs.keys()) &
        df_train_full['Drug2_ID'].isin(valid_drug_graphs.keys())
        ].copy()
    df_test_filtered = df_test_full[
        df_test_full['Drug1_ID'].isin(valid_drug_graphs.keys()) &
        df_test_full['Drug2_ID'].isin(valid_drug_graphs.keys())
        ].copy()
    print(f"Training data rows after filtering: {len(df_train_filtered)}")
    print(f"Test data rows after filtering: {len(df_test_filtered)}")

    if df_train_filtered.empty:
        raise ValueError(
            "Error: Training data is empty after filtering! Please check if the drug IDs from the preprocessing script match the IDs in drug_data.pkl.")

    df_train_final, df_val = enhanced_split_train_valid(df_train_filtered, fold, val_ratio)
    df_test_final = df_test_filtered

    if subset_size is not None and subset_size > 0:
        df_train_final = df_train_final.head(subset_size)

    print("Creating dataset objects...")
    train_dataset = EnhancedDrugPairDataset(df_train_final, valid_drug_graphs, is_train=True, augment_data=augment_data)
    val_dataset = EnhancedDrugPairDataset(df_val, valid_drug_graphs, is_train=False)
    test_dataset = EnhancedDrugPairDataset(df_test_final, valid_drug_graphs, is_train=False)

    train_loader = EnhancedDrugDataLoader(train_dataset, batch_size=batch_size, shuffle=True) if len(
        train_dataset) > 0 else None
    val_loader = EnhancedDrugDataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(
        val_dataset) > 0 else None
    test_loader = EnhancedDrugDataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(
        test_dataset) > 0 else None

    print(f"\n=== Data loading complete ===")
    if train_loader: print(f"Training set: {len(train_dataset)} samples, {len(train_loader)} batches")
    if val_loader: print(f"Validation set: {len(val_dataset)} samples, {len(val_loader)} batches")
    if test_loader: print(f"Test set: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# Please replace the load_ddi_dataset_inductive function in dataset.py with this version

def load_ddi_dataset_inductive(root, batch_size, fold, subset_size=None, train_file_template=None,
                               val_file_template=None, test_file_template=None):
    """
   Load DDI dataset for Inductive experimental setup. (Return value issue fixed)
   - Supports loading separate train, validation, and test files.
   - Supports subset_size for quick debugging.
   """
    # print(f"\n=== Loading DDI Dataset (Inductive, subset_size={subset_size}) ===")

    drug_graphs_dict = read_pickle(os.path.join(root, 'drug_data.pkl'))

    def load_and_create_loader(file_template, is_train):
        if file_template is None:
            return None

        file_path = os.path.join(root, file_template.format(fold))
        if not os.path.exists(file_path):
            # print(f"Info: Data file not found, skipping: {file_path}")
            return None

        df = pd.read_csv(file_path)

        # Apply subset_size
        if subset_size and subset_size > 0:
            # print(f"  Applying subset_size={subset_size} to {os.path.basename(file_path)}")
            df = df.head(subset_size)

        df.rename(columns={'head': 'Drug1_ID', 'tail': 'Drug2_ID', 'rel': 'relation_type', 'label': 'Y'}, inplace=True)

        dataset = EnhancedDrugPairDataset(df, drug_graphs_dict, is_train=is_train)
        loader = EnhancedDrugDataLoader(dataset, batch_size=batch_size, shuffle=is_train) if len(dataset) > 0 else None

        # if loader: print(f"  Successfully loaded {os.path.basename(file_path)}: {len(dataset)} samples,
        # {len(loader)} batches")
        return loader

    train_loader = load_and_create_loader(train_file_template, is_train=True)
    val_loader = load_and_create_loader(val_file_template, is_train=False)
    test_loader = load_and_create_loader(test_file_template, is_train=False)

    return train_loader, val_loader, test_loader


def smiles_to_pyg_data(smiles, node_feature_dim):
    """
   Converts a SMILES string to a PyTorch Geometric Data object.
   Note: This is a simplified version for visualization only; features are random.
   """
    mol = Chem.MolFromSmiles(smiles)

    # Extract atom (node) features - here we use random features as placeholders In a real application, you would use
    # more complex feature extraction, but for visualization, the structure is more important
    node_features = torch.randn(mol.GetNumAtoms(), node_feature_dim)

    # Extract bonds (edges)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=node_features, edge_index=edge_index)


# Test code
if __name__ == "__main__":
    pass
