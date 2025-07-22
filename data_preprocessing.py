from operator import index
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from utils import CustomData


def one_of_k_encoding(k, possible_values):
    """
   Convert integer to one-hot representation.
   Enhanced with error handling.
   """
    if k not in possible_values:
        # Use the last value as an "unknown" category instead of raising an error
        k = possible_values[-1] if possible_values else 0
        warnings.warn(f"Unknown value {k}, using default")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    """
   Convert integer to one-hot representation.
   Keeps original logic but adds better error handling.
   """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=True):
    """
   Get atom features.
   Enhanced feature extraction and error handling.
   """
    try:
        # Basic features
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                  one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]

        # Hybridization state - added UNSPECIFIED type
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]
        results += one_of_k_encoding_unk(atom.GetHybridization(), hybridization_types)

        # Aromaticity
        results += [atom.GetIsAromatic()]

        # Number of hydrogen atoms
        if explicit_H:
            results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        # Chirality information - improved error handling
        if use_chirality:
            try:
                chiral_tag = atom.GetChiralTag()
                results += one_of_k_encoding_unk(
                    chiral_tag,
                    [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                     Chem.rdchem.ChiralType.CHI_OTHER]
                )
                results += [atom.HasProp('_ChiralityPossible')]
            except:
                results += [False, False, False, False, False]

        # Additional chemical features (useful for GNP)
        results += [
            atom.IsInRing(),
            float(atom.GetMass()) / 100.0,  # Standardized atomic mass
        ]

        # Try to get Van der Waals radius
        try:
            vdw_radius = float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())) / 5.0
            results += [vdw_radius]
        except:
            results += [1.0]  # Default value

        results = np.array(results).astype(np.float32)
        return torch.from_numpy(results)

    except Exception as e:
        warnings.warn(f"Error extracting atom features: {e}")
        # Return a default feature vector
        default_features = np.zeros(75, dtype=np.float32)
        return torch.from_numpy(default_features)


def edge_features(bond):
    """
   Get bond features.
   Enhanced edge feature extraction.
   """
    try:
        bond_type = bond.GetBondType()

        # Basic bond type features
        features = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]

        # Additional bond features
        features += [
            bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE,  # Stereochemistry
            float(bond.GetBondTypeAsDouble()),  # Bond order as a double
        ]

        return torch.tensor(features, dtype=torch.float32)

    except Exception as e:
        warnings.warn(f"Error extracting bond features: {e}")
        # Return a default feature vector
        return torch.zeros(8, dtype=torch.float32)


def generate_drug_data(mol_graph, atom_symbols, drug_id=None, smiles=None):
    """
   Drug data generation function that fixes the edge_index_batch dimension issue.
   """
    if mol_graph is None:
        warnings.warn(f"Molecule graph is None for drug {drug_id}")
        return None

    if mol_graph.GetNumBonds() == 0:
        warnings.warn(f"Molecule has no bonds for drug {drug_id}")
        # For molecules with no bonds, create a minimal graph structure
        if mol_graph.GetNumAtoms() > 0:
            features = torch.stack([
                atom_features(atom, atom_symbols)
                for atom in mol_graph.GetAtoms()
            ])
            return CustomData(
                x=features,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 8), dtype=torch.float32),
                line_graph_edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_index_batch=torch.empty(0, dtype=torch.long),  # Fix: ensure correct dimensions
                drug_id=drug_id,
                smiles=smiles
            )
        return None

    try:
        # Extract edge information and features
        edges_with_features = []
        for bond in mol_graph.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_features = edge_features(bond)
            edges_with_features.append([begin_idx, end_idx] + bond_features.tolist())

        if not edges_with_features:
            return None

        edges_tensor = torch.tensor(edges_with_features, dtype=torch.float32)
        edge_list = edges_tensor[:, :2].long()
        edge_feats = edges_tensor[:, 2:]

        # Convert to an undirected graph
        edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0)
        edge_feats = torch.cat([edge_feats, edge_feats], dim=0)

        # Extract atom features
        atom_features_list = []
        for atom in mol_graph.GetAtoms():
            features = atom_features(atom, atom_symbols)
            atom_features_list.append(features)

        if not atom_features_list:
            return None

        features = torch.stack(atom_features_list)

        # Construct line graph - improved algorithm
        line_graph_edge_index = torch.LongTensor([])
        if edge_list.nelement() != 0:
            num_edges = edge_list.size(0)
            edge_connections = []

            for i in range(num_edges):
                for j in range(i + 1, num_edges):
                    edge_i = set(edge_list[i].tolist())
                    edge_j = set(edge_list[j].tolist())

                    # If two edges share a node and are not the same edge
                    if len(edge_i & edge_j) > 0:
                        edge_connections.extend([[i, j], [j, i]])

            if edge_connections:
                line_graph_edge_index = torch.tensor(edge_connections, dtype=torch.long).t()

        new_edge_index = edge_list.T

        # Fix: correctly create edge_index_batch
        # edge_index_batch should be a 1D tensor with length equal to the number of edges
        num_edges = new_edge_index.size(1)
        edge_index_batch = torch.zeros(num_edges, dtype=torch.long)

        # Create data object
        data = CustomData(
            x=features,
            edge_index=new_edge_index,
            edge_attr=edge_feats,
            line_graph_edge_index=line_graph_edge_index,
            edge_index_batch=edge_index_batch,
            drug_id=drug_id,
            smiles=smiles
        )

        return data

    except Exception as e:
        warnings.warn(f"Error generating drug data for {drug_id}: {e}")
        return None


def load_drug_mol_data(args):
    """
   Enhanced drug molecule data loading function, keeping the original function name.
   """
    print(f"Loading drug molecule data from: {args.dataset_filename}")

    try:
        data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
        print(f"Successfully loaded CSV file with {len(data)} rows")
    except Exception as e:
        raise FileNotFoundError(f"Could not load data file {args.dataset_filename}: {e}")

    # Collect all unique drugs and SMILES
    drug_smile_dict = {}
    invalid_smiles_count = 0

    for id1, id2, smiles1, smiles2, relation in zip(
            data[args.c_id1], data[args.c_id2],
            data[args.c_s1], data[args.c_s2], data[args.c_y]
    ):
        # Validate SMILES validity
        if pd.notna(smiles1) and smiles1.strip():
            drug_smile_dict[id1] = smiles1.strip()
        else:
            invalid_smiles_count += 1

        if pd.notna(smiles2) and smiles2.strip():
            drug_smile_dict[id2] = smiles2.strip()
        else:
            invalid_smiles_count += 1

    print(f"Collected {len(drug_smile_dict)} unique drugs")
    if invalid_smiles_count > 0:
        print(f"Warning: Found {invalid_smiles_count} invalid SMILES strings")

    # Convert SMILES to molecule objects and collect atom symbols
    drug_id_mol_tup = []
    symbols = set()
    failed_conversions = 0

    print("Converting SMILES to molecule objects...")
    for drug_id, smiles in tqdm(drug_smile_dict.items(), desc='Converting SMILES'):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Optional: Standardize molecule (add hydrogen atoms)
                # mol = Chem.AddHs(mol)
                drug_id_mol_tup.append((drug_id, mol, smiles))

                # Collect atom symbols
                for atom in mol.GetAtoms():
                    symbols.add(atom.GetSymbol())
            else:
                failed_conversions += 1
                warnings.warn(f"Could not convert molecule from SMILES: {drug_id} - {smiles}")
        except Exception as e:
            failed_conversions += 1
            warnings.warn(f"Error converting SMILES {drug_id}: {e}")

    print(f"Successfully converted {len(drug_id_mol_tup)} molecules")
    if failed_conversions > 0:
        print(f"Warning: {failed_conversions} molecule conversions failed")

    symbols = sorted(list(symbols))
    print(f"Found {len(symbols)} atom types: {symbols}")

    # Generate drug graph data
    print("Generating drug graph data...")
    drug_data = {}
    generation_failures = 0

    for drug_id, mol, smiles in tqdm(drug_id_mol_tup, desc='Generating drug graphs'):
        try:
            graph_data = generate_drug_data(mol, symbols, drug_id, smiles)
            if graph_data is not None:
                drug_data[drug_id] = graph_data
            else:
                generation_failures += 1
        except Exception as e:
            generation_failures += 1
            warnings.warn(f"Failed to generate drug graph data for {drug_id}: {e}")

    print(f"Successfully generated {len(drug_data)} drug graphs")
    if generation_failures > 0:
        print(f"Warning: {generation_failures} drug graph generations failed")

    # Save data
    save_data(drug_data, 'drug_data.pkl', args)

    # Data Quality Report
    print("\n=== Data Quality Report ===")
    if drug_data:
        node_counts = [data.x.size(0) for data in drug_data.values() if data.x is not None]
        edge_counts = [data.edge_index.size(1) for data in drug_data.values() if data.edge_index is not None]

        if node_counts and edge_counts:
            print(
                f"Node count stats: Mean={np.mean(node_counts):.1f}, Median={np.median(node_counts):.1f}, Range=[{min(node_counts)}, {max(node_counts)}]")
            print(
                f"Edge count stats: Mean={np.mean(edge_counts):.1f}, Median={np.median(edge_counts):.1f}, Range=[{min(edge_counts)}, {max(edge_counts)}]")

            # Check feature dimension consistency
            feature_dims = [data.x.size(1) for data in drug_data.values() if data.x is not None]
            edge_feature_dims = [data.edge_attr.size(1) for data in drug_data.values() if
                                 data.edge_attr is not None and data.edge_attr.size(0) > 0]

            print(f"Node feature dimensions: {set(feature_dims)}")
            print(f"Edge feature dimensions: {set(edge_feature_dims)}")

            # Check edge_index_batch dimensions
            batch_dims = [data.edge_index_batch.size(0) for data in drug_data.values() if
                          data.edge_index_batch is not None]
            print(
                f"edge_index_batch dimension range: [{min(batch_dims) if batch_dims else 0}, {max(batch_dims) if batch_dims else 0}]")

    return drug_data


def generate_pair_triplets(args):
    """
   Enhanced triplet generation function, ensuring enough negative samples are generated.
   """
    print("Generating drug pair triplets...")

    # Load drug data
    drug_data_path = f'{args.dirname}/{args.dataset.lower()}/drug_data.pkl'
    if not os.path.exists(drug_data_path):
        raise FileNotFoundError(f"Drug data file not found: {drug_data_path}")

    with open(drug_data_path, 'rb') as f:
        drug_data = pickle.load(f)

    drug_ids = list(drug_data.keys())
    print(f"Number of available drugs: {len(drug_ids)}")

    # Load relation data
    try:
        data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
        print(f"Relation data contains {len(data)} rows")
    except Exception as e:
        raise FileNotFoundError(f"Could not load relation data: {e}")

    # Process positive triplets
    pos_triplets = []
    invalid_triplets = 0

    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_y]):
        if id1 not in drug_ids or id2 not in drug_ids:
            invalid_triplets += 1
            continue

        # Relation IDs for DrugBank dataset need to be decremented by 1 (from 1-based to 0-based)
        if args.dataset in ('drugbank',):
            relation = max(0, relation - 1)  # Ensure it does not become negative

        pos_triplets.append([id1, id2, relation])

    print(f"Valid positive triplets: {len(pos_triplets)}")
    if invalid_triplets > 0:
        print(f"Warning: {invalid_triplets} triplets were skipped because a drug was not found")

    if len(pos_triplets) == 0:
        raise ValueError('No valid triplet data found')

    pos_triplets = np.array(pos_triplets)

    # Generate data statistics
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    # Generate negative samples
    print("Generating negative samples...")
    neg_samples = []

    for pos_item in tqdm(pos_triplets, desc='Generating negative samples'):
        h, t, r = pos_item[:3]

        try:
            if args.dataset == 'drugbank':
                neg_heads, neg_tails = _normal_batch(
                    h, t, r, args.neg_ent, data_statistics, drug_ids, args
                )
                # Use new separator format - ensure format conversion is correct
                temp_neg = [f"{neg_h}$h" for neg_h in neg_heads] + \
                           [f"{neg_t}$t" for neg_t in neg_tails]
            else:
                # Handling for other datasets
                h_corrupted = data_statistics["ALL_TRUE_H_WITH_TR"].get((t, r), [])
                t_corrupted = data_statistics["ALL_TRUE_T_WITH_HR"].get((h, r), [])

                # Convert to list and merge, avoiding np.concatenate
                existing_drug_ids = list(set(h_corrupted + t_corrupted))

                # If you need to ensure it's a numpy array, you can convert it like this
                # existing_drug_ids = np.array(existing_drug_ids)

                temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)

            # Limit the number of negative samples and format them
            neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))

        except Exception as e:
            warnings.warn(f"Failed to generate negative samples for triplet {pos_item}: {e}")
            neg_samples.append('')  # Add an empty string as a placeholder

    # Create DataFrame
    df = pd.DataFrame({
        'Drug1_ID': pos_triplets[:, 0],
        'Drug2_ID': pos_triplets[:, 1],
        'Y': pos_triplets[:, 2],
        'Neg samples': neg_samples
    })

    # Add relation_type column for downstream processing
    df['relation_type'] = df['Y'].copy()

    # Convert Y column to binary label (1 for interaction)
    df['Y'] = 1

    # Add fold column for subsequent splitting
    if 'fold' not in df.columns:
        df['fold'] = np.random.RandomState(args.seed).randint(0, args.n_folds, size=len(df))

    # Save original triplet data
    orig_filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df.to_csv(orig_filename, index=False)
    print(f'Original triplet data saved to: {orig_filename}')

    # Process and add negative samples
    print("Processing and adding negative samples...")

    # Split the positive sample dataset into training and testing parts
    train_df = df[df['fold'] != 0].copy()
    test_df = df[df['fold'] == 0].copy()

    # Process using the load_and_process_negative_samples function
    train_df, test_df = load_and_process_negative_samples(
        train_df, test_df, drug_data, fold=args.seed, target_ratio=1.0
    )

    # Save the processed results
    train_filename = f'{args.dirname}/{args.dataset}/processed_train.csv'
    test_filename = f'{args.dirname}/{args.dataset}/processed_test.csv'

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)

    print(f"Processed training set saved to: {train_filename} (Samples: {len(train_df)})")
    print(f"Processed test set saved to: {test_filename} (Samples: {len(test_df)})")

    # Save statistics
    save_data(data_statistics, 'data_statistics.pkl', args)

    # Data quality check
    print("\n=== Triplet Data Quality Report ===")
    print(f"Total triplets: {len(df)}")
    print(f"Number of relation types: {len(df['relation_type'].unique())}")

    # Check negative sample quality
    train_pos = train_df[train_df['Y'] > 0].shape[0]
    train_neg = train_df[train_df['Y'] == 0].shape[0]
    test_pos = test_df[test_df['Y'] > 0].shape[0]
    test_neg = test_df[test_df['Y'] == 0].shape[0]

    print(f"Processed training set: Positive {train_pos}, Negative {train_neg}, Ratio {train_neg / train_pos:.2f}")
    print(f"Processed test set: Positive {test_pos}, Negative {test_neg}, Ratio {test_neg / test_pos:.2f}")

    return df


def generate_additional_negatives(df, drug_ids, neg_per_pos=5):
    """Generate additional negative samples to ensure enough for training."""
    print("Generating additional independent negative samples...")

    # Create a set of known positive pairs to avoid false negatives
    positive_pairs = set()
    for _, row in df.iterrows():
        drug1, drug2 = row['Drug1_ID'], row['Drug2_ID']
        positive_pairs.add((drug1, drug2))
        positive_pairs.add((drug2, drug1))  # Add the reverse pair

    # Randomly generate negative pairs
    neg_samples = []
    target_neg_count = min(len(df) * neg_per_pos, 5000)  # Limit the maximum number

    with tqdm(total=target_neg_count, desc="Generating additional negatives") as pbar:
        while len(neg_samples) < target_neg_count:
            # Randomly select two drugs
            drug1 = np.random.choice(drug_ids)
            drug2 = np.random.choice(drug_ids)

            # Ensure they are not the same drug and not a known positive pair
            if drug1 != drug2 and (drug1, drug2) not in positive_pairs:
                # Randomly select a relation type as a "dummy relation" - only to maintain format consistency
                rel_type = np.random.choice(df['relation_type'].unique())

                # Create a new negative sample row
                new_row = {
                    'Drug1_ID': drug1,
                    'Drug2_ID': drug2,
                    'relation_type': rel_type,  # Dummy relation type
                    'Y': 0,  # Label is 0, indicating no interaction
                    'Neg samples': ''  # Negative samples do not need additional negative samples
                }
                neg_samples.append(new_row)

                # Add to the set of pairs to avoid duplicate generation
                positive_pairs.add((drug1, drug2))
                positive_pairs.add((drug2, drug1))

                pbar.update(1)

    # Create negative sample DataFrame
    neg_df = pd.DataFrame(neg_samples)

    print(f"Generated {len(neg_df)} additional negative samples")

    # Combine positive and negative samples and shuffle
    combined_df = pd.concat([df, neg_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    return combined_df


def load_and_process_negative_samples(df_train, df_test, valid_drug_graphs, fold=0, target_ratio=1.0):
    """
   Load and process negative samples - completed in one go during preprocessing.

   Args:
       df_train: Training data DataFrame.
       df_test: Test data DataFrame.
       valid_drug_graphs: Dictionary of valid drug graphs.
       fold: Fold ID (for random seed).
       target_ratio: Target positive-to-negative sample ratio.

   Returns:
       Processed training and testing DataFrames.
   """
    print("=== Processing Positive and Negative Sample Data ===")

    # Check if negative samples need to be generated
    train_pos_count = df_train[df_train['Y'] > 0].shape[0]
    train_neg_count = df_train[df_train['Y'] == 0].shape[0]

    test_pos_count = df_test[df_test['Y'] > 0].shape[0]
    test_neg_count = df_test[df_test['Y'] == 0].shape[0]

    print(f"Training set: Positive {train_pos_count}, Negative {train_neg_count}")
    print(f"Test set: Positive {test_pos_count}, Negative {test_neg_count}")

    # If negative samples already exist and the ratio is appropriate, return directly
    if train_neg_count >= train_pos_count * target_ratio and test_neg_count >= test_pos_count * target_ratio:
        print("Sufficient negative samples, no additional generation needed.")
        return df_train, df_test

    # Get all available drug IDs
    drug_ids = list(valid_drug_graphs.keys())
    random_gen = np.random.RandomState(fold)

    # Record known drug pairs (to avoid generating false negatives)
    known_pairs = set()
    for _, row in pd.concat([df_train, df_test]).iterrows():
        known_pairs.add((row['Drug1_ID'], row['Drug2_ID']))
        known_pairs.add((row['Drug2_ID'], row['Drug1_ID']))  # Symmetry

    # Generate negative samples for the training set
    if train_neg_count < train_pos_count * target_ratio:
        train_neg_to_add = int(train_pos_count * target_ratio) - train_neg_count
        print(f"Adding {train_neg_to_add} negative samples to the training set...")

        train_neg_samples = []
        attempts = 0
        max_attempts = train_neg_to_add * 10  # Prevent infinite loop

        with tqdm(total=train_neg_to_add, desc="Generating training set negative samples") as pbar:
            while len(train_neg_samples) < train_neg_to_add and attempts < max_attempts:
                # Randomly select a drug pair
                drug1 = random_gen.choice(drug_ids)
                drug2 = random_gen.choice(drug_ids)

                # Ensure it is a valid negative pair
                if drug1 != drug2 and (drug1, drug2) not in known_pairs:
                    # Randomly select a relation type
                    rel_type = random_gen.choice(df_train['relation_type'].unique())

                    # Create negative sample
                    new_row = {
                        'Drug1_ID': drug1,
                        'Drug2_ID': drug2,
                        'Y': 0,  # Negative sample
                        'relation_type': rel_type,  # Maintain format consistency
                        'Neg samples': ''  # Negative samples do not need additional negative samples
                    }

                    train_neg_samples.append(new_row)
                    known_pairs.add((drug1, drug2))
                    known_pairs.add((drug2, drug1))
                    pbar.update(1)

                attempts += 1

        # Add to the training set
        train_neg_df = pd.DataFrame(train_neg_samples)
        df_train = pd.concat([df_train, train_neg_df], ignore_index=True)
        print(f"Added {len(train_neg_df)} training set negative samples")

    # Generate negative samples for the test set
    if test_neg_count < test_pos_count * target_ratio:
        test_neg_to_add = int(test_pos_count * target_ratio) - test_neg_count
        print(f"Adding {test_neg_to_add} negative samples to the test set...")

        test_neg_samples = []
        attempts = 0
        max_attempts = test_neg_to_add * 10  # Prevent infinite loop

        with tqdm(total=test_neg_to_add, desc="Generating test set negative samples") as pbar:
            while len(test_neg_samples) < test_neg_to_add and attempts < max_attempts:
                # Randomly select a drug pair
                drug1 = random_gen.choice(drug_ids)
                drug2 = random_gen.choice(drug_ids)

                # Ensure it is a valid negative pair
                if drug1 != drug2 and (drug1, drug2) not in known_pairs:
                    # Randomly select a relation type
                    rel_type = random_gen.choice(df_test['relation_type'].unique())

                    # Create negative sample
                    new_row = {
                        'Drug1_ID': drug1,
                        'Drug2_ID': drug2,
                        'Y': 0,  # Negative sample
                        'relation_type': rel_type,  # Maintain format consistency
                        'Neg samples': ''  # Negative samples do not need additional negative samples
                    }

                    test_neg_samples.append(new_row)
                    known_pairs.add((drug1, drug2))
                    known_pairs.add((drug2, drug1))
                    pbar.update(1)

                attempts += 1

        # Add to the test set
        test_neg_df = pd.DataFrame(test_neg_samples)
        df_test = pd.concat([df_test, test_neg_df], ignore_index=True)
        print(f"Added {len(test_neg_df)} test set negative samples")

    # Print final label distribution
    print("Label distribution after generating negative samples:")
    print(f"Training set: \n{df_train['Y'].value_counts()}")
    print(f"Test set: \n{df_test['Y'].value_counts()}")

    # Shuffle the data randomly
    df_train = df_train.sample(frac=1, random_state=fold).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=fold).reset_index(drop=True)

    return df_train, df_test


def load_data_statistics(all_tuples):
    """
   This function calculates probabilities for generating negative samples.
   Enhanced with better error handling and statistics.
   """
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        try:
            statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
            statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
            statistics["FREQ_REL"][r] += 1.0
            statistics["ALL_H_WITH_R"][r][h] = 1
            statistics["ALL_T_WITH_R"][r][t] = 1
        except Exception as e:
            warnings.warn(f"Error processing tuple ({h}, {t}, {r}): {e}")

    # Convert to numpy array and remove duplicates
    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    # Calculate statistics
    for r in statistics["FREQ_REL"]:
        try:
            statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
            statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))

            # Avoid division by zero error
            num_tails = len(statistics["ALL_T_WITH_R"][r])
            num_heads = len(statistics["ALL_H_WITH_R"][r])

            if num_tails > 0:
                statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / num_tails
            else:
                statistics["ALL_HEAD_PER_TAIL"][r] = 0.0

            if num_heads > 0:
                statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / num_heads
            else:
                statistics["ALL_TAIL_PER_HEAD"][r] = 0.0

        except Exception as e:
            warnings.warn(f"Error calculating statistics for relation {r}: {e}")

    print('Getting data statistics done!')

    # Print statistics summary
    print(f"Number of relations: {len(statistics['FREQ_REL'])}")
    print(f"Total triplets: {sum(statistics['FREQ_REL'].values())}")

    return statistics


def _corrupt_ent(positive_existing_ents, max_num, drug_ids, args):
    """
   Enhanced entity corruption function for better negative sample generation.
   """
    corrupted_ents = []
    max_attempts = max_num * 10  # Prevent infinite loop
    attempts = 0

    while len(corrupted_ents) < max_num and attempts < max_attempts:
        try:
            # Generate candidate entities
            candidates_needed = max_num - len(corrupted_ents)
            candidates = args.random_num_gen.choice(
                drug_ids,
                min(candidates_needed * 2, len(drug_ids)),
                replace=False
            )

            # Filter out existing positive samples and already selected negative samples
            invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
            mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
            valid_candidates = candidates[mask]

            corrupted_ents.extend(valid_candidates[:candidates_needed])
            attempts += 1

        except Exception as e:
            warnings.warn(f"Error in _corrupt_ent: {e}")
            break

    # If still not enough, fill randomly
    while len(corrupted_ents) < max_num:
        random_drug = args.random_num_gen.choice(drug_ids)
        if random_drug not in corrupted_ents:
            corrupted_ents.append(random_drug)

    return np.array(corrupted_ents[:max_num])


def _normal_batch(h, t, r, neg_size, data_statistics, drug_ids, args):
    """
   Enhanced normal batch negative sample generation.
   """
    neg_size_h = 0
    neg_size_t = 0

    try:
        # Calculate the probability of replacing the head or tail
        tail_per_head = data_statistics["ALL_TAIL_PER_HEAD"].get(r, 1.0)
        head_per_tail = data_statistics["ALL_HEAD_PER_TAIL"].get(r, 1.0)

        # Avoid division by zero error
        total_ratio = tail_per_head + head_per_tail
        if total_ratio > 0:
            prob = tail_per_head / total_ratio
        else:
            prob = 0.5  # Default probability

        # Allocate the number of negative samples for head and tail
        for i in range(neg_size):
            if args.random_num_gen.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        # Generate negative samples
        neg_heads = _corrupt_ent(
            data_statistics["ALL_TRUE_H_WITH_TR"].get((t, r), np.array([])),
            neg_size_h,
            drug_ids,
            args
        )

        neg_tails = _corrupt_ent(
            data_statistics["ALL_TRUE_T_WITH_HR"].get((h, r), np.array([])),
            neg_size_t,
            drug_ids,
            args
        )

        return neg_heads, neg_tails

    except Exception as e:
        warnings.warn(f"Error in _normal_batch: {e}")
        # Return random negative samples as a fallback
        half_size = neg_size // 2
        return (args.random_num_gen.choice(drug_ids, half_size, replace=False),
                args.random_num_gen.choice(drug_ids, neg_size - half_size, replace=False))


def save_data(data, filename, args):
    """
   Enhanced data saving function, keeping the original function name.
   """
    dirname = f'{args.dirname}/{args.dataset}'
    os.makedirs(dirname, exist_ok=True)

    filepath = os.path.join(dirname, filename)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'\nData has been saved to: {filepath}')

        # Verify the saved data
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        if isinstance(loaded_data, dict):
            print(f"Verification: Successfully loaded a dictionary with {len(loaded_data)} items")
            # Display the first few samples
            sample_items = list(loaded_data.items())[:3]
            for i, (key, value) in enumerate(sample_items):
                print(f"  Sample {i + 1}: {key} -> {value}")
                # Verify edge_index_batch dimension
                if hasattr(value, 'edge_index') and hasattr(value, 'edge_index_batch'):
                    if value.edge_index is not None and value.edge_index_batch is not None:
                        edge_count = value.edge_index.size(1)
                        batch_count = value.edge_index_batch.size(0)
                        print(f"    Edge count: {edge_count}, edge_index_batch length: {batch_count}")
                        if edge_count != batch_count:
                            print(f"    Warning: Dimension mismatch!")
        else:
            print(f"Verification: Successfully loaded data of type {type(loaded_data)}")

    except Exception as e:
        raise IOError(f"Failed to save data to {filepath}: {e}")


def split_data(args):
    """Data splitting function."""
    # Check if preprocessed training and test data already exist
    train_filename = f'{args.dirname}/{args.dataset}/processed_train.csv'
    test_filename = f'{args.dirname}/{args.dataset}/processed_test.csv'

    if os.path.exists(train_filename) and os.path.exists(test_filename):
        print(f"Using preprocessed train/test data...")
        train_df = pd.read_csv(train_filename)
        test_df = pd.read_csv(test_filename)
    else:
        # Use original logic to read and split data...
        filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Triplet file not found: {filename}")

        df = pd.read_csv(filename)
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=args.seed)
        train_index, test_index = next(iter(cv_split.split(X=df, y=df[args.class_name])))
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

    # Save in standard format for training
    save_to_filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets'

    for fold_i in range(args.n_folds):
        # Use preprocessed data but maintain fold compatibility
        if fold_i == 0:
            fold_train_df = train_df
            fold_test_df = test_df
        else:
            # For other folds, the data can be reshuffled
            combined = pd.concat([train_df, test_df])
            shuffled = combined.sample(frac=1, random_state=args.seed + fold_i)
            split_idx = int(len(shuffled) * 0.8)
            fold_train_df = shuffled[:split_idx]
            fold_test_df = shuffled[split_idx:]

        # Save
        train_out = f'{save_to_filename}_train_fold{fold_i}.csv'
        test_out = f'{save_to_filename}_test_fold{fold_i}.csv'

        fold_train_df.to_csv(train_out, index=False)
        fold_test_df.to_csv(test_out, index=False)

        print(f'{train_out} saved! (Samples: {len(fold_train_df)})')
        print(f'{test_out} saved! (Samples: {len(fold_test_df)})')

        # Verify split quality
        if args.class_name in fold_train_df.columns:
            train_class_dist = fold_train_df[args.class_name].value_counts()
            test_class_dist = fold_test_df[args.class_name].value_counts()
            print(f"  Training set class distribution: {train_class_dist.to_dict()}")
            print(f"  Test set class distribution: {test_class_dist.to_dict()}")


# Keep the original main function logic
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['drugbank'],
                        help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str, required=True,
                        choices=['all', 'generate_triplets', 'drug_data', 'split'], help='Operation to perform')
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)

    dataset_columns_map = {
        'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
    }

    dataset_file_name_map = {
        'drugbank': ('data/drugbank.tab', '\t'),
    }

    args = parser.parse_args(args=[
        '-d', 'drugbank',
        '-o', 'all'
    ])
    args.dataset = args.dataset.lower()

    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    args.dirname = 'data/preprocessed'

    args.random_num_gen = np.random.RandomState(args.seed)

    try:
        if args.operation in ('all', 'drug_data'):
            print("=== Starting Drug Data Processing ===")
            load_drug_mol_data(args)

        # First, generate triplets, including negative sample processing
        if args.operation in ('all', 'generate_triplets'):
            print("\n=== Starting Triplet Generation and Negative Sample Processing ===")
            generate_pair_triplets(args)

        # Then, perform data splitting
        if args.operation in ('all', 'split'):
            print("\n=== Starting Data Splitting ===")
            args.class_name = 'Y'
            split_data(args)

        print("\n=== Data Preprocessing Complete ===")
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")
        import traceback

        traceback.print_exc()
