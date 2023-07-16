import sys
from model import *
from utils import *
from edgeshape0 import *
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit_heatmaps import mapvalues2mol
from rdkit_heatmaps.utils import transform2png
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

if not os.path.exists('img_0'):
    os.makedirs('img_0')

# Initialize empty dictionaries for storing all bond types and their neighboring bonds
bond_group_values = defaultdict(list)
bond_shapley_values = defaultdict(list)
high_shapley_bond_neighbour_info = defaultdict(list)
high_shapley_bond_types = defaultdict(list)
high_shapley_bond_neighbour_freq = defaultdict(Counter)

def visualize_shapley_values(mol, edge_index, phi_edges, threshold=0.2):
    rdkit_bonds_phi = [0] * len(mol.GetBonds())
    rdkit_bonds = {(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): i for i, bond in enumerate(mol.GetBonds())}

    for i in range(len(phi_edges)):
        init_atom = edge_index[0][i].item()
        end_atom = edge_index[1][i].item()
        phi_value = phi_edges[i]
        bond = mol.GetBondBetweenAtoms(init_atom, end_atom)

        if bond:

            atom_1 = mol.GetAtomWithIdx(init_atom).GetSymbol()
            atom_2 = mol.GetAtomWithIdx(end_atom).GetSymbol()

            # Check if the atoms are in a ring
            atom_1_in_ring = mol.GetAtomWithIdx(init_atom).IsInRing()
            atom_2_in_ring = mol.GetAtomWithIdx(end_atom).IsInRing()

            # If the atom is in a ring, use lowercase symbol, otherwise use uppercase
            atom_1 = atom_1.lower() if atom_1_in_ring else atom_1.upper()
            atom_2 = atom_2.lower() if atom_2_in_ring else atom_2.upper()

            bond_symbol = None
            if bond.GetIsAromatic():
                bond_symbol = ':'
            else:
                if bond.GetBondType().name == 'SINGLE':
                    bond_symbol = '-'
                elif bond.GetBondType().name == 'DOUBLE':
                    bond_symbol = '='
                elif bond.GetBondType().name == 'TRIPLE':
                    bond_symbol = '≡'

            bond_type = bond_symbol.join(sorted([atom_1, atom_2]))

            # Shapley value
            if bond_type not in bond_shapley_values:
                bond_shapley_values[bond_type] = []
            bond_shapley_values[bond_type].append(phi_value)

            # If the Shapley value is above the threshold, then add this bond type to the high_shapley_bond_types list and find its adjacent bonds.
            if phi_value > threshold:
                high_shapley_bond_types[bond_type].append((init_atom, end_atom, phi_value))

                # Initiate dictionary entry for bond type if it does not exist
                if bond_type not in high_shapley_bond_neighbour_info:
                    high_shapley_bond_neighbour_info[bond_type] = []

                # Getting all bonds connected to each atom
                atom_1_bonds = [bond for bond in mol.GetAtomWithIdx(init_atom).GetBonds() if
                                bond.GetOtherAtomIdx(init_atom) != end_atom]  # exclude bond to the other atom
                atom_2_bonds = [bond for bond in mol.GetAtomWithIdx(end_atom).GetBonds() if
                                bond.GetOtherAtomIdx(end_atom) != init_atom]  # exclude bond to the other atom

                atom_1_neighbours = []
                for atom_bond in atom_1_bonds:
                    neighbour_idx = atom_bond.GetOtherAtomIdx(init_atom)

                    atom_neighbour = mol.GetAtomWithIdx(neighbour_idx).GetSymbol()
                    atom_in_ring = mol.GetAtomWithIdx(neighbour_idx).IsInRing()

                    atom_neighbour = atom_neighbour.lower() if atom_in_ring else atom_neighbour.upper()

                    neighbour_bond_symbol = None
                    if atom_bond.GetIsAromatic():
                        neighbour_bond_symbol = ':'
                    elif atom_bond.GetBondType().name == 'SINGLE':
                        neighbour_bond_symbol = '-'
                    elif atom_bond.GetBondType().name == 'DOUBLE':
                        neighbour_bond_symbol = '='
                    elif atom_bond.GetBondType().name == 'TRIPLE':
                        neighbour_bond_symbol = '≡'

                    neighbour_bond_type = neighbour_bond_symbol.join(sorted([atom_1, atom_neighbour]))
                    atom_1_neighbours.append(neighbour_bond_type)

                atom_2_neighbours = []
                for atom_bond in atom_2_bonds:
                    neighbour_idx = atom_bond.GetOtherAtomIdx(end_atom)

                    atom_neighbour = mol.GetAtomWithIdx(neighbour_idx).GetSymbol()
                    atom_in_ring = mol.GetAtomWithIdx(neighbour_idx).IsInRing()

                    atom_neighbour = atom_neighbour.lower() if atom_in_ring else atom_neighbour.upper()

                    neighbour_bond_symbol = None
                    if atom_bond.GetIsAromatic():
                        neighbour_bond_symbol = ':'
                    elif atom_bond.GetBondType().name == 'SINGLE':
                        neighbour_bond_symbol = '-'
                    elif atom_bond.GetBondType().name == 'DOUBLE':
                        neighbour_bond_symbol = '='
                    elif atom_bond.GetBondType().name == 'TRIPLE':
                        neighbour_bond_symbol = '≡'

                    neighbour_bond_type = neighbour_bond_symbol.join(sorted([atom_2, atom_neighbour]))
                    atom_2_neighbours.append(neighbour_bond_type)

                # Join all atom_1 and atom_2 neighbours with ", " and add to high_shapley_bond_neighbour_info
                high_shapley_bond_neighbour_info[bond_type].append(
                    ["; ".join([", ".join(atom_1_neighbours), ", ".join(atom_2_neighbours)])])

            if (init_atom, end_atom) in rdkit_bonds:
                rdkit_bonds_phi[rdkit_bonds[(init_atom, end_atom)]] += phi_value
            if (end_atom, init_atom) in rdkit_bonds:
                rdkit_bonds_phi[rdkit_bonds[(end_atom, init_atom)]] += phi_value

    plt.clf()
    canvas = mapvalues2mol(mol, None, rdkit_bonds_phi, atom_width=0.2, bond_length=0.5, bond_width=0.5)
    img = transform2png(canvas.GetDrawingText())

    return img, bond_shapley_values, bond_group_values, high_shapley_bond_neighbour_info, high_shapley_bond_types

# Read CSV file
df = pd.read_csv('data/'+'test.csv')
smiles_list = df['SMILES'].tolist()

def predict_and_calculate_shapley(model, device, loader, target_class=0):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_shapley_values = []

    with torch.no_grad():
        for idx, data in enumerate(loader):
            if idx >= 202:
                break
            data = data.to(device)
            output, x_g, y_g = model(data,data.x,data.edge_index,data.batch,data.smi_em)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)

            # Calculate Shapley values for edges in this data batch
            shapley_values = edgeshaper(model, data,data.x,data.edge_index,data.batch,data.smi_em, target_class=target_class, device=device)

            # Append the molecule index, data, and its corresponding Shapley values to the list
            shapley_dict = {'index': idx, 'data': data, 'shapley_values': shapley_values}
            total_shapley_values.append(shapley_dict)

            smiles = smiles_list[idx]
            test_mol = Chem.MolFromSmiles(smiles)
            test_mol = Draw.PrepareMolForDrawing(test_mol)

            img, bond_shapley_values, bond_group_values, high_shapley_bond_neighbour_info, high_shapley_bond_types = visualize_shapley_values(
                test_mol, data.edge_index, shapley_values)

            import re
            safe_smiles = re.sub(r'[\\/*?:"<>|]', '_', smiles)

            img_path = os.path.join('img_0', f'{safe_smiles}.png')
            img.save(img_path)

    return total_labels, total_preds, total_shapley_values, high_shapley_bond_neighbour_info, high_shapley_bond_types


cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

# Load CSV file
df = pd.read_csv('data/test.csv')

# Assume your SMILES sequences are in a column named 'SMILES'
smiles_list = df['SMILES'].tolist()

# Preparing for prediction
processed_test = 'data/processed/test.pt'
if not os.path.isfile(processed_test):
    print('Please run create_data.py to prepare data in PyTorch format!')
else:
    test_data = TestbedDataset(root='data', dataset='test')

    # make data PyTorch mini-batch processing ready
    TEST_BATCH_SIZE = 1
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # Load the trained model
    cuda_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(cuda_name)
    model = CMMS_GCL().to(device)
    model_file_name = 'model.pt'
    model.load_state_dict(torch.load(model_file_name))

    G, P, total_shapley_values, high_shapley_bond_neighbour_info, high_shapley_bond_types = predict_and_calculate_shapley(
        model, device, test_loader)

    print(total_shapley_values)
    print(high_shapley_bond_types)
    print(high_shapley_bond_neighbour_info)

    from collections import Counter

    bond_data = defaultdict(list)

    for bond_type, bonds in high_shapley_bond_types.items():
        # Get the count (frequency) of the current bond type
        bond_count = len(bonds)
        # Get the Shapley values of the current bond type
        shapley_values = [val for _, _, val in bonds]
        # Get all neighboring bond types of the current bond type
        neighbour_bonds = [nb for nb_list in high_shapley_bond_neighbour_info.get(bond_type, []) for nb in nb_list]
        # Count the frequency of each neighboring bond type
        neighbour_bonds_counts = dict(Counter(neighbour_bonds))

        # Store all data together
        bond_data[bond_type].append({
            "count": bond_count,
            "shapley_values": shapley_values,
            "neighbour_bonds": neighbour_bonds,
            "neighbour_bonds_counts": neighbour_bonds_counts
        })

        # Now bond_data contains all the information you need
    print(bond_data)

    # Save data to pickle
    with open('bond_data.pkl', 'wb') as f:
        pickle.dump(bond_data, f)






