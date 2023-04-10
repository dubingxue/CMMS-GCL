from rdkit import Chem
import pandas as pd
import numpy as np
import networkx as nx
from utils import *
from keras.preprocessing import text
from keras.utils import pad_sequences

def smile_w2v_pad(smile, maxlen_,victor_size):

    tokenizer = text.Tokenizer(num_words=100, lower=False,filters="　")
    tokenizer.fit_on_texts(smile)
    smile_ = pad_sequences(tokenizer.texts_to_sequences(smile), maxlen=maxlen_)
    word_index = tokenizer.word_index
    smileVec_model = {}
    with open("Atom.vec", encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            smileVec_model[word] = coefs
    count=0
    embedding_matrix = np.zeros((100, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector=smileVec_model[word] if word in smileVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del smileVec_model
    return smile_, word_index, embedding_matrix

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr','Pt','Hg','Pb','Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,'other']) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    # Sequence encoder
    smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(smile, 100, 100)
    smi_em = np.array(smi_embedding_matrix)

    return c_size, features, edge_index,smi_em

compound_iso_smiles = []
opts = ['train','test']
for opt in opts:
    df = pd.read_csv('data/' + opt + '.csv')
    # print(df)
    compound_iso_smiles += list( df['SMILES'] )
    # print(compound_iso_smiles)
compound_iso_smiles = set(compound_iso_smiles)
# print(compound_iso_smiles)

smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

# convert to PyTorch data format
processed_train = 'data/processed/' + 'train.pt'
processed_test = 'data/processed/' + 'test.pt'
if ((not os.path.isfile(processed_train)) or (not os.path.isfile(processed_test))):

    df = pd.read_csv('data/' + 'train.csv')
    train_compounds = list(df['SMILES'])
    train_compounds = np.asarray(train_compounds)

    train_Y = np.array(pd.DataFrame(df['Label'])).tolist()
    df = pd.read_csv('data/' + 'test.csv')
    test_compounds, test_Y = list(df['SMILES']), list(df['Label'])
    test_compounds= np.asarray(test_compounds)

    test_Y = np.array(pd.DataFrame(df['Label'])).tolist()

    # make data PyTorch Geometric ready
    print('preparing,' + 'train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset='train', xd=train_compounds, y=train_Y,
                                smile_graph=smile_graph)
    print('preparing,' + 'test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset= 'test', xd=test_compounds, y=test_Y,
                               smile_graph=smile_graph)
    print(processed_train, ' and ', processed_test, ' have been created')
else:
    print(processed_train, ' and ', processed_test, ' are already created')
