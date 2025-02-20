import atomInSmiles
import sentencepiece as spm
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles


smiles = 'Cn1c2nc(Nc3cccc(CO)c3)ncc2cc(-c2c(Cl)cccc2Cl)c1=O'

print(CanonSmiles(smiles))

smiles = CanonSmiles(smiles)

# SMILES -> atom-in-SMILES 
ais_tokens = atomInSmiles.encode(smiles) # '[NH2;!R;C] [CH2;!R;CN] [C;!R;COO] ( = [O;!R;C] ) [OH;!R;C]'

# print(ais_tokens)

sp = spm.SentencePieceProcessor()
sp.Load("AIS_src_sp.model")

tokenized = sp.EncodeAsIds(ais_tokens)

decoded = sp.DecodeIds(tokenized)

# print(tokenized)

print(atomInSmiles.decode(decoded))