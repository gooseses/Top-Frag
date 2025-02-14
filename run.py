import atomInSmiles
import sentencepiece as spm

smiles = 'NCC(=O)O'

# SMILES -> atom-in-SMILES 
ais_tokens = atomInSmiles.encode(smiles) # '[NH2;!R;C] [CH2;!R;CN] [C;!R;COO] ( = [O;!R;C] ) [OH;!R;C]'

sp = spm.SentencePieceProcessor()
sp.Load("AIS_src_sp.model")

tokenized = sp.EncodeAsIds(ais_tokens)

decoded = sp.DecodeIds(tokenized)
