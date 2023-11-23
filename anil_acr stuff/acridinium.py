from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions

# # simple replacement; problem with this is it needs the aniline to be the first atom in smiles string
# acridine = 'C12=CC=CC=C1C=C3C(C=CC=C3)=N2'
# aniline = 'CNS(=O)(=O)CC1=CC=C(C=C1)N'
# acridine_mol = Chem.MolFromSmiles(acridine)
# aniline_mol = Chem.MolFromSmiles(aniline)
#
# mod = Chem.ReplaceSubstructs(acridine_mol,
#                              Chem.MolFromSmiles('N'),
#                              aniline_mol,
#                              replaceAll=True)
# img = Chem.Draw.MolToImage(mod[0])
# img.save('test.png')

# find [NH2] in aniline, replace with dummy, for stitching later
aniline = 'CNS(=O)(=O)CC1=CC=C(C=C1)N'
aniline_mol = Chem.MolFromSmiles(aniline)
aniline_mod = Chem.ReplaceSubstructs(aniline_mol, Chem.MolFromSmarts('[NH2]'), Chem.MolFromSmiles('*'))
modified_aniline_smiles = Chem.MolToSmiles(aniline_mod[0])

# replace the Ra with dummy atom, for stitching later
acr = 'CC(C=C1C)=CC(C)=C1C2=C3C=CC(C(C)(C)C)=CC3=[Ra+]C4=CC(C(C)(C)C)=CC=C24.F[B-](F)(F)F'
acr_mol = Chem.MolFromSmiles(acr)
acr_mod = Chem.ReplaceSubstructs(acr_mol, Chem.MolFromSmarts('[Ra]'), Chem.MolFromSmiles('*'))
modified_acr_smiles = Chem.MolToSmiles(acr_mod[0])

# stitch them together
mod = Chem.ReplaceSubstructs(acr_mod[0],
                             Chem.MolFromSmiles('*'),
                             aniline_mod[0],
                             replaceAll=True,
                             replacementConnectionPoint=aniline_mod[0].GetSubstructMatch(Chem.MolFromSmiles('*'))[0])
smi = Chem.MolToSmiles(mod[0])
smi = smi.replace('*', '[N+]')
print(smi)

new_mol = Chem.MolFromSmiles(smi, sanitize=True)
img = Chem.Draw.MolToImage(new_mol)
img.save('test.png')




# combined = '.'.join([modified_aniline_smiles, modified_acr_smiles])
# combined = combined.replace('*', '%9')
# combined_mol = Chem.MolFromSmiles(combined)
# img = Chem.Draw.MolToImage(combined_mol)
# img.save('test.png')
