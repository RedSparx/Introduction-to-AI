import pyfiglet
import numpy as np
import pickle

f = open('Hopfield_Test_Data.pkl','wb')
Glyph='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
Dictionary={}
for c in range(len(Glyph)):
    # Special handling for I and K because they yield a different size array for this font
    Char = pyfiglet.figlet_format(Glyph[c], font='banner')
    if Glyph[c]!='I' and Glyph[c]!='K':
        Char_Array = np.array(list(Char)).reshape((8,9))[:-1, :-2]
        Num_Array = (Char_Array == '#').astype(int)
    else:
        if Glyph[c]=='I':
            Char_Array = np.array(list(Char)).reshape(8,5)[:-1, :-2]
            Num_Array = (Char_Array == '#').astype(int)
            Num_Array = np.pad(Num_Array, ((0,0),(2,2)), 'constant', constant_values=0)
        if Glyph[c]=='K':
            Char_Array = np.array(list(Char)).reshape(8,8)[:-1, :-2]
            Num_Array = (Char_Array == '#').astype(int)
            Num_Array = np.pad(Num_Array, ((0,0),(0,1)), 'constant', constant_values=0)
    Dictionary.update({Glyph[c]:Num_Array})
pickle.dump(Dictionary, f)
f.close()

