import numpy as np
import matplotlib.pyplot as plt

# Symbol_1 = '10000111'
# Symbol_0 = '01110010'
Symbol_1 = '11000011'
Symbol_0 = '10010110'

Message = 'Testing.'

Bit_Sequence = ''
for c in Message:
    ASCII_Value = ord(c)
    Bit_Sequence += bin(ASCII_Value)[2:]
print(Bit_Sequence)

Bit_Symbols = ''
for b in Bit_Sequence:
    if b=='0':
        Bit_Symbols+=Symbol_0
    if b=='1':
        Bit_Symbols+=Symbol_1

Bit_Symbols_Data = np.array([2*int(c)-1 for c in list(Bit_Symbols)])
Flip_Bits = np.random.choice(len(Bit_Symbols_Data), 0, replace=False)
bts = lambda bit_string: ''.join([str(c) for c in bit_string])
# print(Flip_Bits)
# print(bts((Bit_Symbols_Data+1)//2))
Bit_Symbols_Data[Flip_Bits]*=-1
# print(bts((Bit_Symbols_Data+1)//2))

Symbol_1_Data = [2*int(c)-1 for c in list(Symbol_1)]
Symbol_0_Data = [2*int(c)-1 for c in list(Symbol_0)]

Decoded_1 = np.convolve(np.flip(Symbol_1_Data, axis=0), Bit_Symbols_Data, 'same') >6
Decoded_0 = np.convolve(np.flip(Symbol_0_Data, axis=0), Bit_Symbols_Data, 'same') >6

Valid_Symbols = np.array(~(~Decoded_0 ^ Decoded_1), dtype=int)
Valid_1_Symbols = np.array((Decoded_1 & ~Decoded_0), dtype=int)
Valid_0_Symbols = np.array((Decoded_0 & ~Decoded_1), dtype=int)

Valid_Symbols_idx = np.where(Valid_Symbols==True)[0]
Recovered_Data = np.zeros_like(Valid_Symbols)
Recovered_Data[Valid_Symbols_idx]=Valid_1_Symbols[Valid_Symbols_idx]

Recovered_Data = np.zeros_like(Valid_Symbols_idx)
Recovered_Data = Valid_1_Symbols[Valid_Symbols_idx]


print(''.join([str(c) for c in Recovered_Data]))

plt.figure()
plt.step(range(len(Recovered_Data)), Recovered_Data, 'r')
plt.show()


