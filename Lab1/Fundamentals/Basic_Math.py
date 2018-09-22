# region Basic scalar mathematics.
a=1
b=5
c=4

x=3*a+2*b-4*c
print(x)
# endregion
# region Multiplying array structures (not the data.
d=[1,8,4,-3]
e=3*d
print(e)
# endregion
# region Compute the sum of the array.
Sum=0
for value in d:
    Sum = Sum + value
print(Sum)
# endregion
# region compute a dot product of vectors.
g=[3,-4,4]
h=[6,1,4]
Dot=0
for i in range(len(g)):
    Dot= Dot + g[i]*h[i]
print(Dot)
# endregion
# region Display words using a procedure.
def display_word(word):
    print('This is the word: '+word+'!')

name=['John', 'Day', 'Andreea', 'Momo']
for n in name:
    display_word(n)
# endregion
# region A function that computes an N-Dimensional dot product.
def Dot_Product(g,h):
    Dot=0
    for i in range(len(g)):
        Dot= Dot + g[i]*h[i]
    return(Dot)

u=[1,3,5,-2,7]
v=[4,1,8,9,-2]
print(Dot_Product(u,v))
# endregion