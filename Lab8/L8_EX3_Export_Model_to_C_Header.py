"""LABORATORY 8
EXERCISE #3: After loading a pre-trained Keras model consisting of flat and dense layers with activation functions,
write a C-style header file that exports the weights, biases and activation function information.
    (a) Load the Keras module and display a summary of its structure on the screen.
    (b) Open the C header file for writing (a plain text file with a .h extension).
    (c) Insert general comments into the header file (the name of the file and the number of trainable parameters).
    (d) Insert any references to header files that may be required by the module.
    (e) Cycle through the model's layers and extract weights and biases.  Insert the weights into a formatted 2D C array
        and the biases into a formatted 1D C array.  Both are of type 'float'.
    (f) Insert comments that list the activation functions for each layer so that they may be implemented.
"""
import os

import keras
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# region Load the Keras input model and display a summary of its structure.
model = load_model(r'Data/Model.h5')
model.summary()
# endregion
# region Open the header file for writing and insert content.
header_filename='NN_Model.h'
with open(r'Data/'+header_filename, 'wt') as f:
    # region Insert general comments into the header file.
    f.write('/'+'*'*50+'\n')
    f.write('  Neural Network Parameters: %s\n'%header_filename)
    f.write('  -------------------------\n\n')
    f.write('  Trainable Parameters: %d\n'%model.count_params())
    f.write('  Generated from Keras module.\n\n')
    f.write('  '+'*'*50+'/\n\n')
    # endregion
    # region Insert references to required header files.
    f.write('#include "MatrixMult.h"\n\n\n')
    # endregion
    # region Cycle through the model extracting weights and biases.
    Layers = len(model.layers)
    for L in range(Layers):
        # region If layers are densely connected, extract the weights.
        if isinstance(model.layers[L], keras.layers.core.Dense):
            # region Extract the weights and write the appropriate code to initialize the array.
            Weights = model.layers[L].get_weights()[0]
            Bias = model.layers[L].get_weights()[1]
            f.write('// Layer %d, %d inputs, %d outputs.\n'%(L, Weights.shape[0], Weights.shape[1]))
            f.write('float Layer%d_Weight[%d][%d] = { ' %(L, Weights.shape[1], Weights.shape[0]))
            # endregion
            # region Cycle through each unit in the layer and dump the weights.
            for unit in range(Weights.shape[1]):
                f.write('\n\n\t\t{ // Unit %d weights.\n\t\t'%unit)
                Cols = 1
                for input in range(Weights.shape[0]):
                    f.write('%6.3f' % Weights[input, unit])
                    if input==(Weights.shape[0]-1):
                        f.write('}')
                    else:
                        f.write(', ')
                    if Cols%10==0:
                        f.write('\n\t\t')
                    Cols += 1
                if unit==(Weights.shape[1]-1):
                    f.write('\n\t\t}; //////////////// End of weights for layer %d. //////////////// \n'%L)
                else:
                    f.write(', ')
            # endregion
            # region Cycle through each unit in the layer and dump the biases.
            Cols = 1
            f.write('\n\nfloat Layer%d_Bias[%d] = { ' % (L, Bias.shape[0]))
            f.write('\n\t\t{ // Biases for units 0 to %d.\n\t\t' % Bias.shape[0])
            for unit in range(Bias.shape[0]):
                f.write('%6.3f' % Bias[unit])
                if unit==(Bias.shape[0]-1):
                    f.write('\n\t\t}; //////////////// End of biases for layer %d. //////////////// \n'%L)
                else:
                    f.write(', ')
                if Cols%10==0:
                    f.write('\n\t\t')
                Cols += 1
            # endregion
        # endregion
    # endregion
    # region Insert a comment listing all of the activation functions for each layer.
    f.write('\n\n/'+'*'*50+'\n\n')
    f.write('  ACTIVATION FUNCTIONS REQUIRED\n\n')

    # Activation function list.
    act_list = []
    for L in range(Layers):
        if isinstance(model.layers[L], keras.layers.core.Activation):
            act_fn = model.layers[L].get_config()['activation']

            # Build a list of unique activation functions.
            if act_fn not in act_list:
                act_list.append(act_fn)

            # Write the layer's corresponding activation function
            f.write('   - Layer %d: %s\n'%(L, act_fn))
    f.write('\n ' + '*' * 50 + '/\n\n')
    # endregion
    # TODO: Insert the code for the required (vector-valued) activation functions.
    # for fn in act_list:
    #     if fn=='relu':
    #         f.write('float act_relu(float x){\n')
    #         f.write('\tif(x>0) return(x);\n')
    #         f.write('\telse return(0);\n\t};')
f.close()
# endregion