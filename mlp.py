from numpy import load, savez, array, arange, tile, random, delete, exp, dot
import math
import numpy as np


class ANN_MLP:
    # MULTILAYER PERCEPTRON NEURAL NETWORK

    def __init__(self, matrix_training, matrix_testing, matrix_validation):
        self.matrix_training = matrix_training
        self.matrix_testing = matrix_testing
        self.matrix_validation = matrix_validation

    # ------------------

    ''' Normalização do dataset (Comprimir os dados dentro de um certo intervalo)

        x - é um valor do dataset
        m - é o valor minimo pertencente ao dataset
        M - é o valor máximo pertencente ao dataset
    '''

    def scale(self, x, m, M):
        return (x-m)/(M-m)


    def unscale(self, x, m, M):
        return x*(M-m)+m


    # ------------------ Activation functions 

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))


    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def ddx_sigmoid(self, x):
        return x * (1 - x)


    # The Tangent Hiperbolic function, which also describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between -1 and 1.
    def tanh(self, x):
        return math.tanh(x)

    # ------------------ Evaluation Parameters

    def rmse(self, e):
        return math.sqrt(((sum(e))**2)/len(e))


    def mape(self, y_actual, y_pred):
        return np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

    # ------------------

    def train(self, max_epochs, learning_rate, hiddenlayer_neurons, m, M):

        input_training = self.matrix_training.T[1:]
        input_training = input_training.T
        target_training = array([self.matrix_training.T[0]])
        
        X = input_training
        y = target_training

        # Number of Neurons in Input Layer = Number of Features in the data set
        inputlayer_neurons = X.shape[1]

        # Number of Neurons at the Output Layer
        output_neurons = 1 

        # weight and bias initialization
        wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
        bh = np.random.uniform(size=(1, hiddenlayer_neurons))
        wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
        bout = np.random.uniform(size=(1, output_neurons))

        #####   Initialization - END  #####

        # Printing of shapes
        print("\nShape X: ", X.shape, "\nShape Y: ", y.shape)
        print("\nShape WH: ", wh.shape, "\nShape BH: ", bh.shape, "\nShape Wout: ", wout.shape, "\nShape Bout: ", bout.shape)

        # Printing of Values
        print("\nwh:\n", wh, "\n\nbh: ", bh, "\n\nwout:\n", wout, "\n\nbout: ", bout)

        max_iterations = 40
        rmse_testing_string = []
        rmse_training_string = []
        for epoch in range(max_epochs):
            
            #####   Forward Propagation - BEGIN   #####
            
            # Input to Hidden Layer = (Dot Product of Input Layer and Weights) + Bias
            hidden_layer_input = (np.dot(X, wh)) + bh

            # Activation of input to Hidden Layer by using Sigmoid Function
            hiddenlayer_activations = self.sigmoid(hidden_layer_input)

            # # Input to Output Layer = (Dot Product of Hidden Layer Activations and Weights) + Bias
            output_layer_input = np.dot(hiddenlayer_activations, wout) + bout

            # Activation of input to Output Layer by using Sigmoid Function
            output = self.sigmoid(output_layer_input)

            #####   Forward Propagation - END  #####    

            #####   Backward Propagation (Método Gradient Descendent) - BEGIN   #####

            E = y.T - output                                            

            slope_output_layer = self.ddx_sigmoid(output) 
            d_output = E * slope_output_layer                           
            Error_at_hidden_layer = d_output.dot(wout.T)                
            slope_hidden_layer = self.ddx_sigmoid(hiddenlayer_activations)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

            wout += hiddenlayer_activations.T.dot(d_output) * learning_rate
            bout += np.sum(d_output, axis=0, keepdims=True) * learning_rate

            wh += X.T.dot(d_hiddenlayer) * learning_rate
            bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * learning_rate

            #####   Backward Propagation - END   #####
            
            # print("epoch number:", epoch, "rmse treino:", self.rmse(E))
            rmse_escalado = self.rmse(E)  # rmse escalado
            rmse_nao_escalado = self.unscale(rmse_escalado, m, M)  # rmse não escalado
            rmse_training_string.append( [ epoch, rmse_nao_escalado ] )  # guarda em array para ser introduzido num gráfico 

            ##### Testing ##### 
            Test = self.test( wh, bh, wout, bout, m, M )
            rmse_testing_string.append( [ epoch, Test['rmse_nao_escalado'] ] )

        #####   TRAINING - END   #####
        
        y_un = self.unscale(y, m, M)
        y_un = y_un.T
        output_un = self.unscale(output, m, M)

        ##### Validação ##### 
        Validation = self.validation( wh, bh, wout, bout, m, M )

        return {'rmse_treino_nao_escalado': rmse_nao_escalado, 'rmse_training_string': rmse_training_string, 'output_un_train': output_un, 'y_un_train': y_un,
                'rmse_teste_nao_escalado': Test['rmse_nao_escalado'], 'rmse_testing_string': rmse_testing_string, 'output_un_test': Test['output_unscaled'], 'y_un_test': Test['y_un'],
                'rmse_validacao_nao_escalado': Validation['rmse_nao_escalado'], 'output_un_validation': Validation['output_unscaled'], 'y_un_validation': Validation['y_un'],
                'wh': wh, 'bh': bh, 'wout': wout, 'bout': bout}


    def early_stopping_technique( self, error_matrix, max_iterations, m, M ):
        ''' Early stopping criteria is basically stopping the training once your loss starts to increase
        (or in other words testing accuracy starts to decrease).'''
        pass


    def test(self, wh, bh, wout, bout, m, M):

        input_testing = self.matrix_testing.T[1:]
        input_testing = input_testing.T
        target_testing = array([self.matrix_testing.T[0]])
        X = input_testing
        y = target_testing

        hidden_layer_input = (np.dot(X, wh)) + bh

        # Activation of input to Hidden Layer by using Sigmoid Function
        hiddenlayer_activations = self.sigmoid(hidden_layer_input)

        # Input to Output Layer = (Dot Product of Hidden Layer Activations and Weights) + Bias
        output_layer_input = np.dot(hiddenlayer_activations, wout) + bout

        # Activation of input to Output Layer by using Sigmoid Function
        output = self.sigmoid(output_layer_input)
        
        E = y.T - output

        rmse_escalado = self.rmse(E)  # rmse escalado
        rmse_nao_escalado = self.unscale(rmse_escalado, m, M)  # rmse não escalado

        y_un = self.unscale(y, m, M)
        y_un = y_un.T
        output_un = self.unscale(output, m, M)
        
        # plt.figure()
        # plt.plot(y_un, 'r')
        # plt.plot(output_un, 'b')
        # plt.show()

        return {'rmse_escalado': rmse_escalado, 'rmse_nao_escalado': rmse_nao_escalado, 
                'output_unscaled': output_un, 'y_un': y_un}


    def validation(self, wh, bh, wout, bout, m, M):

        input_validation = self.matrix_validation.T[1:]
        input_validation = input_validation.T
        target_validation = array([self.matrix_validation.T[0]])
        X = input_validation
        y = target_validation

        hidden_layer_input = (np.dot(X, wh)) + bh

        # Activation of input to Hidden Layer by using Sigmoid Function
        hiddenlayer_activations = self.sigmoid(hidden_layer_input)

        # Input to Output Layer = (Dot Product of Hidden Layer Activations and Weights) + Bias
        output_layer_input = np.dot(hiddenlayer_activations, wout) + bout

        # Activation of input to Output Layer by using Sigmoid Function
        output = self.sigmoid(output_layer_input)
        
        E = y.T - output

        rmse_escalado = self.rmse(E)  # rmse escalado
        rmse_nao_escalado = self.unscale(rmse_escalado, m, M)  # rmse não escalado

        y_un = self.unscale(y, m, M)
        y_un = y_un.T
        output_un = self.unscale(output, m, M)
        
        # plt.figure()
        # plt.plot(y_un, 'r')
        # plt.plot(output_un, 'b')
        # plt.show()

        return {'rmse_escalado': rmse_escalado, 'rmse_nao_escalado': rmse_nao_escalado, 
                'output_unscaled': output_un, 'y_un': y_un}


    def predict( self, data, PH, index, wh, bh, wout, bout, m, M ):
        ''' Avalia a capacidade de previsão. Faz a previsão para x passos à frente e compara
        com valores reais através do RMSE. É por fim preparado para ser mostrado num gráfico. '''
        
        data_size = int(len(data)/2)  # Do conjunto de dados completo considerado, apenas 50% vai ser considerado para testar a previsão. 
        data = data[data_size:]
        data = self.scale(data, m, M)

        #  -------------------
        # Os valores inseridos na string 'index' são organizados de maneira a podererem ser lidos 
        # pelos proximos métodos.
        splitted_lags = index.split(',')
        indexs = list()
        indexs.append(int(-1))
        for i in range(int(len(splitted_lags))):
            indexs.append(int(splitted_lags[i]))
        
        lags_indexs = array(indexs[1:])
        #  -------------------
        # print(data)
        # print(lags_indexs)
        PH_matrix =[]
        max_index = np.amax(lags_indexs)+1
        for i in range(len(data)-max_index*2):  
            input_string = data[i:max_index+i]
            inverted_data_matrix = input_string[::-1]
            
            for j in range(PH):
                lagged_input = inverted_data_matrix[lags_indexs]
                hidden_layer_input = (np.dot(lagged_input, wh)) + bh
                hiddenlayer_activations = self.sigmoid(hidden_layer_input)  # Activation of input to Hidden Layer by using Sigmoid Function
                output_layer_input = np.dot(hiddenlayer_activations, wout) + bout  # Input to Output Layer = (Dot Product of Hidden Layer Activations and Weights) + Bias
                output = self.sigmoid(output_layer_input)  # Activation of input to Output Layer by using Sigmoid Function
                inverted_data_matrix = np.append(output, inverted_data_matrix)

            PH_string = array(inverted_data_matrix[:PH])
            PH_string = PH_string[::-1]
            PH_string = self.unscale(PH_string, m, M)
            PH_matrix.append(PH_string)
        PH_matrix = np.vstack(PH_matrix)[max_index: len(data)-PH]


        data = self.unscale(data, m, M)     
        real_data_matrix = []
        for x in range(max_index, len(data)-PH):

            dat = []
            for y in range(PH):
                dat = np.append(dat, data[y+x])             

            real_data_string = array(dat)
            real_data_matrix.append(real_data_string)
        
        real_data_matrix = np.vstack(real_data_matrix)
        error = real_data_matrix - PH_matrix

        PH_error = []
        for p in range(PH):
            s = self.rmse(error.T[p])
            PH_error.append(s)
                
        return PH_error


    # def ANN_run(self, dataset, PH, index, wh, bh, wout, bout, m, M):
    #     ''' Esta função vai correr uma rede neuronal desenhada para determinado horizonte de previsão. '''
    #     dataset = np.array(dataset)
    #     dataset = self.scale(dataset, m, M)

    #     splitted_lags = index.split(',')
    #     indexs = list()
    #     for i in range(int(len(splitted_lags))):
    #         indexs.append(int(splitted_lags[i]))

    #     lags_indexs = np.array(indexs)

    #     inverted_dataset = dataset[::-1]
    #     for h in range(PH):
    #         lagged_input = inverted_dataset[lags_indexs]

    #         # Feedforward propagation
    #         hidden_layer_input = (np.dot(lagged_input, wh)) + bh
    #         hiddenlayer_activations = self.sigmoid(hidden_layer_input)  # Activation of input to Hidden Layer by using Sigmoid Function
    #         output_layer_input = np.dot(hiddenlayer_activations, wout) + bout  # Input to Output Layer = (Dot Product of Hidden Layer Activations and Weights) + Bias
    #         output = self.sigmoid(output_layer_input)  # Activation of input to Output Layer by using Sigmoid Function
    #         # print(output)
    #         inverted_dataset = np.append(output, inverted_dataset)
    #         # print(inverted_dataset)

    #     final_matrix = inverted_dataset[::-1]
    #     final_matrix = final_matrix[-PH:]
    #     final_matrix = self.unscale(final_matrix, m, M)

    #     return final_matrix
