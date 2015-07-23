#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from PIL import Image
from scipy import optimize


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    # Derivative of sigmoid
    return np.exp(-z)/((1+np.exp(-z))**2)


def construct_input_vector(img):
    img_array = np.array(img)
    vector = np.array([(int(img_array[0, 0, 0]) + int(img_array[0, 0, 1]) + int(img_array[0, 0, 2])) / (3 * 255),
                       (int(img_array[1, 0, 0]) + int(img_array[1, 0, 1]) + int(img_array[1, 0, 2])) / (3 * 255),
                       (int(img_array[2, 0, 0]) + int(img_array[2, 0, 1]) + int(img_array[2, 0, 2])) / (3 * 255),
                       (int(img_array[0, 1, 0]) + int(img_array[0, 1, 1]) + int(img_array[0, 1, 2])) / (3 * 255),
                       (int(img_array[1, 1, 0]) + int(img_array[1, 1, 1]) + int(img_array[1, 1, 2])) / (3 * 255),
                       (int(img_array[2, 1, 0]) + int(img_array[2, 1, 1]) + int(img_array[2, 1, 2])) / (3 * 255),
                       (int(img_array[0, 2, 0]) + int(img_array[0, 2, 1]) + int(img_array[0, 2, 2])) / (3 * 255),
                       (int(img_array[1, 2, 0]) + int(img_array[1, 2, 1]) + int(img_array[1, 2, 2])) / (3 * 255),
                       (int(img_array[2, 2, 0]) + int(img_array[2, 2, 1]) + int(img_array[2, 2, 2])) / (3 * 255)],
                      dtype=float)
    return vector


class Network:
    def __init__(self):
        # Hyperparameters
        self.input_layer_size = 9
        self.output_layer_size = 2
        self.hidden_layer_size = 9
        # Weights
        self.W1 = np.random.rand(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.rand(self.hidden_layer_size, self.output_layer_size)
        # Initializations
        self.z2 = None
        self.a2 = None
        self.z3 = None

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        return sigmoid(self.z3)

    def cost_function(self, X, y):
        self.classification = self.forward(X)
        return np.sum(0.5*sum((y-self.classification)**2))

    def cost_function_prime(self, X, y):
        self.classification = self.forward(X)
        delta3 = np.multiply(-(y-self.classification), sigmoid_prime(self.z3))
        dJdW2 = np.dot(np.array([self.a2, self.a2]).T, delta3)
        delta2 = np.dot(delta3, self.W2.T) * sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2

    def get_params(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        W1_start = 0
        W1_end = self.hidden_layer_size * self.input_layer_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_layer_size, self.hidden_layer_size))
        W2_end = W1_end + self.hidden_layer_size * self.output_layer_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size))

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        print(dJdW1)
        print(dJdW2)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


def compute_numerical_gradient(network, X, y):
    paramsInitial = network.get_params()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        #Set perturbation vector
        perturb[p] = e
        network.set_params(paramsInitial + perturb)
        loss2 = network.cost_function(X, y)

        network.set_params(paramsInitial - perturb)
        loss1 = network.cost_function(X, y)

        #Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

        #Return the value we changed to zero:
        perturb[p] = 0

    #Return Params to original value:
    network.set_params(paramsInitial)

    return numgrad


class Trainer:
    def __init__(self, net):
        self.network = net
        self.X = None
        self.y = None
        self.costs = []
        self.optimization_results = None

    def cost_function_wrapper(self, params, X, y):
        self.network.set_params(params)
        cost = self.network.cost_function(X, y)
        #grad = self.network.cost_function_prime(X, y)
        grad = compute_numerical_gradient(self.network, X, y)
        return cost, grad

    def optimizer_callback(self, params):
        self.network.set_params(params)
        self.costs.append(self.network.cost_function(self.X, self.y))

    def train(self, X, y):
        self.X = X
        self.y = y
        self.costs = []
        params0 = self.network.get_params()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.cost_function_wrapper, params0,
                                 jac=True, method='BFGS', args=(X, y), options=options,
                                 callback=self.optimizer_callback)
        self.network.set_params(_res.x)
        self.optimization_results = _res

    @staticmethod
    def load_training_data():
        training_inputs = []
        training_classes = []
        for class_folder in ['x', 'o', 'noclass']:
            current_class = np.array([0, 0], dtype=float)
            if class_folder == 'x':
                current_class = np.array([1, 0], dtype=float)
            if class_folder == 'o':
                current_class = np.array([0, 1], dtype=float)
            for i in range(1, 6):
                img = Image.open("data/training/" + class_folder + "/00" + str(i) + ".png")
                training_inputs.append(construct_input_vector(img))
                training_classes.append(current_class)
                training_inputs.append(construct_input_vector(img.rotate(90)))
                training_classes.append(current_class)
                training_inputs.append(construct_input_vector(img.rotate(180)))
                training_classes.append(current_class)
                training_inputs.append(construct_input_vector(img.rotate(270)))
                training_classes.append(current_class)
        return np.array(training_inputs), np.array(training_classes)


if __name__ == '__main__':
    if sys.argv[1] == 'X':
        network = Network()
        trainer = Trainer(network)
        tX, ty = trainer.load_training_data()
        trainer.train(tX, ty)
        correct = 0
        wrong = 0
        for class_folder in ['x', 'o', 'noclass']:
            for i in range(1, 5):
                img = Image.open("data/test/" + class_folder + "/00" + str(i) + ".png")
                classification = network.forward(construct_input_vector(img))
                if classification[0] >= 0.5 and classification[1] < 0.5:
                    if class_folder == 'x':
                        correct += 1
                    else:
                        wrong += 1
                        print("data/test/" + class_folder + "/00" + str(i) + ".png")
                elif classification[0] < 0.5 and classification[1] >= 0.5:
                    if class_folder == 'o':
                        correct += 1
                    else:
                        wrong += 1
                        print("data/test/" + class_folder + "/00" + str(i) + ".png")
                elif classification[0] >= 0.5 and classification[1] >= 0.5:
                    wrong += 1
                    print("data/test/" + class_folder + "/00" + str(i) + ".png")
                else:
                    if class_folder == 'noclass':
                        correct += 1
                    else:
                        wrong += 1
                        print("data/test/" + class_folder + "/00" + str(i) + ".png")
        print("Correct: ", correct)
        print("Wrong: ", wrong)
        print(round(100*correct/float(correct+wrong)), "%")
        print("Weights:")
        print(network.W1)
        print(network.W2)
    else:
        X = construct_input_vector(Image.open(sys.argv[1]))
        network = Network()
        trainer = Trainer(network)
        tX, ty = trainer.load_training_data()
        trainer.train(tX, ty)
        classification = network.forward(X)
        if classification[0] >= 0.5 and classification[1] < 0.5:
            print("x")
        elif classification[0] < 0.5 and classification[1] >= 0.5:
            print("o")
        elif classification[0] >= 0.5 and classification[1] >= 0.5:
            print("xo")
        else:
            print("neither")
