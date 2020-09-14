import numpy as np
from layers import Lstm, Softmax, Embedding
from numpy.random import randint

START = 1
END = 30
EOS = 0

HIDDEN_SIZE = 100
EMBED_SIZE = 4
INPUT_SIZE = 401
OUTPUT_SIZE = 30
INIT_RANGE = 1.0

LEARNING_RATE = 0.01
CLIP_GRAD = 0.5


class Seq2seq (object):

    def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE, embed_size=EMBED_SIZE, lr=LEARNING_RATE, clip_grad=CLIP_GRAD, init_range=INIT_RANGE):
        # this model will generate a vector representation based on the input
        input_layers = [
            Embedding(input_size, embed_size, init_range),
            Lstm(embed_size, hidden_size, init_range),
        ]

        # this model will generate an output sequence based on the hidden vector
        output_layers = [
            Embedding(output_size, embed_size, init_range),
            Lstm(embed_size, hidden_size, init_range, previous=input_layers[1]),
            Softmax(hidden_size, output_size, init_range)
        ]

        self.input_layers, self.output_layers = input_layers, output_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.clip_grad = clip_grad

    def predict(self, X, start, end, max_length=OUTPUT_SIZE):
        flag = np.zeros(OUTPUT_SIZE)
        # print(flag)
        START = start
        END = end
        flag[START-1] = flag[END-1] = 1
        # reset state
        for layer in self.input_layers:
            layer.initSequence()

        # process input sequence
        for x in X:
            h = x
            for layer in self.input_layers:
                h = layer.forward(h)

        # reset state
        for layer in self.output_layers:
            layer.initSequence()

        # apply output model
        out = [START]
        token = START

        while len(out) < (max_length - 1):
            if token == END:
                break
            # start with last generated token
            h = token

            # go though all layers
            for layer in self.output_layers:
                h = layer.forward(h)

            # select token with highest softmax activation
            token = 0
            token_num = 0
            for i in range(1, OUTPUT_SIZE):
                if h[i-1] > token_num and i != START and i != END and flag[i-1] == 0:
                    token_num = h[i-1]
                    token = i
                    flag[i-1] = 1
            # token = np.argmax(h)
            # print('h=', h)

            # stop if we generated end of sequence token
            if token == END or token == 0:
                break

            # add token to output sequence
            out.append(token)

        if out[len(out)-1] != END and START != END:
            out.append(END)

        return out

    def train(self, X, Y, start, end):
        START = start
        END = end
        # reset state
        for layer in self.input_layers:
            layer.initSequence()

        # forward pass
        for x in X:
            h = x
            for layer in self.input_layers:
                h = layer.forward(h)

        # reset state
        for layer in self.output_layers:
            layer.initSequence()

        for y in Y:
            h = y
            for layer in self.output_layers:
                h = layer.forward(h)

        # backward pass
        for y in reversed(Y):
            delta = y
            for layer in reversed(self.output_layers):
                delta = layer.backward(delta)

        for x in reversed(X):
            delta = np.zeros(self.hidden_size)
            for layer in reversed(self.input_layers):
                delta = layer.backward(delta)

        # gradient clipping
        grad_norm = 0.0

        for layer in self.input_layers + self.output_layers:
            for name, param, grad in layer.params:
                grad_norm += (grad ** 2).sum()

        grad_norm = np.sqrt(grad_norm)

        # sgd
        for layer in self.input_layers + self.output_layers:
            for name, param, grad in layer.params:
                if grad_norm > self.clip_grad:
                    grad /= grad_norm / self.clip_grad
                param -= self.lr * grad

        return self.output_layers[-1].getCost()
