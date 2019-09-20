from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np
import cv2
from keras.datasets import mnist




def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255

def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obucavanje NM,
        odnosno niz ciji su svi elementi 0 osim elementa ciji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)



def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()



def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 ciji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        # scale = region.reshape(1,28,28,1)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    return ann



def create_ann():
    '''Implementacija veštacke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    # ann = Sequential()
    # ann.add(Dense(128, input_dim=784, activation='relu'))
    # ann.add(Dense(10, activation='relu'))
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def neuronska():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_s = []
    for x in x_train[:10:]:
        ret, slika = cv2.threshold(x, 180, 255, cv2.THRESH_BINARY)
        x_train_s.append(slika)

    ann = create_ann()
    ann = train_ann(ann, np.array(prepare_for_ann(x_train_s), np.float32), convert_output(y_train[:10:]))
    model_json = ann.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    ann.save_weights("model.h5")
    print("Saved model to disk")

    return ann