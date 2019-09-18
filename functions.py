import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import cv2
import numpy as np

def dot2D(v,w) :
    x,y = v
    X,Y = w
    return x*X + y*Y

def dot3D(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z    
  
def length2D(v) :
    x,y = v
    return math.sqrt(x*x + y*y)

def length3D(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

  
def vector2D(b,e) :
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def vector3D(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit2D(v):
    x,y = v
    mag = length2D(v)
    return (x/mag, y/mag)
  
def unit3D(v):
    x,y,z = v
    mag = length3D(v)
    return (x/mag, y/mag, z/mag)

def distance2D(p0,p1):
    return length2D(vector2D(p0,p1))
  
def scale2D(v,sc):
    x,y = v
    return (x * sc, y * sc)
  

def scale3D(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add2D(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)

def add3D(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)

# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest 
# distance from pnt to the line and the coordinates of the 
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line. 
# Malcolm Kesson 16 Dec 2012

 #ovu koristim 
def pnt2line(pnt, start, end):
    line_vec = vector2D(start, end)
    pnt_vec = vector2D(start, pnt)
    line_len = length2D(line_vec)
    line_unitvec = unit2D(line_vec)
    pnt_vec_scaled = scale2D(pnt_vec, 1.0/line_len)
    t = dot2D(line_unitvec, pnt_vec_scaled)    
    preciCeLiniju = True
    
    if t < 0.0:
        t = 0.0
        preciCeLiniju = False
    elif t > 1.0:
        t = 1.0
        preciCeLiniju = False

    nearest = scale2D(line_vec, t)
    dist = distance2D(nearest, pnt_vec)
    nearest = add2D(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), preciCeLiniju)


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        cv2.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''


    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255
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


def create_ann():
    '''Implementacija veštacke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann


def load_rgb_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def in_rangeImage(image, lower, upper):
    image = load_rgb_image(image)
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)


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

def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaci i vratiti indeks neurona koji je najviše pobuden'''
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs): # dodaje u niz broj koji je prepoznat
    '''za svaki rezultat pronaci indeks pobednickog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''

    alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

