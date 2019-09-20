import math
import matplotlib.pyplot as plt
import cv2
import numpy as np

def dot2D(v,w) :
    x,y = v
    X,Y = w
    return x*X + y*Y


def length2D(v) :
    x,y = v
    return math.sqrt(x*x + y*y)


def vector2D(b,e) :
    x,y = b
    X,Y = e
    return (X-x, Y-y)


def unit2D(v):
    x,y = v
    mag = length2D(v)
    return (x/mag, y/mag)
  


def distance2D(p0,p1):
    return length2D(vector2D(p0,p1))
  
def scale2D(v,sc):
    x,y = v
    return (x * sc, y * sc)


def add2D(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)


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



def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def display_image(image, color= False):
    if color:
        cv2.imshow(image)
    else:
        plt.imshow(image, 'gray')



def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''


    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)



def load_rgb_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def in_rangeImage(image, lower, upper):
    image = load_rgb_image(image)
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)



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

