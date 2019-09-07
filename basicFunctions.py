import math

# https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line/45483585

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

