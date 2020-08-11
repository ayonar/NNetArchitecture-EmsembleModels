# Generate Polygon datasets
# Python 2.7

import math, random
from PIL import Image, ImageDraw
import numpy as np
from random import randint

# Polygon generating function
def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ) :
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points

def clip(x, min, max) :
    if( min > max ):  return x
    elif( x < min ):  return min
    elif( x > max ):  return max
    else:             return x


# Generate Datasets PolygonImages with translational invariance.
r=[14,21,12,19,11,17]
c=0
cc=0
for x in range(3,6):
    for a in [0,255]:
        
        tempObject=np.zeros((3600,3000))
        if a==0:
            b=255
        else:
            b=0
        
        for j in range(0,3000):
            verts = generatePolygon(30+np.random.randint(-8,8), 30+np.random.randint(-8,8), np.random.randint(r[cc],r[cc+1]), 0.5, 0, x)
            im = Image.new('L', (60, 60), a)
            imPxAccess = im.load()
            draw = ImageDraw.Draw(im)
            tupVerts = map(tuple,verts)
            draw.polygon( tupVerts, outline=b,fill=b )
            tempObject[:,j] = np.array(im.getdata())
            im.save('image_' + str(j+c*3000) + '.png')
        
        c = c+1
    cc = cc+2

#Polygons = tempObject

#randint(14,21) for trianges
#randint(12,19) for tetragons
#randint(11,17) for pentagons