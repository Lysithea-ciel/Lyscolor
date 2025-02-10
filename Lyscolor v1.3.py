#Lysithea_ciel's color processing software, intended to aid artists in analyzing color compositions and in picking colors.
#Version 1.3, updated 2/08/2025
from PIL import Image, ImageFilter
import numpy
import copy
import matplotlib as mpl   
from mpl_toolkits import mplot3d
import matplotlib.pyplot as pyplot
import glob
import time
import math

#Start by opening the image. We'll accept either png or jpg.
filename = glob.glob("*.jpg")[0] if glob.glob("*.jpg") else glob.glob("*.png")[0]
fname = filename[:-4]#chop off the last 4 characters, i.e., the file extension, to get just the name. We'll use it later to name our outputs.
image = Image.open(filename, "r")
npimage = numpy.copy(numpy.asarray(image)) #convert image into numpy array; also make a copy so it'll let us actually edit the damn thing
#This "array" is 3 nested lists: It's a list of rows (or columns?), where each row is a list of pixels, where each pixel is a list of values.
#npimage is an array of ints. Good for RGB, not so much for Lab. Let's convert it to float, so we can have decimals.
fimage = npimage.astype('f')

def RGBtoLab(fimage):
    for x in fimage:#go through each pixel
        for y in x:
            #go through each value for that pixel (R, G, B)
            r = y[0] / 255.0; #this sRGB -> linear RGB conversion relies on the values being between 0 and 1, so we need to normalize them
            g = y[1] / 255.0;
            b = y[2] / 255.0;
            
            r = 1.055 * r**(1.0/2.4) - 0.055 if r >= 0.0031308 else 12.92 * r #First, convert our sRGB image to linear RGB
            g = 1.055 * g**(1.0/2.4) - 0.055 if g >= 0.0031308 else 12.92 * g
            b = 1.055 * b**(1.0/2.4) - 0.055 if b >= 0.0031308 else 12.92 * b
            
            #Next, convert our linear RGB image to Oklab
            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
            s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
            
            l_ = numpy.cbrt(l);
            m_ = numpy.cbrt(m);
            s_ = numpy.cbrt(s);
            
            y[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;#l
            y[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;#a
            y[2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;#b
            
            #Now that we're on Oklab, our y vector here has gone from [R G B] to [L a b], where L is lightness.
    return(fimage)

def LabtoRGB(fimage):
    #If we want to actually look at this image, we're gonna have to put it back into RGB.
    for x in fimage:
        for y in x:
            l_ = y[0] + 0.3963377774 * y[1] + 0.2158037573 * y[2];
            m_ = y[0] - 0.1055613458 * y[1] - 0.0638541728 * y[2];
            s_ = y[0] - 0.0894841775 * y[1] - 1.2914855480 * y[2];
            
            l = l_*l_*l_;
            m = m_*m_*m_;
            s = s_*s_*s_;
            
            r = (+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s)
            g = (-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s)
            b = (-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
            
            r = ((r + 0.055)/(1 + 0.055))**2.4 if r >= 0.04045 else r / 12.92#de-linearization
            g = ((g + 0.055)/(1 + 0.055))**2.4 if g >= 0.04045 else g / 12.92
            b = ((b + 0.055)/(1 + 0.055))**2.4 if b >= 0.04045 else b / 12.92
            
            r = 1 if r >=1 else r #cap out the RGB values at 1 so they don't overflow
            g = 1 if g >= 1 else g
            b = 1 if b >= 1 else b
            r = 0 if r <= 0 else r #bottom out the RGB values at 0 so they don't underflow
            g = 0 if g <= 0 else g
            b = 0 if b <= 0 else b
            
            y[0] = r * 255.0#de-normalization
            y[1] = g * 255.0
            y[2] = b * 255.0
    return(fimage)

def LchtoLab(fimage):
    for x in fimage:
        for y in x:
            #a = C*cos(hdeg) | b = C*sin(hdeg)
            C = copy.deepcopy(y[1]);
            y[1] = C*numpy.cos(y[2]);
            y[2] = C*numpy.sin(y[2]);
    return(fimage)

def RGBtoHSL(colors):#input 'colors' must be a 2 x n x 3 matrix, where n is the number of individual points whose color we're dealing with.
    #The matrix is basically two identical lists of points, each with 3 rgb values for every point. There's two identical lists to deal with something else later; we can just ignore that and only look at one of them.
    hslcolors = copy.deepcopy(colors[1])
    for index, rgb in enumerate(colors[1]):
        Xmax = max(rgb)#V (value)
        Xmin = min(rgb)#V - C
        C = Xmax - Xmin        #2(V - L) (chroma)
        L = (Xmax + Xmin)/2    #V - C/2 (lightness)
        #Hue comes from a piecewise function. Hue lies in the range of 0-360 deg.
        if C == 0:
            H = 0 #(hue)
        elif Xmax == rgb[0]:#V = R
            H = 60*((rgb[1] - rgb[2])/C % 6) #60 deg * ((G-B)/C mod 6)
        elif Xmax == rgb[1]:#V = G
            H = 60*((rgb[2] - rgb[0])/C + 2) #60 deg * ((B-R)/C + 2)
        else: #V = B
            H = 60*((rgb[0] - rgb[1])/C + 4) #60 deg * ((R-G)/C + 4)
        #Saturation is also piecewise
        if L == 0 or L == 1:
            Sl = 0 #Distinct saturation?
        else:
            Sl = C/(1 - abs(2*L - 1))
        hslcolors[index] = [H, Sl, L]
    return(hslcolors)

def ImageProcessing(fimage):
    fimage = RGBtoLab(fimage)
    #Now let's make some edits. We want greyscale, "ascale" and "bscale". Maybe also "chromascale" and "huescale" later.

    #Greyscale:
    gfimage = copy.deepcopy(fimage);#make a copy to mess with
    for x in gfimage:
        for y in x:
            #Let's make our image greyscale. This is done by just making everything except the lightness value equal to zero.
            y[1] = 0;#a
            y[2] = 0;#b
    gfimage = LabtoRGB(gfimage)
    #Convert array back into an image
    npimage = gfimage.astype(numpy.uint8)#convert back to integers
    temp = Image.fromarray(npimage)
    temp.save(fname+" greyscale.png")
    print("Greyscale image created.")

    #"Ascale":
    #Show just the 'a' value, which is the green-red spectrum.
    gfimage = copy.deepcopy(fimage)
    for x in gfimage:
        for y in x:
            #print(y)
            y[0] = 0.81; #Lightness is on a range of 0 to 1 here
            y[1] = y[1] + 0.007 #edit a and b to make it a little easier to see
            y[2] = 0.03
    gfimage = LabtoRGB(gfimage)
    npimage = gfimage.astype(numpy.uint8)
    temp = Image.fromarray(npimage)
    temp.save(fname+" green-red.png")
    print('Green-red spectrum image created.')

    #"Bscale":
    #Show just the 'b' value, which is the blue-yellow spectrum.
    gfimage = copy.deepcopy(fimage)
    for x in gfimage:
        for y in x:
            y[0] = 0.81;
            y[1] = 0.03;
            y[2] = y[2] + 0.007
    gfimage = LabtoRGB(gfimage)
    npimage = gfimage.astype(numpy.uint8)
    temp = Image.fromarray(npimage)
    temp.save(fname+" blue-yellow.png")
    print('Blue-yellow spectrum image created.')

    #Now LCh it- this means turning our cartesian Lab into polar LCh.
    #C = sqrt(a^2 + b^2) | hdeg = atan2(b,a)
    #hdeg is hue angle, expressed here in radians.
    lchimage = copy.deepcopy(fimage);
    for x in lchimage:
        for y in x:
            C = numpy.sqrt(y[1]*y[1] + y[2]*y[2]);
            hdeg = numpy.arctan2(y[2], y[1]);#radians
            if hdeg < 0:#arctan2 shits out values on a range of pi to -pi like a lunatic, so this compensates for that
                hdeg += 2*numpy.pi;
            y[1] = C;
            y[2] = hdeg;
    #TODO: consider finding the max and min hue angles, and displaying the resultant sweep of the color wheel

    gfimage = copy.deepcopy(lchimage);#another copy, another edit.
    #Now that we have our LCh image (array), let's visualize our hue angle
    for x in gfimage:
        for y in x:
            y[0] = 0.9;#constant lightness
            y[1] = 0.03;#constant chroma. Both of these values are actually pretty specific; deviate much from these and things go to hell.
    gfimage = LchtoLab(gfimage)
    gfimage = LabtoRGB(gfimage)
    npimage = gfimage.astype(numpy.uint8)
    temp = Image.fromarray(npimage)
    temp.save(fname+" hue angle.png")
    print('Hue angle image created.')

    #Let's look at chroma too.
    gfimage = copy.deepcopy(lchimage);
    for x in gfimage:
        for y in x:
            y[0] = .9;#constant lightness
            y[2] = .1;#constant hue
    gfimage = LchtoLab(gfimage)
    gfimage = LabtoRGB(gfimage)
    npimage = gfimage.astype(numpy.uint8)
    temp = Image.fromarray(npimage)
    temp.save(fname+" chroma.png")
    print('Chroma image created.')


#Let user choose if they want to actually do all those image edits, or just jump straight to extracting and graphing the colors.
if input("Create image variants? On a slower computer with a large image, this might take a long time. (y/n) ") == "y":
    starttime = time.time()
    ImageProcessing(fimage)
    endtime = time.time()#every END is a new BEGINNING
    elapsed = int(endtime - starttime)
    minelapsed = int(elapsed / 60)
    secelapsed = elapsed % 60
    print(minelapsed, "minutes", secelapsed, "seconds elapsed.")


#Let's pick some colors from the original image, no pre-processing involved.
#First, blur the image a bit so our random color choices are more consistent with the rest of the image.
#we need to make another copy of the original 'image' though, since that one's read-only
nimage = copy.deepcopy(image)
blurred = nimage.filter(ImageFilter.GaussianBlur(radius=3))
#blurred.save(fname+" blurred.png")#possibly necessary?
npimage = numpy.copy(numpy.asarray(blurred))
#We'll get the colors from a selection of points across the image.
height = blurred.height
width = blurred.width
base = 30#Determines the number of points spread across the image. The total count will be between (base/2)^2 and base, depending on the dimensions.
if height/width < 1:#picture is wider than it is tall
    ypoints = height/(width + height) * base
    ypoints = numpy.floor(ypoints) #we need an integer
    xpoints = base - ypoints
else:#picture is taller than it is wide
    xpoints = width/(height + width) * base
    xpoints = numpy.floor(xpoints)
    ypoints = base - xpoints
#xpoints and ypoints are currently float64s (despite being floored) but we need them as integers
xpoints = int(xpoints)#Number of columns of points
ypoints = int(ypoints)#Number of rows of points
colors = numpy.ndarray([xpoints*ypoints, 3], dtype = numpy.uint8)#create array for storing the RGB values for each point
#Now space them out evenly
interval = width / xpoints
xcoords = list(range(xpoints))#this starts at 0 and ends at xpoints - 1
for index, x in enumerate(xcoords):
    xcoords[index] = int((x + 1)*interval - interval/2)#the '-interval/2' bit centers the points
interval = height / ypoints
ycoords = list(range(ypoints))
for index, y in enumerate(ycoords):
    ycoords[index] = int((y + 1)*interval - interval/2)
#Now run through the grid we've come up with and grab the color at each point
for xindex, x in enumerate(xcoords):
    for yindex, y in enumerate(ycoords):
        colors[xindex * ypoints + yindex] = npimage[y, x][:3]#the [:3] grabs only the first 3 elements of the list. Our image could be in RGBA, but we only want to use the RGB components.

#Feed colors into RGB -> Oklab function. It's expecting a 2d array of vectors; we have a 1d array of vectors.
colors = numpy.array([colors, colors])#Behold, 2 dimensions.

Labcolors = copy.deepcopy(colors)
Labcolors = Labcolors.astype(numpy.float32)
Labcolors = RGBtoLab(Labcolors)
#We need our RGB colors to be on a scale of 0 to 1 because matplotlib is weird
colors = colors.astype(numpy.float32)
colors = colors/255.0
#Now plot points, using the Lab coordinates for the position and RGB values for the color.
fig = pyplot.figure()

# #The ranges for the 'a' and 'b' axes are about the same usually, but L is like an order of magnitude above them. This difference seems to mess various things up, so let's normalize.
# #We define our bounds by the highest and lowest values of a or b.
# Graphcolors = copy.deepcopy(Labcolors)
# huemin = min(min(Graphcolors[0,:,1]),min(Graphcolors[0,:,2]))
# huemax = max(max(Graphcolors[0,:,1]),max(Graphcolors[0,:,2]))
# # Then fit all the dimensions into that same range.
# multiplier = (huemax - huemin) / (max(Graphcolors[0,:,0]) - min(Graphcolors[0,:,0]))
# Graphcolors[0,:,0] = numpy.multiply(Graphcolors[0,:,0], multiplier)
# multiplier = (huemax - huemin) / (max(Graphcolors[0,:,1]) - min(Graphcolors[0,:,1]))
# Graphcolors[0,:,1] = numpy.multiply(Graphcolors[0,:,1], multiplier)
# multiplier = (huemax - huemin) / (max(Graphcolors[0,:,2]) - min(Graphcolors[0,:,2]))
# Graphcolors[0,:,2] = numpy.multiply(Graphcolors[0,:,2], multiplier)
# # And finally, center everything around (0, 0, 0).
# Graphcolors[0,:,0] = numpy.subtract(Graphcolors[0,:,0], numpy.mean(Graphcolors[0,:,0]))
# Graphcolors[0,:,1] = numpy.subtract(Graphcolors[0,:,1], numpy.mean(Graphcolors[0,:,1]))
# Graphcolors[0,:,2] = numpy.subtract(Graphcolors[0,:,2], numpy.mean(Graphcolors[0,:,2]))

#Time to plot
ax = fig.add_subplot(projection='3d')
ax.set(xlabel="Lightness", ylabel="Red-green", zlabel="Blue-yellow", title="Image Colors in OKLab")
for i in range(xpoints*ypoints):
    ax.scatter(Labcolors[0, i, 0], Labcolors[0, i, 1], Labcolors[0, i ,2], color=(colors[0, i, 0], colors[0, i, 1], colors[0, i, 2]))

#The color scatter plots we get show two main trends: 1. Colors in a sphere and 2. Colors on a plane.
#Colors in a sphere can be easily translated into a range of Lab values; translating that into a range of RGB values is messier, but we can probably simplify it a lot.

#Colors on a plane can be displayed as a slice of the OKLab gamut.
#To do this, we first find an average plane from our list of points.
centroid = numpy.mean(Labcolors[0], axis=0)
#Labcolors[0] = Graphcolors[0] + centroid
Labcolors[1] = Labcolors[0] - centroid#center the points

#Let's remove outliers to give more clean results
distance = numpy.zeros(len(Labcolors[1]))
for index, val in enumerate(Labcolors[1]):
    distance[index] = math.sqrt(val[0]**2 + val[1]**2 + val[2]**2)
stdev = numpy.std(distance)
trimLabcolors = copy.deepcopy(Labcolors[1])
trimLabcolors = trimLabcolors.tolist()
for index, val in enumerate(distance):
    if distance[index] >= 100*stdev:#actually this might be cutting off the edges of the gradient too. Really, it needs to be distance from the plane, not the centroid.
        trimLabcolors[index] = [999, 999, 999]
num = trimLabcolors.count([999, 999, 999])
for i in range(num):
    trimLabcolors.remove([999, 999, 999])
trimLabcolors = numpy.array(trimLabcolors)

#Credit for this SVD stuff goes to ChatGPT. Welcome to the future- "How I Became A 10x Programmer With One Easy Trick"
_, _, vh = numpy.linalg.svd(trimLabcolors)#Singular value decomposition (compsci black magic)
normal = vh[2]#"The normal to the plane is the last singular vector". What do the results of SVD even mean?

#The centroid and normal define the plane. Now, to be able to display it, we'll create a grid of points that lie on this plane and find the RGB value of each.
#start by finding two orthogonal vectors that lie within the plane
#two vectors are orthogonal if their dot product is zero
#so normal[0]*orth1[0] + normal[1]*orth1[1] + normal[2]*orth1[2] = 0
orth1 = [0, 0, 1] if normal[2] == 0 else [0, 1, -normal[1]/normal[2]]
#we get our second orthogonal vector by just taking the cross product
orth2 = numpy.cross(normal, orth1)
#normalize orthogonal vectors
orth1 = numpy.divide(orth1, math.sqrt(orth1[0]**2 + orth1[1]**2 + orth1[2]**2))
orth2 = numpy.divide(orth2, math.sqrt(orth2[0]**2 + orth2[1]**2 + orth2[2]**2))

#Set up the grid of points
#First, define bounds. These aren't used currently, but this is the sort of range that the picked colors generally fall into.
#For reference, L is usually between 0.5 and 1.0, a between -0.1 and 0.1, and b between -0.1 and 0.1.
#the fact that these are unequal means the dimensions of our grid will depend on whether it's more horizontal or vertical.
#fuck
#let's just pretend it's not a problem for now
lmin = 1.414*min(trimLabcolors[:,0])#This is where removing the outliers before really helps
lmin = 0 if lmin < 0 else lmin#Cap minimum lightness at zero
lmax = 1.414*max(trimLabcolors[:,0])#And then we just multiply by sqrt(2) like we know what we're doing.
lmax = 1 if lmax > 1 else lmax#Cap maximum lightness at 1
amin = min(trimLabcolors[:,1])
amax = max(trimLabcolors[:,1])
bmin = min(trimLabcolors[:,2])
bmax = max(trimLabcolors[:,2])
huemin = 1.414*min(amin, bmin)
huemax = 1.414*max(amax, bmax)
#Earlier values: u between -0.1 and 0.1, v between -0.08 and 0.08
u = numpy.linspace(lmin, lmax, 200)#variance in L. some amount of this might be just outside the bounds of rgb and possibly oklab; whatever, it'll just get capped off
v = numpy.linspace(huemin, huemax, 200)#variance in a
points = numpy.zeros((len(u),len(v),3))#40,000 point matrix

#Each point on our grid of points will be defined as a multiple of orth1 plus a multiple of orth1; these are 3d vectors, so each of their multiples will be a 3d vector as well,which is the same as a point in 3d space.
#Each point in 3d space has its 3 components; we start by interpreting them as coordinates, but later we'll use them as the three components of a color in Lab, then convert that to RGB.
for uindex, unum in enumerate(u):
    for vindex, vnum in enumerate(v):
        points[uindex, vindex] = centroid + numpy.multiply(orth1, vnum) + numpy.multiply(orth2, unum)

RGBpoints = copy.deepcopy(points)
RGBpoints = LabtoRGB(RGBpoints)


#We can also plot the gradient we created. We'll let the user opt in.
if input("Plot suggested gradient alongside colors from image? (y/n) ") == "y":
    for pindex, row in enumerate(points):
        #'points' has 40,000 fucking points in it; that will destroy matplotlib, so we want to grab a much smaller amount. Let's just grab every fifteenth value.
        if pindex % 15 == 0:
            for index, val in enumerate(row):
                #ax.scatter(points[row, index, 0], points[row, index, 1], points[row, index ,2], color=(RGBpoints[row, index, 0], RGBpoints[row, index, 1], RGBpoints[row, index, 2]))
                if index % 15 == 0:
                    ax.scatter(val[0], val[1], val[2], color=(RGBpoints[pindex, index, 0]/255, RGBpoints[pindex, index, 1]/255, RGBpoints[pindex, index, 2]/255))

RGBpoints = RGBpoints.astype(numpy.uint8)
gradient = Image.fromarray(RGBpoints)
gradient.save(fname+" suggested gradient.png")
print('Gradient created.')
#pyplot.show()

#Let's also plot our colors within the HSL color space- OKLab is great and all, but the trends shown in OKLab are a bit hard to translate into the HSL that your art software will be using.
hslcolors = RGBtoHSL(colors)
#If you're using a color *bar*, this is a pretty direct translation. If you're using a color *wheel*, its a bit less intuitive. But we can maybe graph that too?
fig2 = pyplot.figure()
ax2 = fig2.add_subplot(projection='3d')
#ax2.set(xlabel="Red", ylabel="Green", zlabel="Blue", title="Image Colors in RGB")#no idea if this will match, we'll return to this later
#Art software usually displays HSL with L being vertical, S being horizontal, and H being a third dimension- usually the hue slider.
ax2.set(xlabel="Saturation", ylabel="Hue", zlabel="Lightness", title="Image Colors in HSL")
for i in range(xpoints*ypoints):
    ax2.scatter(hslcolors[i,1], hslcolors[i,0], hslcolors[i,2], color=(colors[0, i, 0], colors[0, i, 1], colors[0, i, 2]))
#We can also add a box that more clearly shows what the HSL color space looks like

pyplot.show()