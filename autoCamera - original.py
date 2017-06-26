## Author: Michael Sheely
## June 24, 2015
## To be run on MUFASA-PC with two cameras connected via FireWire
## Modified by Henry Ehrhard

# review note: not all functions defined used
#              Detect signal light: cv2.findContours(..), identify(..)

import numpy as np
from numpy import dot
from numpy import average as avg
from numpy import subtract as sub
import time; import cv2; import math; import csv; import sys; import os

# use the following to run the code profiler
# python -m cProfile camera.py


#writer.writerow(['frame_number','center_x','center_y'])

#All vectors are assumed to be two dimensional

HCELLS = 3 #number of horizontal cells on a tag
VCELLS = 3 #number of vertical cells on a tag

HEIGHT = 480
WIDTH = 640
HORIZ_OFFSET = 7 # 9
VERT_OFFSET = 77 # 73
RED,GREEN,BLUE,YELLOW,DARKRED = ([0,0,255],[0,255,0],[255,0,0],[0,255,255],[0,0,170])
PRINTING = True
DISPLAY_TAGS = True
Robots = {}
CALIBRATION_MODE = False
RECORD = True
TANK = False # the tank robot can only move in one-dimention without turn

# initialize file storage path
if TANK:
    path = "TankRuns\\"
    TIME_LIMIT = 480
else:
    path = "KiwiRuns\\RowsTestbed\\"  # kiwi is the kind of robot which can move to all directions
    TIME_LIMIT = 180


def update_HORIZ(x):
    global HORIZ_OFFSET
    HORIZ_OFFSET = x

def update_VERT(x):
    global VERT_OFFSET
    VERT_OFFSET = x

#cv2.cv.CreateTrackbar('H Offset', 'combFrame', HORIZ_OFFSET, 20, update_HORIZ)
#cv2.cv.CreateTrackbar('V Offset', 'combFrame', VERT_OFFSET, 100, update_VERT)


class RobotData:
    def __init__(self, center):
        self.center = center
        #self.orientation = orientation
        self.updated = True

    # center is an (x,y) tuple, orientation is an angle in degrees measured from
    # the positive x axis, frame is a number which designates which frame the robot
    # is in, and updated is a boolean which tells if that particular robot has been updated

    def __repr__(self):
        return ("Robot at " + str(integerize(self.center)) + ".")

    def update(self, center):
        self.center = center
        #self.orientation = angle
        self.updated = True

    def reset(self):
        self.updated = False

def writeData(frameNum):
    for robotIndex in Robots:
        robot = Robots[robotIndex]
        #writer.writerow([robotIndex, robot.center[0], robot.center[1], robot.orientation])
        writer.writerow([frameNum, robot.center[0], robot.center[1]])

def threshold(src, value=120):
    ret, thresh = cv2.threshold(src, value, 255, cv2.THRESH_BINARY)
    return thresh

#finds all contours and keeps only those with area between 50 and 1000 pixels
def findAprilTags(threshed, img):
    contours, hierarchy = cv2.findContours(threshed, 1, cv2.CHAIN_APPROX_SIMPLE) # contours are stored as vectors of points
    return filter(lambda c: isTag(c, img), contours)

def isTag(c, img):
    #determines if the image is a tag, based on its area and intensity
    MIN_SIZE = 700
    MAX_SIZE = 2000
    #MIN_VAL = 300
    #MAX_VAL = 700
    
    boxVertices = cv2.cv.BoxPoints(cv2.minAreaRect(c))
    boxArea = math.sqrt((boxVertices[0][0]-boxVertices[1][0])**2 + (boxVertices[0][1]-boxVertices[1][1])**2) * \
        math.sqrt((boxVertices[0][0]-boxVertices[3][0])**2 + (boxVertices[0][1]-boxVertices[3][1])**2)
    if cv2.contourArea(c) > 0:
        boundingRatio = boxArea / cv2.contourArea(c)
    else:
        boundingRatio = 2

    return (MIN_SIZE < cv2.contourArea(c) < MAX_SIZE) and goodAspectRatio(c, img) and boundingRatio < 1.15

def goodAspectRatio(c, img):
    _, (width, height), _ = cv2.minAreaRect(c)
    aspectRatio = max([width/height, height/width])
    #M = cv2.moments(c)
    #center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    #cv2.putText(img, str(aspectRatio), center, cv2.FONT_HERSHEY_SIMPLEX, .45, BLUE,2)
    return 1 < aspectRatio < 2

def averageValue(img):
    height, width = img.shape[:2]
    val = avg(img.sum(axis=0).sum(axis=0))
    return val/(height*width)

def drawTags(tagList, img):
    cv2.drawContours(img, tagList, -1, DARKRED, 2) #draw all contours in red, with thickness 2

def drawCircle(point, img, color):
    cv2.circle(img, integerize(point), 2, color, 2)

#marks each robot with its index on the given image
#def drawRobots(img):
#    ARROW_LENGTH = 22
#    for index in Robots:
#        center = integerize(Robots[index].center)
#        angle = Robots[index].orientation
#        cv2.putText(img, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, .7, (255,51,51),2)
#        p2 = integerize((center[0]+ARROW_LENGTH*math.cos(angle),center[1]+ARROW_LENGTH*math.sin(angle)))
#        cv2.line(img, center, p2, (255,255,0), 2, 2)

def updateDict(tagList, img, thresh):
    global Robots
    tagViews = []
    for tag in tagList:
        rect = cv2.minAreaRect(tag)
        tagImg = getTagImg(tag, rect, img)
        #idMatrix = identify(tagImg)
        #if idMatrix == None: #we could not calculate the intensity of the cells, so the image was not a valid tag
        #    continue
        tagViews.append(tagImg)
        #index = matrixToIndex(idMatrix)
        #angle = calculateAngle(tag, rect, idMatrix)
        M = cv2.moments(tag)
        center = (M['m10']/M['m00']), (M['m01']/M['m00'])
        Robots[0] = RobotData(center)
    Robots = {key: rob for key, rob in Robots.items() if rob.updated} #remove any robots from our list that were not updated
    for r in Robots.values(): #reset all robots to their 'nonupdated' status for the next iteration
        r.reset()
    return tagViews

def getTagImg(tag, rect, img):
    #extracts the image of the tag from the main image, and rotates it appropriately
    bottom, left, top, right = cv2.cv.BoxPoints(rect)
    try:
        if dist(left, top) < dist(left, bottom):
            posSlope = False
            theta = math.atan((left[1] - bottom[1])/(left[0] - bottom[0]))
        else:
            posSlope = True
            theta = math.atan((right[1] - bottom[1])/(right[0] - bottom[0]))
    except ZeroDivisionError:
        theta = math.atan(float('inf')) #slope is pi/2
    theta
    height = dist(right, bottom)
    width = dist(right, top)
    if posSlope:
        width, height = height, width
    fcenter = rect[0][0], rect[0][1]
    return subimage(img, fcenter, theta, int(width), int(height))

#Developed from code by user rroowwllaanndd of stack overflow
#http://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
def subimage(image, centre, theta, width, height):
    output_image = cv2.cv.fromarray(np.zeros((height, width,3), np.uint8))
    image = cv2.cv.fromarray(image)
    mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                        [np.sin(theta), np.cos(theta), centre[1]]])
    map_matrix_cv = cv2.cv.fromarray(mapping)
    cv2.cv.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
    return np.asarray(output_image)

def identify(img):
    XBUF = 4 #pixels of buffer zone
    YBUF = 4
    matrix = np.zeros((VCELLS,HCELLS), dtype=bool)

    threshed = threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200)
    contours, hierarchy = cv2.findContours(threshed, 1, 2)
    largest = largestContour(contours)
    try:
        x,y,w,h = cv2.boundingRect(largest)
    except cv2.error:
        return
    dx = int((w - 2*XBUF)/float(HCELLS))
    dy = int((h - 2*YBUF)/float(VCELLS))
    for i in range(HCELLS):
        for j in range(VCELLS):
            filled = isFilled(threshed, (x+XBUF+i*dx, y+YBUF+j*dy), (x+XBUF+(i+1)*dx, y+YBUF+(j+1)*dy), dx*dy, img)
            if filled != None:
                matrix[j,i] = filled
            else:
                return None
    return matrix

def largestContour(contourList):
    contour = None
    size = 0
    for current in contourList:
        currentArea = cv2.contourArea(current)
        if currentArea > size:
            contour = current
            size = currentArea
    return contour

def isFilled(img, p1, p2, area, defacing):
    DARKNESS_THRESHOLD = 0.7 #dark squares will have an intensity below this percentage
    p1, p2 = integerize(p1), integerize(p2)
    intensity = 0
    for x in range(p1[0], p2[0]):
        for y in range(p1[1], p2[1]):
            intensity += bool(img[y,x])
            if x in (p1[0],p2[0]-1) or y in (p1[1],p2[1]-1):
                defacing[y,x] = RED
    if area == 0:
        return None #this means that we are picking up some edge motion
    return (intensity/float((p2[1]-p1[1])*(p2[0]-p1[0]))) < DARKNESS_THRESHOLD

#calculates the cartesian distance between two points
def dist(p1, p2):
    return np.linalg.norm(sub(p1,p2))

def calculateAngle(tag, rect, idMatrix):
    bottom, left, top, right = cv2.cv.BoxPoints(rect)
    if dist(left, top) < dist(left, bottom):
        if left[0] == bottom[0]:
            theta = math.atan(float('inf')) #avoid division by zero
        else:
            theta = math.atan((left[1] - bottom[1])/(left[0] - bottom[0]))
    else:
        if right[0] == bottom[0]:
            theta = math.atan(float('inf')) #avoid division by zero
        else:
            theta = math.atan((right[1] - bottom[1])/(right[0] - bottom[0]))
    if idMatrix[0,0] and idMatrix[0,1] and idMatrix[0,2]: #top is dark
        return theta + math.pi
    elif idMatrix[2,0] and idMatrix[2,1] and idMatrix[2,2]: #bottom is dark
        return theta
    else:
        #print 'Could not identify tag orientation'
        return theta

#determines whether the point p is inside a rectangle with the given points
    #projection of top->(x,y) onto top->left must have length between 0 and top->left
    #projection of top->(x,y) onto top->right must have length between 0 and top->right
def within(p, top, left, right):
    return (0 < dot(sub(p,top),sub(left,top)) < dot(sub(left,top),sub(left,top)) and 
        0 < dot(sub(p,top),sub(right,top)) < dot(sub(right,top),sub(right,top)))

def binaryDigitsToDecimalString(L):
    return str(int(''.join([str(int(x)) for x in L]),2))

def matrixToIndex(matrix):
    if np.all(matrix[:,0]):
        index = binaryDigitsToDecimalString(matrix[:,2]) + binaryDigitsToDecimalString(matrix[:,1])
    elif np.all(matrix[:,2]):
        index = binaryDigitsToDecimalString(matrix[:,0][::-1]) + binaryDigitsToDecimalString(matrix[:,1][::-1])
    else:
        print "Unable to identify tag"
        index = binaryDigitsToDecimalString(matrix[:,1][::-1]) + binaryDigitsToDecimalString(matrix[:,2][::-1])
    return index

def integerize(point):
    return (int(point[0]),int(point[1]))

#### MAIN ####

def main():

    # turn on the camera
    cam0 = cv2.VideoCapture(1) # VideoCapture is a class defined in opencv.
    cam1 = cv2.VideoCapture(0) # 0 as input means open the default camera.
    
    #find transformations to line up frames
    
    #calibrate for a continuous shot of the floor
    #calibrationPoints0 = np.float32([[35,1],[35,424],[627,2],[630,422]])
    #calibrationPoints1 =  np.float32([[31,42],[27,472],[627,49],[625,470]])
    
    #calibrate for a continuous shot at the height of the tags
    calibrationPoints0 = np.float32([[58,57],[61,438],[580,54],[582,432]])
    calibrationPoints1 = np.float32([[54,40],[55,426],[578,43],[574,418]])
    corners = np.float32([[0,0],[0,HEIGHT],[WIDTH,0],[WIDTH,HEIGHT]]) 
    
    #error test calibration
    #calibrationPoints0 = np.float32([[105,19],[109,411],[622,15],[626,402]])
    #calibrationPoints1 =  np.float32([[105,14],[109,407],[623,15],[626,400]])
    
   
    transform0 = cv2.getPerspectiveTransform(calibrationPoints0,corners) # transform coordinate from the original to a rectangle with 0,0 to the top-left corner.
    transform1 = cv2.getPerspectiveTransform(calibrationPoints1,corners) # first param is the source coordinates, second is the destination coordinates
    
    # prepare to write in .csv and .avi files
    if RECORD:
        if len(sys.argv) > 1: # sys.argv stores the input user puts in the command line, default size of 1.
            f = open(sys.argv[1], 'wb') # w: writing mode; b: open as binary file
        else:
            i = 1
            while os.path.isfile(path + "tracking" + str(i) + ".csv"): # isfile check if a regular file exist with the given path
                i = i + 1
            f = open(path+"tracking"+str(i)+".csv", 'wb')
        writer = csv.writer(f)
        out = cv2.VideoWriter(path+"trial" + str(i) + '.avi', -1, 27.0, (WIDTH, 2*HEIGHT)) # open video file for writing.
        # param1: file name; param2: codec of the file, such as mpeg ; param3: fps; param4: size

    cv2.namedWindow('combFrame', cv2.WINDOW_NORMAL)
    lastX = None
    lastY = None
    startTime = time.time()

    while(cam0.isOpened() & cam1.isOpened()):
        
        
        # Capture frame-by-frame
        ret0, frame0 = cam0.read() # read the first frame. first return value is a bool, says if successfully capture a frame or not. second return is the frame.
        ret1, frame1 = cam1.read()
        
        if ret0 & ret1:
            
            if sum(abs(cv2.subtract(np.int32(frame0)[HEIGHT-1,:,0], np.int32(frame1)[0,:,0]))) > 50000: # image stored as a matrix?? Check for shift of focus during the recording
                cam0.release()
                cam1.release()
                cam0 = cv2.VideoCapture(1)
                cam1 = cv2.VideoCapture(0)
                ret0, frame0 = cam0.read()
                ret1, frame1 = cam1.read()
                print "cameras reset"
            
            # covert color to gray
            thresh0 = threshold(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY,260)) # 260 - number of channels
            thresh1 = threshold(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY,260))
 
            tagList = findAprilTags(thresh0, frame0)
            drawTags(tagList,frame0)
            
            # record the position of the robot
            for tag in tagList:
                M = cv2.moments(tag)
                center = cv2.perspectiveTransform(np.float32([[(M['m10']/M['m00']), (M['m01']/M['m00'])]])[None,:,:],transform0) # transform the tag shape to the one conform to the rectangular window??
                lastX = center[0][0][0] # x coordinate of the robot
                lastY = center[0][0][1]
                
            
            tagList = findAprilTags(thresh1, frame1)
            drawTags(tagList,frame1)
            for tag in tagList:
                M = cv2.moments(tag)
                center = cv2.perspectiveTransform(np.float32([[(M['m10']/M['m00']), (M['m01']/M['m00'])]])[None,:,:],transform1)
                lastX = center[0][0][0]
                lastY = center[0][0][1] + HEIGHT
            
            if CALIBRATION_MODE:
                for point in calibrationPoints0.astype(int):
                    cv2.circle(frame0, tuple(point.tolist()), 2, (255,0,0),-1)
                for point in calibrationPoints1.astype(int):
                    cv2.circle(frame1, tuple(point.tolist()), 2, (255,0,0),-1)        
            else:
                frame0 = cv2.warpPerspective(frame0, transform0, (WIDTH,HEIGHT)) # transform the frame
                frame1 = cv2.warpPerspective(frame1, transform1, (WIDTH,HEIGHT))
            
            combFrame = np.concatenate((frame0,frame1),axis=0)
                
            #drawTags(tagList,combFrame)q
            #drawRobots(combFrame)
            cv2.imshow('combFrame', combFrame) # display the frame
            
            # write frame(video) to file and write position of the tag to .csv
            if RECORD:
                out.write(combFrame)
                if lastX != None: # is it possible that this value == None during the recording?
                    writer.writerow([lastX, lastY])
        
        # Automatically stop the process when the length of time exceeds the time limit
        now = time.time()
        if now - startTime > TIME_LIMIT:
            break
        if cv2.waitKey(1) == ord('q'):
            break
        
    # When everything is done, release the capture

   #cv2.imwrite("background.jpg", combFrame)
    # turn off the cameras
    cam0.release()
    cam1.release()
    if RECORD:
        out.release()
        f.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
