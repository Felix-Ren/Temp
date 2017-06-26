## Author: Michael Sheely
## June 24, 2015
## Modified by Fei Ren

import numpy as np
from numpy import dot
from numpy import average as avg
from numpy import subtract as sub
import time; import cv2; import math; import csv; import sys; import os

# use the following to run the code profiler
# python -m cProfile camera.py


# writer.writerow(['frame_number','center_x','center_y'])

# All vectors are assumed to be two dimensional
    
HCELLS = 3 # number of horizontal cells on a tag
VCELLS = 3 # number of vertical cells on a tag

HEIGHT = 500  # doesn't affect camera coverage. affect resolution and window size
WIDTH = 500
testbedLength = 327.66 # in cm (the one used at gym)
testbedWidth = 327.66 # in cm
HORIZ_OFFSET = 14.8 # in cm.
# VERT_OFFSET = 77 # 73
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
    # TIME_LIMIT = 1000000 # 480
else:
    path = "F:\Gym_experiment\\"  # "KiwiRuns2017\\"  # "KiwiRunsWinter2017\\"  # kiwi is the kind of robot which can move to all directions
    # TIME_LIMIT = 180  # 10 #

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
    return filter(lambda c: isTag(c, img), contours) #the second 'c' refers to elements in contours

def isTag(c, img):
    # determines if the image is a tag, based on its area and intensity
    MIN_SIZE = 30
    MAX_SIZE = 350

    boxVertices = cv2.cv.BoxPoints(cv2.minAreaRect(c))
    boxArea = math.sqrt((boxVertices[0][0]-boxVertices[1][0])**2 + (boxVertices[0][1]-boxVertices[1][1])**2) * \
        math.sqrt((boxVertices[0][0]-boxVertices[3][0])**2 + (boxVertices[0][1]-boxVertices[3][1])**2)
    if cv2.contourArea(c) > 0:
        boundingRatio = boxArea / cv2.contourArea(c)
    else:
        boundingRatio = 2
        
#    # test measurement
#    if boundingRatio < 1.15:
#        print cv2.contourArea(c)
        
    return (MIN_SIZE < cv2.contourArea(c) < MAX_SIZE) and goodAspectRatio(c, img) and boundingRatio < 1.15

def goodAspectRatio(c, img):
    _, (width, height), _ = cv2.minAreaRect(c)
    aspectRatio = max([width/height, height/width])
    #M = cv2.moments(c)
    #center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    #cv2.putText(img, str(aspectRatio), center, cv2.FONT_HERSHEY_SIMPLEX, .45, BLUE,2)
    # print aspectRatio
    return 1.4 < aspectRatio < 3.5  # 2.6

def averageValue(img):
    height, width = img.shape[:2]
    val = avg(img.sum(axis=0).sum(axis=0))
    return val/(height*width)

def drawTags(tagList, img):
    cv2.drawContours(img, tagList, -1, GREEN, 2) #draw all contours in red, with thickness 2

def drawSignalLight(lightList, img):
    cv2.drawContours(img, lightList, -1, BLUE, 1) #draw all contours in blue, with thickness 2

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
        print 'Could not identify tag orientation'
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
        #print "Unable to identify tag"
        index = binaryDigitsToDecimalString(matrix[:,1][::-1]) + binaryDigitsToDecimalString(matrix[:,2][::-1])
    return index

def integerize(point):
    return (int(point[0]),int(point[1]))

# # check if the contour is the signal light
# def isSignalLight(c):
    # # determines if the image is a tag, based on its area and aspect ratio
    # MIN_SIZE = 80  #need to reset this
    # MAX_SIZE = 200  #need to reset this
    
# #    boxVertices = cv2.cv.BoxPoints(cv2.minAreaRect(c))
# #    boxArea = math.sqrt((boxVertices[0][0]-boxVertices[1][0])**2 + (boxVertices[0][1]-boxVertices[1][1])**2) * \
# #        math.sqrt((boxVertices[0][0]-boxVertices[3][0])**2 + (boxVertices[0][1]-boxVertices[3][1])**2)

    # _, (width, height), _ = cv2.minAreaRect(c)
    # if width == 0 or height == 0:
        # return False
    # else:        
        # aspectRatio = max([width/height, height/width])
        # return (MIN_SIZE < cv2.contourArea(c) < MAX_SIZE) and (1 <= aspectRatio < 1.3)  # may need to reset aspectRatio bounds


# check the state of the robot (either flying or hovering) using color, shape and size
def checkState(frame, colorLowerBd, colorUpperBd):
    hsv_color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_color, colorLowerBd, colorUpperBd)
    
    # for debug and demo purpose
    cv2.imshow("blue mask", mask)  # white dots are signal of interest
    
#    testbed_top_left_x = 150# enter correct value here
#    testbed_top_left_y = 135# enter correct value here
#    testbed_bottom_right_x = frame.shape[1] - 65 # enter correct value here
#    testbed_bottom_right_y = frame.shape[0] - 40 # enter correct value here
#    frame = mask[testbed_top_left_y:testbed_bottom_right_y, testbed_top_left_x:testbed_bottom_right_x]

    # print np.sum((mask>0))
    # if np.sum((frame>0)) > 50: # mic drop
    if np.sum((mask>0)) > 16:  #260: #211: # mic drop
        # print 'blue light detected'
        return 'F',mask
    else:
        # print 'no blue'
        return 'H',mask

# check if the robot transitions from one step to another (red light)
# @param frame is the original frame after the perspective transformation
# @param hLowBd is a 1 x 2 array. It stores two lower bounds of the hue ranges.
# @param hUppBd is a 1 x 2 array. It stores two upper bounds of the hue ranges.
# @param sUppBd is a 1 x 1 array. It stores the lower bound of the saturation range.
# @param vLowBd is a 1 x 1 array. It stores the lower bound of the value range.
# @returns true if this frame corresponds to the end of a step or the beginning of the first step
# @returns false if this frame corresponds to the middle of a step
def checkTransition(frame, hLowBd, hUppBd, sLowBd, sUppBd, vLowBd, vUppBd):
    hsv_color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert the frame from BGR to HSV color space
	
	# manually convert the frame to a binary frame according to the HSV ranges.
	# A pixel gets value 1 if its color falls in all the ranges; it gets value 0 otherwise.
    h_bool = ((hLowBd[0] <= hsv_color[:, :, 0]) & (hsv_color[:, :, 0] <= hUppBd[0])) | (
              (hLowBd[1] <= hsv_color[:, :, 0]) & (hsv_color[:, :, 0] <= hUppBd[1]))
    s_bool = (sLowBd[0] <= hsv_color[:, :, 1]) & (hsv_color[:, :, 1] <= sUppBd[0])
    mask = np.zeros((hsv_color.shape[0], hsv_color.shape[1]))
    mask[(h_bool & s_bool)] = 1

    # debug purpose. test
    cv2.imshow("red mask", mask)  # show the window, which is named "red mask," of the masked frame.

    # print np.sum((mask>0))
    if np.sum((mask > 0)) > 82:  # mic drop. Need to adjust/change. Set the value to be the average of the minimum value when the red light is on and the maximum value when the red light is off.
        return True, mask
    else:
        return False, mask
		
# def checkTransition(frame, colorLowerBd, colorUpperBd):
    # hsv_color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert the frame from BGR to HSV color space
	
    # mask = cv2.inRange(hsv_color, colorLowerBd, colorUpperBd)

    # # debug purpose. test
    # cv2.imshow("red mask", mask)  # show the window, which is named "red mask," of the masked frame.

    # # print np.sum((mask>0))
    # if np.sum((mask > 0)) > 82:  # mic drop. Need to adjust/change. Set the value to be the average of the minimum value when the red light is on and the maximum value when the red light is off.
        # return True, mask
    # else:
        # return False, mask

#### MAIN ####
def main():

    # turn on the camera
    cam0 = cv2.VideoCapture(1)  # VideoCapture is a class defined in opencv.
    # make sure to sure 1 instead of 0 as input for laptop
    # cam0.set(cv.CAP_PROP_FRAME_HEIGHT, 1000)

    #calibrate for a continuous shot at the height of the tags
    calibrationPoints0 = np.float32([[72,3],[83,478],[585,480],[595,0]]) # order: from top-left corner, CCW
    corners = np.float32([[0,0],[0,WIDTH],[HEIGHT,WIDTH],[HEIGHT,0]])  # dimension of the window
      
    transform0 = cv2.getPerspectiveTransform(calibrationPoints0,corners) # transform coordinate from the original to a rectangle with 0,0 to the top-left corner.
    
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
        out = cv2.VideoWriter(path+"trial" + str(i) + '.avi', -1, 27.0, (WIDTH, HEIGHT)) # open video file for writing.
       # out = cv2.VideoWriter(path+"trial" + str(i) + '.avi', -1, 27.0, (WIDTH, 2*HEIGHT)) # open video file for writing.
        # param1: file name; param2: codec of the file, such as mpeg ; param3: fps; param4: size

    # cv2.namedWindow('processedFrame', cv2.WINDOW_NORMAL)
    lastX = None
    recordedX = None
    lastY = None
    state = None
    stepEnd = False  # true if the row of data correponds to the end/beginning of a step
    startTime = time.time()

    while(cam0.isOpened()):
        # Capture frame-by-frame
        ret0, frame0 = cam0.read() # read the first frame. first return value is a bool, says if successfully capture a frame or not. second return is the frame.    

        if ret0:             
            # covert color to gray
            thresh0 = threshold(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY,260)) # 260 - number of channels
            
#            #check state of the robot in two separate frames
#            state0 = checkState(thresh0, frame0)


            # # test. debug
            # cv2.imshow("original", frame0)



            tagList = findAprilTags(thresh0, frame0)
            drawTags(tagList,frame0)

            # record the position of the robot
            for tag in tagList:
                M = cv2.moments(tag)
                center = cv2.perspectiveTransform(np.float32([[(M['m10']/M['m00']), (M['m01']/M['m00'])]])[None,:,:],transform0) # transform the tag shape to the one conform to the rectangular window??
                lastY = center[0][0][0] # y coordinate of the robot
                lastX = center[0][0][1] # x coordinate of the robot
                
            if CALIBRATION_MODE:
                cv2.circle(frame0, (72,3), 2, (255,0,0),-1)  # top-left corner
                cv2.circle(frame0, (83,478), 2, (255,0,0),-1)
                cv2.circle(frame0, (585,480), 2, (255,0,0),-1)
                cv2.circle(frame0, (595,0), 2, (255,0,0),-1)

                # for point in calibrationPoints0.astype(int):
                #     cv2.circle(frame0, tuple(point.tolist()), 2, (255,0,0),-1)
            else:
                frame0 = cv2.warpPerspective(frame0, transform0, (WIDTH,HEIGHT)) # transform the frame
            cv2.imshow('calibrated window', frame0)  # use this to fix position of the testbed on the ground

            # check if the blue signal light is on.
            stateLightColorLowBd = np.array([100, 75, 0])
            stateLightColorUppBd = np.array([120, 255, 255])
            state, mask1 = checkState(frame0, stateLightColorLowBd, stateLightColorUppBd)

            # check if there is transition of steps from one step to another
            # range: h [0,7] U [169,179]; s: [95,230]; v: [0,255]  can adjust s and v
            transitionHueLowBd = np.array([0, 169])  
            transitionHueUppBd = np.array([7, 179])
            transitionSatLowBd = np.array([95])
            transitionSatUppBd = np.array([230])
            transitionValLowBd = np.array([0])
            transitionValUppBd = np.array([255])
            stepEnd, mask2 = checkTransition(frame0, transitionHueLowBd, transitionHueUppBd,
                                             transitionSatLowBd, transitionSatUppBd,
                                             transitionValLowBd, transitionValUppBd)
            # the beginning of the first step is also considered as "step end"
			
			## if floor is still captured as red light, use this set of ranges and function			
			# transLightColorLowBd = np.array([169, 95, 0])
            # transLightColorUppBd = np.array([179, 230, 255])
            # stepEnd, mask2 = checkTransition(frame0, transLightColorLowBd, transLightColorUppBd)


            # write frame(video) to file and write position of the tag to .csv
            if RECORD:
                out.write(frame0)
                now = time.time()
                if lastX != None: # is it possible that this value == None during the recording?

                    if recordedX != lastX:  # in case tag detection signal is off/discontinuous
                        # shift the origin from the top left corner to bottom left corner.
						# (x-axis is the side parallel to the closest wall)
                        lastY = WIDTH - lastY       						
                        
                        # convert unit of lastX and lastY to centimeter.
                        lastX = testbedWidth * lastX / HEIGHT
                        lastY = testbedLength * lastY / WIDTH
						
						# adjust for the extra white margin in the x-axis
                        lastX = lastX - HORIZ_OFFSET

                    
                    recordedX = lastX
                    
                    # test
                    print "lastX: ", lastX
                    print "lastY: ", lastY
                    print "state: ", state
                    print "time: ", now-startTime
                    print "step transitions now: ", stepEnd


                    writer.writerow([lastX, lastY, state, now - startTime, stepEnd])
                    # state: either flying ('F') or hovering ('H') 
                    # now - startTime recording the current timestamp of the robot (in second) with the beginning of the program t = 0s.
                else:
                    writer.writerow([None, None, state, now - startTime, stepEnd])

        # Automatically stop the process when the length of time exceeds the time limit
        now = time.time()
        # if now - startTime > TIME_LIMIT:
        #     break
        if cv2.waitKey(1) == ord('q'):
            break
        
    # When everything is done, release the capture

    # turn off the cameras
    cam0.release()
    if RECORD:
        out.release()
        f.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
