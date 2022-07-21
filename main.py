from concurrent.futures import thread
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import cv2
import pytesseract
from mss import mss
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.interpolate import make_interp_spline, BSpline

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func: {f.__name__} took: {te - ts} sec")

        return result
    return wrap

#Processes the given image
def processImage(image):
    darkened = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[...,2]*0.8
    _, thresholded = cv2.threshold(darkened, 130, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    eroded = cv2.erode(opening, np.ones((2,2), np.uint8))

    return cv2.cvtColor(np.uint8(eroded), cv2.COLOR_BGR2RGB)

#Processes the text to remove stray characters
def processText(input):
    matches = re.findall(r'[-+]?\d*\.?\d+|\d+', input)
    return ''.join(matches)

#Performs OCR on the given image
def ocrImage(image, time=False):
    addedBorder = cv2.copyMakeBorder(image, 25, 25, 25, 25, cv2.BORDER_REPLICATE)
    ocrResult = pytesseract.image_to_string(addedBorder, "digits", config='--psm 6')

    if time:
        replacedStr = processText(ocrResult.replace(".", ""))

        if(len(replacedStr) == 5):
            print("Removed stray")
            replacedStr = replacedStr[:2] + replacedStr[3:]


        return replacedStr[:2] + ":" + replacedStr[2:]

    return processText(ocrResult)

#Extracts speed, altitude, and time from the image
def getTelemetry(image, bounds):
    speedCrop = image[bounds[0][0][1]:bounds[0][0][1]+(bounds[0][1][1] - bounds[0][0][1]), bounds[0][0][0]:bounds[0][0][0]+(bounds[0][1][0] - bounds[0][0][0])]
    altCrop = image[bounds[1][0][1]:bounds[1][0][1]+(bounds[1][1][1] - bounds[1][0][1]), bounds[1][0][0]:bounds[1][0][0]+(bounds[1][1][0] - bounds[1][0][0])]
    timeCrop = image[bounds[2][0][1]:bounds[2][0][1]+(bounds[2][1][1] - bounds[2][0][1]), bounds[2][0][0]:bounds[2][0][0]+(bounds[2][1][0] - bounds[2][0][0])]

    return [ocrImage(speedCrop), ocrImage(altCrop), ocrImage(timeCrop, True)]

#Annotates the output image with the telemetry
def annotateOutput(image, ocrResults):
    cv2.rectangle(image, speedBounds[0], speedBounds[1], (0, 255, 0), thickness=2) #Location of Speed - (1, 8), (104, 44)
    cv2.rectangle(image, altBounds[0], altBounds[1], (0, 255, 0), thickness=2) #Location of Altitude - (170, 8), (260, 44)
    cv2.rectangle(image, timeBounds[0], timeBounds[1], (0, 255, 0), thickness=2) #Location of Time - (778, 26), (980, 65)

    cv2.putText(image, ocrResults[0], (speedBounds[0][0] + 20, 95), 1, 2, (0, 0, 255), lineType=1, thickness=2)
    cv2.putText(image, ocrResults[1], (altBounds[0][0] + 20, 95), 1, 2, (0, 0, 255), lineType=1, thickness=2)
    cv2.putText(image, ocrResults[2], (timeBounds[0][0] + 20, 95), 1, 2, (0, 0, 255), lineType=1, thickness=2)

#Calculates acceleration from the given data
def getAcceleration(speeds, times, gap):
    if len(speeds) > gap and len(times) > gap:
        x1 = times[-gap]
        x2 = times[-1]

        y1 = speeds[-gap]
        y2 = speeds[-1]

        return abs(round(((y2 - y1) / (x2 - x1)) / 3.6, 1))
    return 0

#Initializes the plots
def getPlots():
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax1.set_ylim(0, 10000)
    ax1.set_xlim(0, 650)
    ax2.set_ylim(0, 200)
    ax3.set_ylim(0, 60)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocity (km/h)", color='g')
    ax2.set_ylabel("Altitude (km)", color='b')
    ax3.set_ylabel("Acceleration (m/s²)", color='r')

    ax3.spines['right'].set_position(("outward", -60))

    global accelText
    accelText = ax1.text(0.5, 0.95, "0", horizontalalignment='center', transform=ax1.transAxes)
            
    return [ax1, ax2, ax3]

#telemetry[speeds,alts,times,accelerations]
def updatePlot(telemetry, plot):
    #Velocity
    plot[0].plot(telemetry[2], telemetry[0], 'g-')

    #Altitude
    plot[1].plot(telemetry[2], telemetry[1], 'b-')

    #Acceleration
    # plot[2].plot(telemetry[2], telemetry[3], 'r-')

    accelText.set_text(f"Accl: {str(telemetry[3][-1])}m/s²")

    plt.draw()
    plt.pause(0.01)

#Extracts the time from the image
def getTime(timeStr):
    split = timeStr.split(":")

    minutes = int(split[0])
    seconds = int(split[1])

    return (minutes * 60) + seconds

#Path to tessdata directory
os.environ["TESSDATA_PREFIX"] = r"E:\Desktop\Programming\Python\F9Telem\new\tessdata"
bounding_box = {'top': 960, 'left': 110, 'width': 3010 - 2030, 'height': 1030 - 969}

speedBounds = [(1, 5), (104, 44)]
altBounds = [(170, 5), (260, 44)]
timeBounds = [(850, 24), (985, 70)]

sct = mss()



#Initialize plots and start timer
startTime = timer()
plots = getPlots()

#declaring telemetry variables
allSpeeds = []
allAlts = []
allTimes = []
allAccelerations = []

while True:
    try:
        screenshot = cv2.copyMakeBorder(np.array(sct.grab(bounding_box)), 0, 5, 0, 5, cv2.BORDER_CONSTANT)

        processed = processImage(screenshot)
        ocr = getTelemetry(processed, [speedBounds, altBounds, timeBounds])

        outputImage = cv2.copyMakeBorder(screenshot, 0, 40, 0, 0, cv2.BORDER_CONSTANT)
        annotateOutput(outputImage, ocr)    
        cv2.imshow('screen', outputImage)

        ##############################
        parsedtelemetry = [float(ocr[0]), float(ocr[1]), float(timer() - startTime)]

        allSpeeds.append(parsedtelemetry[0])
        allAlts.append(parsedtelemetry[1])
        allTimes.append(parsedtelemetry[2])

        accl = getAcceleration(allSpeeds, allTimes, 5)

        allAccelerations.append(accl)

        updatePlot([allSpeeds, allAlts, allTimes, allAccelerations], plots)
        
        print(f"Speed: {ocr[0]} Alt: {ocr[1]} Time: {getTime(ocr[2])}s")
    except Exception as e:
        print(f"An exception occurred: {e}")
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break