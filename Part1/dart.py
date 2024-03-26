import numpy as np 
import cv2
import os
import sys
import argparse
import csv

parser = argparse.ArgumentParser(description="dart board detection")
parser.add_argument('-name', '-n', type = str, default = './Dartboard')
args = parser.parse_args()

cascade_name = "./Dartboardcascade/cascade.xml"

def detect(frame):
    # 1. Convert image to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    dartboards = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(20,20), maxSize=(300,300))
    # 3. Draw box around dartboards found
    detectedBoards = []
    for i in range(0, len(dartboards)):
        # print("No. ",i ,": ",dartboards[i])
        start_point = (dartboards[i][0], dartboards[i][1])
        end_point = (dartboards[i][0] + dartboards[i][2], dartboards[i][1] + dartboards[i][3])
        detectedBoards.append([start_point, end_point])
    return detectedBoards

def display(frame, detectedBoards, colour):
    #Draw box around dartboards found
    for i in range(0, len(detectedBoards)):
        start_point, end_point = detectedBoards[i]
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

def readGroundtruth(frame, frame_path, filename='ground_truth.txt'):
    # read bounding boxes as ground truth
    ground_truths = []
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            if img_name in frame_path:
                x = int(content_list[1])
                y = int(content_list[2])
                width = int(content_list[3])
                height = int(content_list[4])
                start_point = (x, y)
                end_point = (x+width, y+height)
                ground_truths.append([img_name, start_point, end_point])

    return ground_truths

def calcPerformance(truths, boards):
    # ious = []
    ious = np.zeros(len(truths))
    for i in range(len(boards)):
        board = boards[i]
        # Find out which square is lower and more right
        board = np.array(board)
        board_width, board_height = board[1] - board[0]
        board_area = board_width * board_height
        for j in range(len(truths)):
            truth = truths[j]
            truth = np.array(truth)
            iou = 0
            truth_width, truth_height = truth[1] - truth[0]
            if ((board[0][0]) < (truth[0][0] + truth_width)) and ((board[0][0]+board_width) > truth[0][0]) and ((board[0][1]) < (truth[0][1] + truth_height)) and ((board[0][1]+board_height) > truth[0][1]):
                truth_area = truth_width * truth_height
                y_intersect_0 = max(board[0][1], truth[0][1])
                y_intersect_1 = min(board[0][1] + board_height, truth[0][1] + truth_height)
                x_intersect_0 = max(board[0][0], truth[0][0])
                x_intersect_1 = min(board[0][0] + board_width, truth[0][0] + truth_width)
                crossover_area = (abs(x_intersect_1 - x_intersect_0)) * (abs(y_intersect_1 - y_intersect_0))
                total_area = board_area + truth_area - crossover_area
                iou = crossover_area / total_area
                if iou > ious[j]:
                    ious[j] = iou
    return(ious.tolist())

#get scores takes ground truths and detected boxes and returns TPR and F1-score
def getScores(relevant_truths, joint_results):
    tp = len(relevant_truths)
    fp = len(joint_results) - tp  
    tpr = len([x for x in ious if x >0.5]) / (tp+ 0.00000000000000001)
    precision = tp / (tp + fp + 0.000000000000000001)
    f1_score = 2 * (precision * tpr) / (precision + tpr + 0.00000000000000001)
    return tpr, f1_score

# 1. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier(cascade_name)
if not model.load(cascade_name): # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    print('--(!)Error loading cascade model')
    exit(0)

#2. Read and classify 

imageName = args.name

#define colors of ground truth and detection bounding boxes
colour_detected = (0,255,0)
colour_truth = (0,0,255)

#find viola jones detections for all images
if imageName == './Dartboard':
    #iterate through directory
    for root, dirs, files in os.walk(imageName):
        #set up csv headings
        csv_file_path = "./data/data_all_images.csv"
        headings = ["File path", "F1 score", "TPR"]
        #sort files
        files.sort()
        files.sort(key = len)
        f1_list = []
        tpr_list = []
        with open(csv_file_path, 'w', newline='') as csv_file:
            # Create a csv.writer object
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headings)
            #iterate through files
            for file in files:
                file_path = os.path.join(root,file)
                #read in image
                frame = cv2.imread(file_path, 1)
                #detect viola jones frames
                boards = detect(frame)
                #draw on viola jones frames
                display(frame, boards, colour_detected)
                #get the relevant ground truths
                truth = readGroundtruth(frame, file_path)
                relevant_truths = []
                for i in range(len(truth)):
                    if (truth[i][0]+".jpg") in file_path:
                        relevant_truths.append(truth[i][1:])
                #draw on the relevant ground truths
                display(frame, relevant_truths, colour_truth)
                #find intersection over union for each box
                ious = calcPerformance(relevant_truths, boards)

                tpr, f1_score = getScores(relevant_truths, boards)
                tpr_list.append(tpr)
                f1_list.append(f1_score)
                row = [file,f1_score, tpr]
                csv_writer.writerow(row)
                cv2.imwrite("./viola_detections/detected_"+file, frame)
            tpr_mean = np.mean(tpr_list)
            f1_mean = np.mean(f1_list)
            averages = ["AVERAGES", f1_mean, tpr_mean]
            csv_writer.writerow(averages)
#logic for one image
else:
    #prepare csv headings
    csv_file_path = "./data/data_all_images.csv"
    headings = ["File path", "F1 score", "TPR"]
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a csv.writer object
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headings)
        frame = cv2.imread(imageName, 1)
        boards = detect(frame)
        display(frame, boards, colour_detected)
        truth = readGroundtruth(frame, imageName)
        relevant_truths = []
        for i in range(len(truth)):
            if truth[i][0] in imageName:
                relevant_truths.append(truth[i][1:])
        display(frame, relevant_truths, colour_truth)
        calcPerformance(relevant_truths, boards)
        ious = calcPerformance(relevant_truths, boards)
        tpr, f1_score = getScores(relevant_truths, boards)
        row = [imageName,f1_score, tpr]
        csv_writer.writerow(row)
        cv2.imwrite("./viola_detected.jpg", frame)
