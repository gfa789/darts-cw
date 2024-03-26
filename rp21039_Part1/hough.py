import numpy as np
import cv2
import os
import sys
import argparse
import dart
import csv

parser = argparse.ArgumentParser(description='Convert RGB to GRAY')
parser.add_argument('-name', '-n', type = str, default = './Dartboard')
args = parser.parse_args()

#take an input image and kernel
#return blurred image
def convolution(input, kernel):
	# intialise the output using the input
    blurredOutput = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
	# we need to create a padded version of the input
	# or there will be border effects
    kernelRadiusX = round(( kernel.shape[0] - 1 ) / 2)
    kernelRadiusY = round(( kernel.shape[1] - 1 ) / 2)
    # SET KERNEL VALUES - replace Gaussian with average blur
	# kernel[:,:] = 1/(size*size)
    paddedInput = cv2.copyMakeBorder(input, 
    	kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, 
    	cv2.BORDER_REPLICATE)

    for i in range(0, input.shape[0]):	
    	for j in range(0, input.shape[1]):
            patch = paddedInput[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            sum = (np.multiply(patch, kernel)).sum()
            blurredOutput[i, j] = sum
    return blurredOutput

#take input image
#return gradient magnitude and direction 
def sobel(input):
    dx_kernel = np.array([[-1, 0, 1],
					  [-3, 0, 3],
					  [-1, 0, 1]], dtype=np.float64)
    norm_factor = np.sqrt(np.sum(dx_kernel**2))
    dx_kernel /= norm_factor
    dy_kernel = dx_kernel.T
    dx = convolution(input, dx_kernel)
    dy = convolution(input, dy_kernel)
    mag = np.sqrt(np.square(dx) + np.square(dy))
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    dir = np.arctan(dy / (dx + 0.000000001))
    dir = cv2.normalize(dir, None, 0, 255, cv2.NORM_MINMAX)
    return mag, dir

#take gradient magnitude image
#return thresholded gradient magnitude image
def thresholdMagImage(mag):
    height, width = mag.shape
    thres_mag = np.zeros((height,width), dtype=np.uint8)
    #threshold magnitude image
    for i in range(height):
        row = [255 if x > 130 else 0 for x in mag[i]]
        thres_mag[i] = row
    return thres_mag

#take input image, thresholded magnitude image, and viola jones boxes
#return found dartboard centers, and image of lines displayed on image.
def hough_line(input, thres_mag, vj_boxes):
    
    height, width = thres_mag.shape
    #theta list excluding vertical and horizontal thetas
    th_0 = np.arange(-88, 2, 0.25)
    th_1 = np.arange(-2, 88, 0.25)
    th_2 = np.concatenate((th_0, th_1))
    # thetas = np.deg2rad(np.arange(-90, 90, 1))
    thetas = np.deg2rad(th_2)
    #define rhos to iterate through
    max_rho = int(np.ceil(np.sqrt(height**2 + width**2)))
    rhos = np.linspace(-max_rho, max_rho, max_rho * 2)

    hough_image = np.zeros((len(rhos), len(thetas)))
    #cos and sin all thetas beforehand to quicken processing
    coses = np.cos(thetas)
    sines = np.sin(thetas)
    #find ids of all nonzero points in threshold image
    y_idxs, x_idxs = np.nonzero(thres_mag)

    #iterate through all points and if in a viola-jones box search for a line
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        condition3 = True
        for vj_box in vj_boxes:
            #condition 3 to stop program looking for same line twice
            #may happen if there is an overlapping viola jones box
            if condition3:
                vj_x0, vj_y0 = vj_box[0]
                vj_w, vj_h = vj_box[1]
                vj_x1 = vj_x0 + vj_w
                vj_y1 = vj_y0 + vj_h 
                #condition1 and condition2 = is point in viola jones box?
                condition1 = x >= vj_x0 and x <= vj_x1
                condition2 = y >= vj_y0 and y <= vj_y1
            
                if condition1 and condition2:
                    for t_idx in range(len(thetas)):
                        rho = int(round(x * coses[t_idx] + y * sines[t_idx]) + max_rho)
                        hough_image[rho, t_idx] += 1
                    condition3 = False
    #normalize hough image   
    hough_image = cv2.normalize(hough_image, None, 0, 255, cv2.NORM_MINMAX)

    #get rhos and thetas of hough where detections > 130
    idxs = np.argwhere(hough_image > 130)
    rho_values = rhos[idxs[:,0]]
    theta_values = thetas[idxs[:,1]]
    lines = [(rho,theta) for rho, theta in zip(rho_values, theta_values)]

    intersects = np.zeros(input.shape)

    #find intersects by iterating through every line pair
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]
            #find angle between lines 
            angle_between_degrees = abs(np.degrees(np.arccos(np.cos(theta1 - theta2))))
            #check if they intersect at around 18 degrees
            #if they don't intersect in this range it is likely not dartboard
            if theta1 != theta2 and 16 <= angle_between_degrees <=20:
                A1, B1, C1 = np.cos(theta1), np.sin(theta1), rho1
                A2, B2, C2 = np.cos(theta2), np.sin(theta2), rho2

                # calculate determinant
                det = A1 * B2 - A2 * B1
                # calculate intersection point using Cramer's rule
                x = int(np.round((C1 * B2 - C2 * B1) / det))
                y = int(np.round((A1 * C2 - A2 * C1) / det))
                in_vj = False
                # cramer's rule doesn't take into account line length
                # only increase accumulator if intersection is in a Viola-Jone's box
                for vj_box in vj_boxes:
                    if not in_vj:
                        vj_x0, vj_y0 = vj_box[0]
                        vj_w, vj_h = vj_box[1]
                        vj_x1 = vj_x0 + vj_w
                        vj_y1 = vj_y0 + vj_h 
                        if (0 <= x < intersects.shape[1] and 0 <= y < intersects.shape[0]):
                            if ((x >= vj_x0 and x <= vj_x1)) and ((y >= vj_y0 and y <= vj_y1)):
                                in_vj = True
                            if ((x>= vj_x0 and x <= vj_x1)) and ((y >= vj_y0 and y <= vj_y1)):
                                in_vj = True
                if in_vj:
                    intersects[y][x] += 1
    
    #copy original image ready for drawing
    #this section draws detected lines on image
    #useful for analysis but not essential
    output_image = np.copy(image)
    if hough_image is not None:
        for line in lines:
            rho, theta= line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # x1, y1, x2, y2 are the endpoints of the line
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(output_image, (x1,y1), (x2,y2), (255,0,0), 1)

    detected = []
    #find 50 most detected intersections
    for i in range(50):
        found_already = False
        #gets index of most found point
        max_index_flat = np.argmax(intersects)
        if max_index_flat > 0:
            #convert this index back to array indices
            max_index_2d = np.unravel_index(max_index_flat, mag.shape)
            point = np.array([max_index_2d[1], max_index_2d[0]])
            #ensure points are good distance apart to avoid double detections
            if detected is not None:
                for d in detected:
                    if np.linalg.norm(point - d) < 150:
                        found_already = True
            #draw circle if not found already
            if not found_already:
                # draw center on output image
                cv2.circle(output_image, (point[0],point[1]), 5, (0,0,255), -1)
                detected.append(point)
            #set to zero so next iteration the second highest detections is now the highest
            intersects[max_index_2d[0],max_index_2d[1]] = 0
    return detected, output_image, hough_image

#cropMag takes magnitude image and turns every pixel not in viola-jones box to 0
#not used in final version so commented out
# def cropMag(vj_boxes, mag):
#     cropped_mag = np.zeros(mag.shape)
#     for i in range(mag.shape[0]):
#         for j in range(mag.shape[1]):
#             condition3 = False
#             for vj_box in vj_boxes:
#                 x = j
#                 y = i
#                 vj_x0, vj_y0 = vj_box[0]
#                 vj_w, vj_h = vj_box[1]
#                 vj_x1 = vj_x0 + vj_w
#                 vj_y1 = vj_y0 + vj_h 
                
#                 condition1 = x >= vj_x0 and x <= vj_x1
#                 condition2 = y >= vj_y0 and y <= vj_y1
            
#                 if condition1 and condition2 and not condition3:
#                     condition3 = True
#             if condition3:
#                 cropped_mag[i][j] = mag[i][j]
#     return cropped_mag
                
#joinResults takes the Viola Jones boxes and our detected points as arguments
#returns boxes that have been associated with one of our detected points
def joinResults(vj_boxes, detected):

    true_boxes = []
    #for each detected point, find biggest viola jones box nearby
    for d in detected:
        closest_distance = 100000
        closest_index = None
        closest_area = 0
        for v in range(len(vj_boxes)):
            vj_x0, vj_y0 = vj_boxes[v][0]
            vj_x1, vj_y1 = vj_boxes[v][1]
            width = abs(vj_x1 - vj_x0)
            height = abs(vj_y1 - vj_y0)

            x,y = d
            
            condition1 = x >= vj_x0 and x <= vj_x1
            condition2 = y >= vj_y0 and y <= vj_y1
            #are viola jones box less than 50 pixels away?
            condition3 = abs(x - vj_x0) < 50 or abs(x - vj_x1) < 50
            condition4 = abs(y - vj_y0) < 50 or abs(y - vj_y1) < 50
            #if they are outside vj box by less than 50 pixels then recentre vj box to our point
            if condition3 and condition4:
                if not condition1:
                    vj_x0 = x - width//2
                    vj_x1 = x + width//2
                if not condition2: 
                    vj_y0 = y - height//2
                    vj_y1 = y + height//2
                vj_boxes[v] = [(vj_x0, vj_y0), (vj_x1, vj_y1)]
        
            if (condition1 and condition2) or (condition3 and condition4):
                #find distance between center of vj box and our point
                vj_mid_x = (vj_x0 + vj_x1) //2
                vj_mid_y = (vj_y0 + vj_y1) // 2
                distance = np.sqrt((vj_mid_x - x)**2 + (vj_mid_y - y)**2)
                #find area of vj box
                area = abs(vj_x1 - vj_x0) * abs(vj_y1 - vj_y0)
                #if area significantly bigger than current area...
                if area - closest_area > 20:
                    # ... and if distance significantly closer...
                    if distance < closest_distance + 20:
                        #then take this box as the best box
                        closest_distance = distance
                        closest_index = v
                        closest_area = area
        if closest_index is not None:
            true_boxes.append(vj_boxes[closest_index])        
    return true_boxes

#get scores takes ground truths and detected boxes and returns TPR and F1-score
def getScores(relevant_truths, joint_results):
    tp = len(relevant_truths)
    fp = len(joint_results) - tp  
    tpr = len([x for x in ious if x >0.5]) / (tp+ 0.00000000000000001)
    precision = tp / (tp + fp + 0.000000000000000001)
    f1_score = 2 * (precision * tpr) / (precision + tpr + 0.00000000000000001)
    return tpr, f1_score

#incBoxSize takes a list of bounding boxes and expands them by 20% 
#returns the list of expanded boxes
def incBoxSize(boxes):
    for i in range(len(boxes)):
        b = boxes[i]
        x0, y0 = b[0]
        x1, y1 = b[1]
        width = abs(x1 - x0)
        height = abs(y1 - y0) 
        x0 = int(np.round(x0 - width *0.2))
        x1 = int(np.round(x1 + width *0.2))
        y0 = int(np.round(y0 - width *0.2))
        y1 = int(np.round(y1 + width *0.2))
        boxes[i] = [(x0,y0),(x1,y1)]
    return boxes

#Take image name from arguments - default does every image

imageName = args.name
kernelsize = 3

if kernelsize%2==0:
	print('kernel size must be odd number!')
	sys.exit(1)

#Create gaussian kernel of size k 

kernel = np.ones((kernelsize, kernelsize)) / kernelsize**2
gaussian_kernel = np.zeros((kernelsize,kernelsize))

sum_gauss = 0 
for i in range(kernelsize):
    for j in range(kernelsize):
        x = -(kernelsize//2) + i 
        y = -(kernelsize//2) + j 
        ga = np.exp(-(x**2 + y**2) / 2)
        gaussian_kernel[i][j] = ga
        sum_gauss += ga
gaussian_kernel = gaussian_kernel / sum_gauss

#define ground truth and detection colors
colour_detected = (0,255,0)
colour_truth = (0,0,255)

# if ./Dartboard then run on every dartboard
# outputs detected_"filename.jpg"

if imageName == './Dartboard':
    # iterate through directory
    for root, dirs, files in os.walk(imageName):
        #prepare csv file
        csv_file_path = "./data/data_hough_vj.csv"
        headings = ["File path", "F1 score", "TPR"]
        #sort files 
        files.sort()
        files.sort(key = len)
        #lists to store f1 scores and tpr 
        f1_list = []
        tpr_list = []
        with open(csv_file_path, 'w', newline='') as csv_file:
            # Create a csv.writer object
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headings)
            #iterate through files
            for file in files:
                file_path = os.path.join(root,file)
                image = cv2.imread(file_path, 1)

                #convert image to grayscale
                if image.shape[2] >= 3:
                    gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
                    gray_image = gray_image.astype(np.float32)
                else:
                    gray_image = image.astype(np.float32)

                #normalize image
                blurred_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                #apply gaussian blur 
                blurred_image = convolution(blurred_image, gaussian_kernel)
                #do sobel filter on processed image
                mag, dire = sobel(blurred_image)
                #get viola jones boxes from dart.py
                viola_jones = dart.detect(image)
                #run hough line - get back detected dartboard centers
                thres = thresholdMagImage(mag)
                detected,output,hough= hough_line(gray_image, thres, viola_jones)
                
                cv2.imwrite("./lines/lines_"+file, output)
                cv2.imwrite("./hough/hough_"+file, hough)
                cv2.imwrite("./threshold/threshold_"+file, thres)

                #get groundtruths as in dart.py
                truth = dart.readGroundtruth(image, file_path)
                relevant_truths = []
                #filter to the relevant ground truths
                for i in range(len(truth)):
                    if (truth[i][0]+".jpg") in file_path:
                        relevant_truths.append(truth[i][1:])
                #display relevant ground truths
                dart.display(image, relevant_truths, colour_truth)
                
                #join results by pairing centers with viola jones box
                joint_results = joinResults(viola_jones, detected)

                #increase bounding box size
                joint_results = incBoxSize(joint_results)

                #draw on detected bounding box
                dart.display(image, joint_results, colour_detected)

                #save image with boxes
                cv2.imwrite("./detected/detected_"+file, image)

                #find ious as in dart.py
                ious = dart.calcPerformance(relevant_truths, joint_results)

                tpr, f1_score = getScores(relevant_truths, joint_results)
                tpr_list.append(tpr)
                f1_list.append(f1_score)
                #insert scores into list
                row =[file, f1_score, tpr]
                csv_writer.writerow(row)
            #mean scores and write in averages
            tpr_mean = np.mean(tpr_list)
            f1_mean = np.mean(f1_list)
            averages = ["AVERAGES", f1_mean, tpr_mean]
            csv_writer.writerow(averages)
#for single file processing            
else:
    #prepare csv writer
    csv_file_path = "./data/data_hough_vj.csv"
    headings = ["File path", "F1 score", "TPR"]
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a csv.writer object
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headings)
        #read in image
        image = cv2.imread(imageName, 1)
        #convert image to grey scale
        if image.shape[2] >= 3:
            gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
            gray_image = gray_image.astype(np.float32)
        else:
            gray_image = image.astype(np.float32)
        #normalize image
        blurred_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
        #apply gaussian blur 
        blurred_image = convolution(blurred_image, gaussian_kernel)
        #do sobel filter on processed image
        mag, dire = sobel(blurred_image)
        #get viola jones boxes from dart.py
        viola_jones = dart.detect(image)
        #run hough line - get back detected dartboard centers
        thres = thresholdMagImage(mag)
        
        detected,output,hough= hough_line(gray_image, thres, viola_jones)
        #save lines drawn on and hough space            
        cv2.imwrite("./lines_drawn.jpg", output)
        cv2.imwrite("./hough_space.jpg", hough)
        cv2.imwrite("./threshold_image.jpg", thres)
        #get groundtruths as in dart.py
        truth = dart.readGroundtruth(image, imageName)
        relevant_truths = []
        #filter to the relevant ground truths
        for i in range(len(truth)):
            if (truth[i][0]+".jpg") in imageName:
                relevant_truths.append(truth[i][1:])
        #display relevant ground truths
        dart.display(image, relevant_truths, colour_truth)
        #join results by pairing centers with viola jones box
        joint_results = joinResults(viola_jones, detected)

        #increase bounding box size
        joint_results = incBoxSize(joint_results)

        #draw on detected bounding box
        dart.display(image, joint_results, colour_detected)

        #save image with boxes
        cv2.imwrite("./detected.jpg", image)

        #find ious as in dart.py
        ious = dart.calcPerformance(relevant_truths, joint_results)

        tpr, f1_score = getScores(relevant_truths, joint_results)
        
        # print(f'Tpr: {tpr}, f1_score: {f1_score}')
        csv_writer.writerow([imageName,f1_score,tpr])
     





