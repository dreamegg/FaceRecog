import numpy as np
import cv2 as cv
import os
import argparse
import time
import logging
import logging.handlers
from datetime import datetime

from matplotlib import pyplot as plt
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from knn import KNN
Q
parser = argparse.ArgumentParser(description='Face Recognition using LinearSVM')
parser.add_argument('--LBPsize', default= 8*4)
parser.add_argument('--FDfile', default='../Trained/haarcascade_frontalface_default.xml')
parser.add_argument('--sf', default= 1.001)
parser.add_argument('--k', default= 8)
parser.add_argument('--C', default= 1)
parser.add_argument('--classifier', default="NN", help='SVM/NN')
parser.add_argument('--dataset', default="FACE", help='FACE/CASIA')

opts = parser.parse_args()


logger = logging.getLogger('mylogger')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
fileHandler = logging.FileHandler('../logs/{:%Y%m%d_%H%M%S}.log'.format(datetime.now()))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(fomatter)
streamHandler.setFormatter(fomatter)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

def LBP (img, cell_h, cell_w) :
	size_h, size_w= img.shape
	cell_h_size = size_h // cell_h
	cell_w_size = size_w // cell_w
	out = np.zeros(shape=(cell_h-2, cell_w-2))
	#out = np.dtype(int)

	for x in range(1,cell_w-1) :
		for y in range(1,cell_h-1) :
			onecell = np.zeros(shape=(3,3)).astype(float)
			startx = x * cell_w_size
			starty = y * cell_h_size
			onecell[0, 0] = np.average(img[starty-cell_h_size: starty, startx-cell_w_size : startx])
			onecell[0, 1] = np.average(img[starty-cell_h_size: starty, startx: startx + cell_w_size])
			onecell[0, 2] = np.average(img[starty-cell_h_size: starty, startx + cell_w_size: startx + cell_w_size*2])
			onecell[1, 0] = np.average(img[starty: starty + cell_h_size, startx-cell_w_size : startx])
			onecell[1, 1] = np.average(img[starty: starty + cell_h_size, startx: startx + cell_w_size])
			onecell[1, 2] = np.average(img[starty: starty + cell_h_size, startx + cell_w_size: startx + cell_w_size*2])
			onecell[2, 0] = np.average(img[starty + cell_h_size: starty + cell_h_size*2, startx-cell_w_size : startx])
			onecell[2, 1] = np.average(img[starty + cell_h_size: starty + cell_h_size*2, startx: startx + cell_w_size])
			onecell[2, 2] = np.average(img[starty + cell_h_size: starty + cell_h_size*2, startx + cell_w_size: startx + cell_w_size*2])

			lbp_code = 0
			if (onecell[1, 0] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 7
			if (onecell[2, 0] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 6
			if (onecell[2, 1] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 5
			if (onecell[2, 2] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 4
			if (onecell[1, 2] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 3
			if (onecell[0, 2] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 2
			if (onecell[0, 1] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 1
			if (onecell[0, 0] > onecell[1, 1]): lbp_code = lbp_code + 2 ^ 1
			out[y-1,x-1] = lbp_code

			logger.debug(onecell)
	logger.debug(out)
	return out

def ReadPreprocessData (rootPath) :
	logger.info("=====>[Read Train Data and Preprocessing]")
	#image_list = []
	#label_list = []
	img_id_list = []

	file_count = 0

	for root, dirs, files in os.walk(rootPath):
		for fname in files:
			file_count = file_count+1
			full_fname = os.path.join(root, fname)
			# print (full_fname)

			img = cv.imread(full_fname)
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

			faces = face_cascade.detectMultiScale(gray, scaleFactor = opts.sf, 5)
			for (x, y, w, h) in faces:
				cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
				face_array = gray[y:y + h, x:x + w]
				face_rewsized = cv.resize(face_array, (128, 128))
				normalizedImg = cv.normalize(face_rewsized, face_rewsized, 0, 255, cv.NORM_MINMAX)
				#image_list.append(normalizedImg)
				#label_list.append(full_fname.split("\\")[-2])
				img_id_list.append((normalizedImg,full_fname.split("\\")[-2]))

				#roi_gray = gray[y:y + h, x:x + w]
				#roi_color = img[y:y + h, x:x + w]
				#eyes = eye_cascade.detectMultiScale(roi_gray)
				#for (ex, ey, ew, eh) in eyes:
				#	cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

	logger.info ("find {} face in {} files ({}%)".format(len(img_id_list),file_count,len(img_id_list)*100//file_count))
	return img_id_list

face_cascade = cv.CascadeClassifier(opts.FDfile)
eye_cascade = cv.CascadeClassifier('../Trained/haarcascade_eye.xml')

start_time = time.time()

outLBP_list= []
outHisto_list= []

logger.info(opts)
logger.info("=====>[Prepare Train Data]")

#=======Make Face List==========
if opts.dataset == 'FACE' :  face_list = ReadPreprocessData('../data/Face_Database/Train')
if opts.dataset == 'CASIA' :
	all_list = ReadPreprocessData('../data/CASIA-WebFace')
	face_list ,testImage_List = train_test_split(all_list,test_size=0.1)

logger.info("=====>[Extract Feature]")
for i in range(len(face_list)) :
	out = LBP(face_list[i][0], opts.LBPsize,opts.LBPsize)
	hist, bin_edges = np.histogram(out, bins=255)

	outLBP_list.append(out)
	outHisto_list.append(hist)

if (opts.classifier == 'SVM') :
	logger.info("=====>[Train SVM]")
	model = LinearSVC(C=opts.C, random_state=42)
elif (opts.classifier == 'NN') :
	logger.info("=====>[Train NN]")
	model = KNN(opts.k)

model.fit(outHisto_list,[x[1] for x in face_list])

#plt.subplot(221), plt.imshow(face_list[1], 'gray'), plt.title('Origon')
#plt.subplot(222), plt.imshow(cv.resize(outLBP_list[1], (128, 128), interpolation=cv.INTER_NEAREST), 'gray'), plt.title('LBP')
#plt.subplot(223), plt.bar(bin_edges[:-1], outHisto_list[1], width=1)
#plt.xlim(min(bin_edges), max(bin_edges))
#plt.show()


logger.info("=====>[Prepare Test Data]")
if opts.dataset == 'FACE' :  testImage_List =  ReadPreprocessData('../data/Face_Database/Test')

#plt.subplot(221), plt.imshow(testImage_List[1], 'gray'), plt.title('Origon')
#plt.subplot(222), plt.imshow(cv.resize(outLBP_list[1], (128, 128), interpolation=cv.INTER_NEAREST), 'gray'), plt.title('LBP')
#plt.subplot(223), plt.bar(bin_edges[:-1], outHisto_list[1], width=1)
#plt.xlim(min(bin_edges), max(bin_edges))
#plt.show()

testHisto_list=[]
for i in range(len(testImage_List)) :
	out = LBP(testImage_List[i][0], opts.LBPsize,opts.LBPsize)
	hist, bin_edges = np.histogram(out, bins=255)

	testHisto_list.append(hist)

logger.info("=====>[prediction using SVM]")
prediction = model.predict(testHisto_list)

correct =0
for i in range(len(testImage_List)) :
	# display the image and the prediction
	#cv.putText(testImage_List[i], prediction[i]+"  GT:"+test_label_list[i], (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	#cv.imshow("Image", testImage_List[i])
	#cv.waitKey(0)
	if testImage_List[i][1] == prediction[i] :correct=correct+1
	else : logger.debug ("Fail!")
	logger.debug("test count [{}] : Predict =>".format(i)+prediction[i] + "    GT : " +testImage_List[i][1])

logger.info ("Test Result ={} /  {} ({}%)".format(correct, len(testImage_List),correct*100 // len(testImage_List)))
