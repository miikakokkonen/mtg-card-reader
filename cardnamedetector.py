import multiprocessing as mp
import pytesseract
import cv2
import sys
import time
import requests
import numpy as np
from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi

class FeedProcess(mp.Process):
	def __init__(self, fQueue,iQueue):
		mp.Process.__init__(self)
		self.fQueue = fQueue
		self.iQueue = iQueue
		self.feed = cv2.VideoCapture('http://192.168.1.63:8080/video')
		requests.get('http://192.168.1.63:8080/settings/quality?set=100')
		requests.get('http://192.168.1.63:8080/settings/exposure?set=0')
		requests.get('http://192.168.1.63:8080/nofocus')
		self.run()

	def run(self):
		while True:
			if not self.iQueue.empty():
				command = self.iQueue.get()
				getattr(self, command)()
			ret, frame = self.feed.read()
			if ret:
				self.fQueue.put(frame)
				

class CardDetector(QDialog):
	
	def __init__(self):
		super().__init__()
		loadUi('carddetector.ui',self)

		self.buttonlist = [self.readButton, self.setBackgroundButton, self.autoFocusButton, self.lockFocusButton, self.exportButton]
		for button in self.buttonlist:
			button.clicked.connect(lambda x: getattr(self,self.sender().objectName()+'Clicked')())
		self.threshold = 0
		self.thresholdBox.valueChanged.connect(lambda x: setattr(self,'threshold',x))
		self.historyListWidget.itemDoubleClicked.connect(lambda x:(self.collectedListWidget.addItem(x.clone()),self.historyListWidget.clear()))
		self.collectedListWidget.itemDoubleClicked.connect(lambda x: [item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable) for item in self.collectedListWidget.selectedItems()])
		self.frameQueue = mp.Queue(maxsize=1)
		self.inputQueue = mp.Queue()
		self.feedProcess = mp.Process(target = FeedProcess, args =(self.frameQueue,self.inputQueue), daemon = True).start()
		self.background = None
		self.timer = QTimer(self)
		self.timer.setTimerType(Qt.PreciseTimer)
		self.timer.timeout.connect(self.updateFrame)
		self.timer.start(5)
		self.autosave = QTimer(self)
		self.autosave.setTimerType(Qt.PreciseTimer)
		self.autosave.timeout.connect(self.autosaver)
		self.autosave.start(30000)
	
	def autosaver(self):
		with open('as.txt','a') as f:
			backup = '\n\nBackup\n\n'
			rows = self.collectedListWidget.count()
			for i in range(rows):
				backup += ('1 x '+self.collectedListWidget.item(i).text()+'\n')
			f.write(backup)

	def keyPressEvent(self,e):
		if e.key() == Qt.Key_Delete:
			try:
				[self.collectedListWidget.takeItem(self.collectedListWidget.row(item)) for item in self.collectedListWidget.selectedItems()]
			except:
				print('nothing to delete')

	def exportButtonClicked(self):
		timen = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
		fname = 'output_'+ str(timen)+'.txt'
		data = []
		rows = self.collectedListWidget.count()
		for i in range(rows):
			data.append('1x '+self.collectedListWidget.item(i).text()+'\n')
		with open(fname,'w') as f:
			for line in data:
				f.write(line)
	def autoFocusButtonClicked(self):
		requests.get('http://192.168.1.63:8080/nofocus')

	def lockFocusButtonClicked(self):
		requests.get('http://192.168.1.63:8080/focus')

	def readButtonClicked(self):
		try:
			frame = self.frame.copy()
			frame2 = frame.copy()
			card_contour = self.find_card_borders(frame)
			warped = self.warp(card_contour,frame)
			y,x = warped.shape[0],warped.shape[1]
			#yy1,yy2,xx1,xx2 = int(y/25),int(y/7),int(x/11),int(x*5/7)
			yy1,yy2,xx1,xx2 = int(y/30),int(y/7),int(x/13),int(x*5/7)
			roi = warped[yy1:yy2,xx1:xx2].copy()

			#y_roi, x_roi = roi.shape[0],roi.shape[1]
			#color = roi[int(y_roi/2),x_roi-1]
			#rroi = cv2.inRange(roi, color-20, color+20)
			#droi = cv2.morphologyEx(rroi,cv2.MORPH_DILATE,(3,3))
			#_, ncontours, _= cv2.findContours(droi,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			#sncontours = sorted([ (cv2.contourArea(i), i) for i in ncontours ], key=lambda a:a[0], reverse=True)
			#_, name_contour = sncontours[0]
			#x,y,w,h = cv2.boundingRect(name_contour)
			#namebox = cv2.cvtColor(roi[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
			#cv2.imshow('before', namebox)
			#namebox2 = self.contrast(namebox,0.8)
			#cv2.imshow('namebox', namebox)
			#cv2.imshow('namebox2', namebox2)
			cv2.imshow('roi', roi)
			teksti = pytesseract.image_to_string(roi)
			if len(teksti)>=3:
				teksti.replace('\n','')
				teksti.replace('\r','')

				teksti.replace('’',"'")
				while teksti[0] in ["'",'"','_','—','.','-',' ','\n','/','(',')','´','`','‘','|','[',']','<','>','\r']:
					if len(teksti)==0:
						break
					teksti = teksti[1:]
				while teksti[-1] in ["'",'"','_','—','.','-',' ','\n','/','(',')','´','`','‘','|','[',']','<','>','\r']:
					if len(teksti)==0:
						break
					teksti = teksti[:-2]
				print(teksti)
				if len(teksti)>2:
					self.historyListWidget.addItem(str(teksti))
				else:
					pass
			print('done')
		except AttributeError:
			print('need bg')

	def deleteButtonClicked(self):
		print('deleted')

	def setBackgroundButtonClicked(self):
		self.background = self.frame.copy()
		self.gbackground = cv2.cvtColor(self.background,cv2.COLOR_BGR2GRAY)

		cv2.imshow('bg',self.background)

	def updateFrame(self):
		if not self.frameQueue.empty():
			self.frame = self.frameQueue.get()	
			try:
				self.find_card_borders(self.frame)
			except:
				pass
			cv2.imshow('frame', self.frame)
	

	def find_card_borders(self,frame):
		gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		difference = cv2.subtract(self.gbackground, gframe)
		_,tresh = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
		_, contours, _= cv2.findContours(tresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		scontours = sorted([ (cv2.contourArea(i), i) for i in contours ], key=lambda a:a[0], reverse=True)
		_,card_contour = scontours[0]
		cv2.drawContours(frame, [card_contour], -1, (0, 255, 0), 2)
		return card_contour


	def contrast(self,image,gamma):
		invGamma = 1/gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
    		for i in np.arange(0, 256)]).astype("uint8")
		image2 = cv2.LUT(image, table)
		return image2

	def warp(self,card_contour,frame):
		rect = cv2.minAreaRect(card_contour)
		points = cv2.boxPoints(rect)
		box = np.int0(points)

		rect = np.zeros((4, 2), dtype = "float32")

		# get top left and bottom right points
		s = box.sum(axis = 1)
		rect[0] = box[np.argmin(s)]
		rect[2] = box[np.argmax(s)]

		# get top right and bottom left points
		diff = np.diff(box, axis = 1)
		rect[1] = box[np.argmin(diff)]
		rect[3] = box[np.argmax(diff)]

		(tl, tr, br, bl) = rect

		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))

		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		dst = np.array([
		    [0, 0],
		    [maxWidth - 1, 0],
		    [maxWidth - 1, maxHeight - 1],
		    [0, maxHeight - 1]], dtype = "float32")

		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
		return warped

if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = CardDetector()
	window.setWindowTitle('card detector')
	window.show()
	sys.exit(app.exec_())