##############
# Name:
# email:
# Date:

import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plot

def entropy(freqs):
	""" 
	entropy(p) = -SUM (Pi * log(Pi))
	>>> entropy([10.,10.])
	1.0
	>>> entropy([10.,0.])
	0
	>>> entropy([9.,3.])
	0.811278
	"""
	all_freq = sum(freqs)
	entropy = 0 
	for fq in freqs:
		prob = fq * 1.0 / all_freq
		if abs(prob) > 1e-8:
			entropy += -prob * np.log2(prob)
	return entropy
	
def infor_gain(before_split_freqs, after_split_freqs):
	"""
	gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
	>>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
	0.02922
	"""
	gain = entropy(before_split_freqs)
	overall_size = sum(before_split_freqs)
	for freq in after_split_freqs:
		ratio = sum(freq) * 1.0 / overall_size
		gain -= ratio * entropy(freq)
	return gain

#count those who survived and those who didnt
#send in the label file and then also the indexes you want to be checking
def countlabel(label, indexes):
	count0 = 0
	count1 = 0
	for x in indexes:
		if label[0][x] == 0:
			count0 += 1
		else:
			count1 += 1
	return [count0, count1]


def bestVar(data, attributes, label, indexes):
	splits = threshold(data, attributes, label, indexes)
	before = countlabel(label,indexes)	
	gain = []
	max = 0
	index = []
	for x in splits:
		gain.append([x[0],x[1], infor_gain(before,x[2]), x[3], x[4]])
	for x in gain:
		if x[2] > max:
			max = x[2]
			index = x
	return index




#go through the range of each attribute and get the label count based on that
def threshold(data, attributes, label,indexes):
	splits = []
	for x in attributes:
		max = data[x].max()
		min = data[x].min()
		for y in range(min, max):
			if y == max:
				pass
			else:
				lessIndex = []
				greaterIndex = []
				for i in indexes :
					if data[x][i] <= y:
						lessIndex.append(i)
					else:
						greaterIndex.append(i)
				#append name of attribute, where its being split at (<=,>), and then the frequencies
				#so I need to trim the data when i pass it to countlabel
				if len(lessIndex) == 0 or len(greaterIndex) == 0:
					pass
				else:
					lessCount = countlabel(label,lessIndex)
					greaterCount = countlabel(label, greaterIndex)
					splits.append([x,y,[lessCount,greaterCount], lessIndex, greaterIndex])

	return splits

def getAttributes(data):
	attributes = []
	for x in data:
		attributes.append(x)
	return attributes


#def predictValue(dataRow, leaves):
#	for leaf in leaves:
#		inleaf = 1
#		for thresh in leaf.path:
#			attribute = thresh[0]
#			threshval = thresh[1]
#			threshtrue = thresh[2]
#			if threshtrue == 1:
#				if dataRow[attribute] > threshval:
#					inleaf = 0
#					break
#			else:
#				if dataRow[attribute] <= threshval:
#					inleaf = 0
#					break
#		if inleaf == 1:	
#			return leaf.predictedLabel

	
class Node(object):
	def __init__(self):
		self.leaf = None
		self.predictedLabel = None
		self.attributeName = None
		self.threshold = None
		self.parent = None
		self.path = None
		self.right = None
		self.left = None

class Tree(object):
	def __init__(self):
		#self.numNodes = 0
		pass
		
	#def checkAccuracy(self, data,label, start, percent):
	#	totalcorrect = 0
	#	dataSize = int(len(data.values) * (percent/100))
	#	for x in range(start, start + dataSize):
	#		dataRow = data.iloc[x]
	#		predictedVal = predictValue(dataRow, self.leaves)
	#		actualVal = label[0][x]
	#		if predictedVal == actualVal :
	#			totalcorrect += 1
	#	return totalcorrect/dataSize

	def prune(self, head, data, label, start, percent):
		accuracy = Tree.checkAccuracy(self,head,data,label,start,percent)
		leaves = Tree.getLeaves(self, head)
		for x in leaves:
			n = x.parent
			n.leaf = 1
			changedAccuracy = Tree.checkAccuracy(self,head,data,label,start,percent)
			if changedAccuracy >= accuracy:
				#self.numNodes -= 1
				n.left = None
				n.right = None
				Tree.prune(self,head, data, label, start, percent)
				break
			else:
				n.leaf = 0


	def checkAccuracy(self, head, data, label, start, percent):
		totalcorrect = 0
		dataSize = int(len(data.values) * (percent/100))
		for x in range(start, start+dataSize):
			dataRow = data.iloc[x]
			actualVal = label[0][x]
			node = head
			while node.leaf == 0:
				attribute = node.attributeName
				threshold = node.threshold
				if dataRow[attribute] <= threshold:
					node = node.left
				else:
					node = node.right
			predictedVal = node.predictedLabel
			if actualVal == predictedVal:
				totalcorrect += 1
		return totalcorrect/dataSize

	def getLeaves(self, node):
		leaves = []

		if node.leaf == 0:
			left = Tree.getLeaves(self,node.left)
			right = Tree.getLeaves(self,node.right)
			leaves = left + right
		else:
			return [node]
		return leaves




	def ID3Prune(self, data, label, indexes, attributes, path, node):
		#ADD: Possibly send in attributes and prune out attribute that has already been divided
		count = countlabel(label,indexes)
		if count[0] == 0:
			n = Node()
			#self.numNodes += 1
			n.leaf = 1
			n.predictedLabel = 1
			n.parent = node
			n.path = path.copy()
			return n
		elif count[1] == 0:
			n = Node()
			#self.numNodes += 1
			n.leaf = 1
			n.predictedLabel = 0
			n.parent =node
			n.path = path.copy()
			return n
			#create new leaf node and return that node
		else:
			gain = bestVar(data, attributes, label, indexes)
			n = Node()
			#self.numNodes += 1
			n.parent = node 
			n.path = path.copy()
			if count[0] >= count[1]:
					n.predictedLabel = 0
			else:
					n.predictedLabel = 1
			if len(gain) == 0:
				n.leaf = 1
			else: 
				n.leaf = 0
				n.attributeName = gain[0]
				n.threshold = gain[1]
				leftThresh = [gain[0], gain[1], 1]
				rightThresh = [gain[0], gain[1], 0]
				path.append(leftThresh)
				n.left = Tree.ID3Prune(self, data,label,gain[3], attributes, path, n)
				path.remove(leftThresh)
				path.append(rightThresh)
				n.right = Tree.ID3Prune(self, data,label, gain[4], attributes, path, n)
				path.remove(rightThresh)
			return n

	def ID3Depth(self, data, label, indexes, attributes, path, maxDepth):
		pathLength = len(path)
		count = countlabel(label,indexes)
		if count[0] == 0:
			n = Node()
			#self.numNodes += 1
			n.leaf = 1
			n.predictedLabel = 1
			n.path = path.copy()
			return n
		elif count[1] == 0:
			n = Node()
			#self.numNodes += 1
			n.leaf = 1
			n.predictedLabel = 0
			n.path = path.copy()
			return n
			#create new leaf node and return that node
		else:
			gain = bestVar(data, attributes, label, indexes)
			n = Node()
			#self.numNodes += 1
			n.path = path.copy()
			if count[0] >= count[1]:
					n.predictedLabel = 0
			else:
					n.predictedLabel = 1
			if len(gain) == 0 or pathLength > maxDepth:
				n.leaf = 1
			else: 
				n.leaf = 0
				n.attributeName = gain[0]
				n.threshold = gain[1]
				leftThresh = [gain[0], gain[1], 1]
				rightThresh = [gain[0], gain[1], 0]
				path.append(leftThresh)
				n.left = Tree.ID3Depth(self, data,label,gain[3], attributes, path, maxDepth)
				path.remove(leftThresh)
				path.append(rightThresh)
				n.right = Tree.ID3Depth(self, data,label, gain[4], attributes, path, maxDepth)
				path.remove(rightThresh)
			return n

	def ID3Split(self, data, label, indexes, attributes, path, minSplit):
		#ADD: Possibly send in attributes and prune out attribute that has already been divided
		count = countlabel(label,indexes)
		size = count[0] + count[1]
		if count[0] == 0:
			n = Node()
			n.leaf = 1
			n.predictedLabel = 1
			n.path = path.copy()
			return n
		elif count[1] == 0:
			n = Node()
			n.leaf = 1
			n.predictedLabel = 0
			n.path = path.copy()
			return n
			#create new leaf node and return that node
		else:
			gain = bestVar(data, attributes, label, indexes)
			n = Node()
			n.path = path.copy()
			if count[0] >= count[1]:
					n.predictedLabel = 0
			else:
					n.predictedLabel = 1
			if len(gain) == 0 or size< minSplit: 
				n.leaf = 1
			else: 
				n.leaf = 0
				n.attributeName = gain[0]
				n.threshold = gain[1]
				leftThresh = [gain[0], gain[1], 1]
				rightThresh = [gain[0], gain[1], 0]
				path.append(leftThresh)
				n.left = Tree.ID3Split(self, data,label,gain[3], attributes, path, minSplit) 
				path.remove(leftThresh)
				path.append(rightThresh)
				n.right = Tree.ID3Split(self, data,label, gain[4], attributes, path, minSplit)
				path.remove(rightThresh)
			return n

	def ID3(self, data, label, indexes, attributes, path):
		#ADD: Possibly send in attributes and prune out attribute that has already been divided
		count = countlabel(label,indexes)
		if count[0] == 0:
			n = Node()
			#self.numNodes += 1
			n.leaf = 1
			n.predictedLabel = 1
			n.path = path.copy()
			return n
		elif count[1] == 0:
			n = Node()
			#self.numNodes += 1
			n.leaf = 1
			n.predictedLabel = 0
			n.path = path.copy()
			return n
			#create new leaf node and return that node
		else:
			gain = bestVar(data, attributes, label, indexes)
			n = Node()
			#self.numNodes += 1
			n.path = path.copy()
			if count[0] >= count[1]:
					n.predictedLabel = 0
			else:
					n.predictedLabel = 1
			if len(gain) == 0:
				n.leaf = 1
			else: 
				n.leaf = 0
				n.attributeName = gain[0]
				n.threshold = gain[1]
				leftThresh = [gain[0], gain[1], 1]
				rightThresh = [gain[0], gain[1], 0]
				path.append(leftThresh)
				n.left = Tree.ID3(self, data,label,gain[3], attributes, path)
				path.remove(leftThresh)
				path.append(rightThresh)
				n.right = Tree.ID3(self, data,label, gain[4], attributes, path)
				path.remove(rightThresh)
			return n


if __name__ == "__main__" :
	# parse arguments
	count = 0
	type = 0
	percent = 0
	valPercent = 0
	depth = 0
	minSplit = 0
	for x in sys.argv:
		print('arg: ', x)
		if count == 1:
			trainData = pd.read_csv(x + "/titanic-train.data", delimiter = ',', index_col=None, engine='python')
			trainLabel = pd.read_csv(x + "/titanic-train.label", delimiter = ',', index_col=None, engine='python', header=None)
		elif count == 2:
			testData = pd.read_csv(x + "/titanic-test.data", delimiter = ',', index_col=None, engine='python')
			testLabel = pd.read_csv(x + "/titanic-test.label", delimiter = ',', index_col=None, engine='python', header=None)
		elif count == 3:
			type = x
		elif count == 4:
			percent = int(x)
		elif count == 5:
			valPercent = int(x)
		elif count == 6:
			if type == ('depth'):
				depth = int(x)
			elif type == ('min_split'):
				minSplit = int(x)
		count += 1	
	# build decision tree
	t = Tree()
	trainLength = len(trainData.values)
	weightedLength = int(trainLength * (percent/100))
	trainAttributes = getAttributes(trainData)
	trainIndexes = [x for x in range(0,weightedLength)]
	path = []


	#for graphing purposes
	#graphTrainAccuracy = []
	#graphTestAccuracy = []
	#percentage = []
	#numberOfNodes = []
	##-------------40
	#weightedLength40 = int(trainLength * (40/100))
	#trainIndexes40 = [x for x in range(0,weightedLength40)]
	#head40 = t.ID3Prune(trainData, trainLabel, trainIndexes40, trainAttributes, path, None)
	#t.prune(head40, trainData, trainLabel, weightedLength40, valPercent)
	#percentage.append(40)
	#numberOfNodes.append(t.numNodes)
	#graphTrainAccuracy.append(t.checkAccuracy(head40,trainData, trainLabel, 0, 40))
	#graphTestAccuracy.append(t.checkAccuracy(head40,testData,testLabel,0,100))
	##-------------50
	#a = Tree()
	#weightedLength50 = int(trainLength * (50/100))
	#trainIndexes50 = [x for x in range(0,weightedLength50)]
	#head50 = a.ID3Prune(trainData, trainLabel, trainIndexes50, trainAttributes, path, None)
	#a.prune(head50, trainData, trainLabel, weightedLength50, valPercent)
	#percentage.append(50)
	#numberOfNodes.append(a.numNodes)
	#graphTrainAccuracy.append(a.checkAccuracy(head50,trainData, trainLabel, 0, 50))
	#graphTestAccuracy.append(a.checkAccuracy(head50,testData,testLabel,0,100))
	##-------------60
	#n = Tree()
	#weightedLength60 = int(trainLength * (60/100))
	#trainIndexes60 = [x for x in range(0,weightedLength60)]
	#head60 = n.ID3Prune(trainData, trainLabel, trainIndexes60, trainAttributes, path, None)
	#n.prune(head60, trainData, trainLabel, weightedLength60, valPercent)
	#percentage.append(60)
	#numberOfNodes.append(n.numNodes)
	#graphTrainAccuracy.append(n.checkAccuracy(head60,trainData, trainLabel, 0, 60))
	#graphTestAccuracy.append(n.checkAccuracy(head60,testData,testLabel,0,100))
	##-------------70
	#b = Tree()
	#weightedLength70 = int(trainLength * (70/100))
	#trainIndexes70 = [x for x in range(0,weightedLength70)]
	#head70 = b.ID3Prune(trainData, trainLabel, trainIndexes70, trainAttributes, path, None)
	#b.prune(head70, trainData, trainLabel, weightedLength70, valPercent)
	#percentage.append(70)
	#numberOfNodes.append(b.numNodes)
	#graphTrainAccuracy.append(b.checkAccuracy(head70,trainData, trainLabel, 0, 70))
	#graphTestAccuracy.append(b.checkAccuracy(head70,testData,testLabel,0,100))
	##-------------80
	#p = Tree()
	#weightedLength80 = int(trainLength * (80/100))
	#trainIndexes80 = [x for x in range(0,weightedLength80)]
	#head80 = p.ID3Prune(trainData, trainLabel, trainIndexes80, trainAttributes, path, None)
	#p.prune(head80, trainData, trainLabel, weightedLength80, valPercent)
	#percentage.append(80)
	#numberOfNodes.append(p.numNodes)
	#graphTrainAccuracy.append(p.checkAccuracy(head80,trainData, trainLabel, 0, 80))
	#graphTestAccuracy.append(p.checkAccuracy(head80,testData,testLabel,0,100))
	##--------------100
	#k = Tree()
	#weightedLength100 = int(trainLength * (100/100))
	#trainIndexes100 = [x for x in range(0,weightedLength100)]
	#head100 = k.ID3(trainData, trainLabel, trainIndexes100, trainAttributes, path)
	#percentage.append(100)
	#numberOfNodes.append(k.numNodes)
	#graphTrainAccuracy.append(t.checkAccuracy(head100,trainData, trainLabel, 0, 100))
	#graphTestAccuracy.append(t.checkAccuracy(head100,testData,testLabel,0,100))

	#plot.plot(percentage, graphTestAccuracy, label= 'Test Accuracy')
	#plot.plot(percentage, graphTrainAccuracy, label= 'Train Accuracy')

	#plot.plot(percentage, numberOfNodes)
	#plot.ylabel('Number of Nodes', fontsize='12')
	#plot.plot(percentage, [10,5,10,5,5])
	#plot.ylabel('Optimal Depth', fontsize='12')
	#plot.xlabel('Percentage of Data Set', fontsize='12')
	#plot.show()



	if type == 'vanilla':
		head = t.ID3(trainData,trainLabel, trainIndexes, trainAttributes, path)
	elif type == 'depth':
		head = t.ID3Depth(trainData, trainLabel, trainIndexes, trainAttributes, path, depth)
	elif type == 'min_split':
		head = t.ID3Split(trainData, trainLabel, trainIndexes, trainAttributes, path, minSplit)
	elif type == 'prune':
		head = t.ID3Prune(trainData, trainLabel, trainIndexes, trainAttributes, path, None)
		t.prune(head, trainData, trainLabel, weightedLength, valPercent)

	testLength = len(testData.values)
	testAttributes = getAttributes(testData)
	testIndexes = [x for x in range(0,testLength)]
	# predict on testing set & evaluate the testing accuracy
	trainAcurracy = t.checkAccuracy(head, trainData,trainLabel, 0, percent)
	print('Train Accuracy:', trainAcurracy)
	if type == 'depth' or type == 'min_split' or type == 'prune':
		validationAccuracy = t.checkAccuracy(head, trainData,trainLabel, weightedLength, valPercent)
		print('Validation Accuracy:', validationAccuracy)
	testAcurracy = t.checkAccuracy(head, testData,testLabel, 0 ,100)
	print('Test Accuracy:', testAcurracy)
