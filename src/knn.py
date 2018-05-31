import math
import operator

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((x,dist))
	distances.sort(key=operator.itemgetter(1))
	return distances[:k]


def getResponse(neighbors,label):
	classVotes = {}
	for x in range(len(neighbors)):
		index = neighbors[x][0]
		response = label[index]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes, key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0]

class KNN :
	def __init__(self, K):
		self.k = K

	def fit(self, X, y, sample_weight=None):
		self.tainlist = X
		self.tain_labellist = y
		return 0

	def predict(self, testlist):
		predictions = []
		for x in range(len(testlist)):
			neighbors = getNeighbors(self.tainlist, testlist[x], self.k)
			result = getResponse(neighbors, self.tain_labellist)
			predictions.append(result)
		return predictions
