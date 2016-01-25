import numpy as np
import copy

def readCRFgraph(poseDataset,noise=1e-10,forecast_on_noisy_features=False):
	'''
	Understanding data structures
	nodeToEdgeConnections: node_type ---> [edge_types]
	nodeConnections: node_name ---> [node_names]
	nodeNames: node_name ---> node_type
	nodeList: node_type ---> output_dimension
	nodeFeatureLength: node_type ---> feature_dim_into_nodeRNN

	edgeList: list of edge types
	edgeFeatures: edge_type ---> feature_dim_into_edgeRNN
	'''
	
	filename = poseDataset.crf_file

	global nodeNames, nodeList, nodeToEdgeConnections, nodeConnections, nodeFeatureLength, edgeList, edgeFeatures

	lines = open(filename).readlines()
	nodeOrder = []
	nodeNames = {}
	nodeList = {}
	nodeToEdgeConnections = {}
	nodeFeatureLength = {}
	for node_name, node_type in zip(lines[0].strip().split(','),lines[1].strip().split(',')):
		nodeOrder.append(node_name)
		nodeNames[node_name] = node_type
		nodeList[node_type] = 0
		nodeToEdgeConnections[node_type] = {}
		nodeToEdgeConnections[node_type][node_type+'_input'] = [0,0]
		nodeFeatureLength[node_type] = 0
	
	edgeList = []
	edgeFeatures = {}
	nodeConnections = {}
	edgeListComplete = []
	for i in range(2,len(lines)):
		first_nodeName = nodeOrder[i-2]
		first_nodeType = nodeNames[first_nodeName]
		nodeConnections[first_nodeName] = []
		connections = lines[i].strip().split(',')
		for j in range(len(connections)):
			if connections[j] == '1':
				second_nodeName = nodeOrder[j]
				second_nodeType = nodeNames[second_nodeName]
				nodeConnections[first_nodeName].append(second_nodeName)
		
				edgeType_1 = first_nodeType + '_' + second_nodeType
				edgeType_2 = second_nodeType + '_' + first_nodeType
				edgeType = ''
				if edgeType_1 in edgeList:
					edgeType = edgeType_1
					continue
				elif edgeType_2 in edgeList:
					edgeType = edgeType_2
					continue
				else:
					edgeType = edgeType_1
				edgeList.append(edgeType)
				edgeListComplete.append(edgeType)

				if (first_nodeType + '_input') not in edgeListComplete:
					edgeListComplete.append(first_nodeType + '_input')
				if (second_nodeType + '_input') not in edgeListComplete:
					edgeListComplete.append(second_nodeType + '_input')

				edgeFeatures[edgeType] = 0
				nodeToEdgeConnections[first_nodeType][edgeType] = [0,0]
				nodeToEdgeConnections[second_nodeType][edgeType] = [0,0]

	trX = {}
	trY = {}
	trX_validate = {}
	trY_validate = {}
	trX_forecast = {}
	trY_forecast = {}
	trX_nodeFeatures = {}
	poseDataset.addNoiseToFeatures(noise=noise)
	for nodeName in nodeNames.keys():
		edge_features = {}
		validate_edge_features = {}
		forecast_edge_features = {}

		nodeType = nodeNames[nodeName]
		edgeTypesConnectedTo = nodeToEdgeConnections[nodeType].keys()
		low = 0
		high = 0

		for edgeType in edgeTypesConnectedTo:
			[edge_features[edgeType],validate_edge_features[edgeType],forecast_edge_features[edgeType]] = poseDataset.getfeatures(nodeName,edgeType,nodeConnections,nodeNames,forecast_on_noisy_features=forecast_on_noisy_features)

		edgeType = nodeType + '_input'
		D = edge_features[edgeType].shape[2]
		nodeFeatureLength[nodeType] = D
		high += D
		nodeToEdgeConnections[nodeType][edgeType][0] = low
		nodeToEdgeConnections[nodeType][edgeType][1] = high
		low = high
		nodeRNNFeatures = copy.deepcopy(edge_features[edgeType])
		validate_nodeRNNFeatures = copy.deepcopy(validate_edge_features[edgeType])
		forecast_nodeRNNFeatures = copy.deepcopy(forecast_edge_features[edgeType])

		for edgeType in edgeList:
			if edgeType not in edgeTypesConnectedTo:
				continue
			D = edge_features[edgeType].shape[2]
			edgeFeatures[edgeType] = D
			high += D
			nodeToEdgeConnections[nodeType][edgeType][0] = low
			nodeToEdgeConnections[nodeType][edgeType][1] = high
			low = high
			nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)	
			validate_nodeRNNFeatures = np.concatenate((validate_nodeRNNFeatures,validate_edge_features[edgeType]),axis=2)	
			forecast_nodeRNNFeatures = np.concatenate((forecast_nodeRNNFeatures,forecast_edge_features[edgeType]),axis=2)	

		[Y,Y_validate,Y_forecast,X_forecast,num_classes] = poseDataset.getlabels(nodeName)
		nodeList[nodeType] = num_classes
		
		idx = nodeName + ':' + nodeType
		trX[idx] = nodeRNNFeatures
		trX_validate[idx] = validate_nodeRNNFeatures
		trX_forecast[idx] = forecast_nodeRNNFeatures
		trY[idx] = Y
		trY_validate[idx] = Y_validate
		trY_forecast[idx] = Y_forecast
		trX_nodeFeatures[idx] = X_forecast
	print nodeToEdgeConnections
	print edgeListComplete
	return nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validate,trY_validate,trX_forecast,trY_forecast,trX_nodeFeatures	

def getNodeFeature(nodeName,nodeFeatures,nodeFeatures_t_1,poseDataset):
	edge_features = {}
	nodeType = nodeNames[nodeName]
	edgeTypesConnectedTo = nodeToEdgeConnections[nodeType].keys()
	low = 0
	high = 0

	for edgeType in edgeTypesConnectedTo:
		edge_features[edgeType] = poseDataset.getDRAfeatures(nodeName,edgeType,nodeConnections,nodeNames,nodeFeatures,nodeFeatures_t_1)

	edgeType = nodeType + '_input'
	nodeRNNFeatures = copy.deepcopy(edge_features[edgeType])

	for edgeType in edgeList:
		if edgeType not in edgeTypesConnectedTo:
			continue
		nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)

	return nodeRNNFeatures
