import numpy as np

def predictLastTimeManeuver(time_prediction,maneuver_name):
	return time_prediction[-1]

def predictManeuver(time_prediction,maneuver_name):
	end_action_index = maneuver_name.index('end_action')
	prediction = end_action_index
	for i in time_prediction:
		if not i == end_action_index:
			prediction = i
			break
	return prediction
	
