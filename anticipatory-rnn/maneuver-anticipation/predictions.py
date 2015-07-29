import numpy as np

def predictLastTimeManeuver(time_prediction,maneuver_name):
	return time_prediction[-1]

def predictManeuver(time_prediction,maneuver_name):
	end_action_index = maneuver_name.index('end_action')
	prediction = end_action_index
	count = 1.0
	delta_frames = 20.0
	anticipation_time = 0.0
	for i in time_prediction:
		anticipation_time = (len(time_prediction)*1.0 - count)*delta_frames*1.0/25.0
		if not i == end_action_index:
			prediction = i
			break
		count += 1.0
	return prediction,anticipation_time
	
