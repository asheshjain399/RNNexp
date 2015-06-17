import random

def sixDigitRandomNum():	
	st = ''
	for i in range(6):
		st = st + str(random.randint(0,9))
	return st
	
