import os
import pandas as pd
import os
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
datafiles = os.listdir('json')
os.makedirs('TripadvisorHotels')
for i in range(0,6):
	os.makedirs(os.path.join('TripadvisorHotels',str(i)))
index =0
for i in datafiles:
	reviews = pd.read_json(os.path.join('json',i),typ="series")['Reviews']
	for j in range(len(reviews)):
		rating = int(float(reviews[j]['Ratings']['Overall']))
		content = reviews[j]['Content']
		target = open(os.path.join('TripadvisorHotels',str(rating),'trip'+str(index)+'.txt'),'w')
		target.write(content)
		target.close()
		index = index+1

