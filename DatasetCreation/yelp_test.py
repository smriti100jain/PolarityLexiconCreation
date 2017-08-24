import pandas as pd
import os
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
category = 'All'
# read the entire file into a python array
with open('yelp_business.json', 'rb') as f:
    data = f.readlines()
# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ','.join(data) + "]"
# now, load it into pandas
data_df = pd.read_json(data_json_str)
restaurants = []
for i in range(data_df.shape[0]):
	temp = data_df['categories'][i]
	#for temp1 in temp:
		#if(temp1==category):
	restaurants.append(data_df['business_id'][i])
	#		break
with open('yelp_review.json', 'rb') as f:
    data = f.readlines()
# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ','.join(data) + "]"
# now, load it into pandas
data_df2 = pd.read_json(data_json_str)
data_df.loc[data_df['business_id'].isin(restaurants)]['categories']
RestraReviews = data_df2.loc[data_df2['business_id'].isin(restaurants)][['stars','text']]
#os.makedirs('Yelp_Restaurant')
'''
for i in range(6):
	os.makedirs(os.path.join('Yelp_Restaurant',str(i)))
'''
for i in RestraReviews.index:
	print(i)
	rating = RestraReviews['stars'][i]
	review = RestraReviews['text'][i]
	target = open(os.path.join('Yelp_'+category,str(rating),str(i)+'.txt'),'w')
	#print(review)
	#print(rating)
	target.write(review)
	target.close()


#15264 reviews in total.
