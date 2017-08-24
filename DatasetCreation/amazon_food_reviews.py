import pandas as pd
import numpy as np
import os
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
np.random.seed(0)

def read_text_file(f):
    df_complete = pd.read_csv(f)
    #df = df_complete.loc[:,["Text","Score"]]
    #df.dropna(how="any", inplace=True)    
    return df_complete

df = read_text_file("Reviews.csv")
print (df.head())
os.makedirs('AmazonFineFood')
for i in range(1,6):
	os.makedirs(os.path.join('AmazonFineFood',str(i)))
for index,row in df.iterrows():
	review = str(row['Summary'])+'.'+str(row['Text'])
	score = row['Score']
	target = open(os.path.join('AmazonFineFood',str(score),str(index)+'.txt'),'w')
	target.write(review)
	target.close()