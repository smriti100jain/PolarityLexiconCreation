import sys
import xml.etree.ElementTree as ET
import os
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
train_filename = 'ABSA16_Restaurants_Train_SB1_corrected.xml'
tree = ET.parse(train_filename)
os.makedirs('ABSA','train')
os.makedirs(os.path.join('ABSA','train','pos'))
os.makedirs(os.path.join('ABSA','train','neg'))
os.makedirs(os.path.join('ABSA','train','unsup'))

sentences = tree.findall('.//sentence')
print(sentences)
count = 0
for sentence in sentences:
	count = count+1
	review = sentence.find('.//text').text
	if(sentence.find('.//Opinions') is None):
		target = open(os.path.join('ABSA','train','unsup',str(count)+'.txt'),'w')
 		target.write(review)
 		target.close()
 		continue
 	opinions = sentence.find('.//Opinions').findall('.//Opinion')
 	if(len(opinions)==1):
 		for opinion in opinions:
 			polarity = opinion.attrib['polarity']
 			print(polarity)
 			if(polarity=='negative'):
 				target = open(os.path.join('ABSA','train','neg',str(count)+'.txt'),'w')
 				target.write(review)
 				target.close()	
 			elif(polarity=='positive'):
 				target = open(os.path.join('ABSA','train','pos',str(count)+'.txt'),'w')
 				target.write(review)
 				target.close()
 			else:
 				target = open(os.path.join('ABSA','train','unsup',str(count)+'.txt'),'w')
 				target.write(review)
 				target.close()
 	elif(len(opinions)>1):
 		pos = 0
 		neg = 0
 		neut = 0
 		for opinion in opinions:
 			polarity = opinion.attrib['polarity']
 			print(polarity)
 			if(polarity=='negative'):
 				neg=neg+1
 			elif(polarity=='positive'):
 				pos=pos+1
 			else:
 				neut=neut+1
 		if((pos+neut)==len(opinions)):
 			target = open(os.path.join('ABSA','train','pos',str(count)+'.txt'),'w')
 			target.write(review)
 			target.close()
 		elif((neg+neut)==len(opinions)):
 			target = open(os.path.join('ABSA','train','neg',str(count)+'.txt'),'w')
 			target.write(review)
 			target.close()
 		else:
 			target = open(os.path.join('ABSA','train','unsup',str(count)+'.txt'),'w')
 			target.write(review)
 			target.close()

 				
 			 				 




