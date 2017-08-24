import sys
import xml.etree.ElementTree as ET
import os
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
train_filename = 'ABSA16_Restaurants_Train_SB1_corrected.xml'
tree = ET.parse(train_filename)
os.makedir('ABSA')
os.makedir(os.path.join('ABSA','pos'))
os.makedir(os.path.join('ABSA','neg'))
sentences = tree.findall('.//sentence')
count = 0
for sentence in sentences:
	review = sentence.find('.//text').text
 	opinions = sentence.find('.//Opinions').findall('.//Opinion')

 	if(len(opinions)==1):
 		for opinion in opinions:
 			polarity = opinion.attrib('polarity')
 			if(polarity=='negative'):
 				target = open(os.path.join('ABSA','neg',str(count)+'.txt'),'w')
 				target.write(review)
 				target.close()
 				
 			if(polarity=='positive'):
 				target = open(os.path.join('ABSA','pos',str(count)+'.txt'),'w')
 				target.write(review)
 				target.close()

 		count = count + 1
 				 




