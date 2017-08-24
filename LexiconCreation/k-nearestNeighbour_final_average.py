from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import numpy as np
import sys
import codecs
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
reload(sys)
sys.setdefaultencoding('UTF8')
baseline = 0

if(not(os.path.isdir('knearestNeighbourFinalaverage'))):
        os.mkdir('knearestNeighbourFinalaverage')
dataset = 1
 
if(dataset == 1):
    data_vocab = '../WordEmbedding/ABSA/vocab.txt'
elif(dataset == 2):
    data_vocab = '../WordEmbedding/ABSA/yelpabsa/vocab.txt'
elif(dataset ==3):
    data_vocab = '.../WordEmbedding/ABSA/imdbabsa/vocab.txt'
elif(dataset ==4):
    data_vocab = '../WordEmbedding/ABSA/fullData/vocab.txt'



posTerms = ["happy", "satisfied", "good", "positive", "excellent", "flawless", "unimpaired"]
negTerms = ["unhappy", "unsatisfied", "bad", "negative", "poor", "flawed", "defective"]
neutTerms = ["food","building",'administration','people','cars','fruits','evening','lunch','place','dish']
#,'did','is','was','the','restaurant','waiter'
POSNEG_TOTAL_SIM_MIN = 0.09#0.078
POSNEG_SINGLE_SIM_MIN = 0.00
POSNEG_DIFF_MIN = 0.02
minDiff = 0.01#0.005
filename = '../WordEmbedding/adversarial_text/data/wordvectors_w2v800k.tsv.gz'
devpol = 'gold_dev.pol'
testpol = 'gold_test.pol'
#######################################################################################################################################


vocab_existing = []
embd = []
count = 0
nonnormal = 0
file_h = codecs.getreader("utf-8")(gzip.open(filename))
#file_h = gzip.open(filename,'rt',encoding='utf-8')
for line in file_h:
    line = line.strip()
    word,vec = line.split('\t')
    if(len(word)>=2 and all(unicodedata.category(i)=='Ll' for i in word) ):
        vec = vec.split(' ')
        vocab_existing.append(word)
        vec = map(float,vec[0:])
        embd.append(vec)
        count = count + 1
    else:
        nonnormal = nonnormal + 1


vocab_size = len(vocab_existing)
embedding_dim = len(embd[0])
embd = np.array(embd)

if(baseline == 1):
    vocab = vocab_existing
else:
    f = codecs.open(data_vocab, "r", "utf-8")
    vocab = (f.read().strip()).split('\n')
    f.close()

vocab_present = []
count = 0
mask = np.in1d(vocab_existing, vocab)
vocab_present = list(np.where(mask==True)[0])


vocab_not_present = list(set(range(vocab_size))-set(vocab_present))
for i in vocab_not_present:
        vocab.append(vocab_existing[i])
vocab_size = len(vocab)
print('W2vec loading finished.......................................')
#######################################################################################################################################
TrainAcc = []
TestAcc = []

filename = 'manually_mine_treshold_w2v_'+str(dataset)
if(not(os.path.isdir(os.path.join('knearestNeighbourFinalaverage',str(dataset))))):
        os.mkdir(os.path.join('knearestNeighbourFinalaverage',str(dataset)))
file3 = open(os.path.join('knearestNeighbourFinalaverage',str(dataset),filename+'_confusionmatrix_test'+'.csv'),'wb')
file4 = open(os.path.join('knearestNeighbourFinalaverage',str(dataset),filename+'_confusionmatrix_train'+'.csv'),'wb')






posTerms_id = [vocab.index(item) for item in posTerms]
negTerms_id = [vocab.index(item) for item in negTerms]
neutTerms_id = [vocab.index(item) for item in neutTerms]

f = codecs.open(devpol, "r", "utf-8")
dev_vocab = []
dev_polarity = []
for line in f:
        Temp=line.split()
        word = Temp[0]
        polarity = Temp[2]
        word.replace('"','')
        word.replace('_','')
        word.replace('-','')
        dev_vocab.append(word.lower())
        if 'positive' in polarity.lower():
            dev_polarity.append(1)
        elif('negative' in polarity.lower()):
            dev_polarity.append(-1)
        else:
            dev_polarity.append(0)

dev_vocab.extend(posTerms)
dev_vocab.extend(negTerms)
dev_vocab.extend(neutTerms)
dev_polarity.extend([1]*len(posTerms))
dev_polarity.extend([-1]*len(posTerms))
dev_polarity.extend([0]*len(neutTerms))

devpol_id = []
for item in dev_vocab:
        if(item in vocab):
            devpol_id.append(vocab.index(item))
        else:
            devpol_id.append(-1) #not present


#test
f = codecs.open(testpol, "r", "utf-8")

test_vocab = []
test_polarity = []
for line in f:
        Temp=line.split()
        word = Temp[0]
        polarity = Temp[2]
        word.replace('"','')
        word.replace('_','')
        word.replace('-','')
        test_vocab.append(word.lower())
        if 'positive' in polarity.lower():
            test_polarity.append(1)
        elif('negative' in polarity.lower()):
            test_polarity.append(-1)
        else:
            test_polarity.append(0)



testpol_id = []
for item in test_vocab:
        if(item in vocab):
            testpol_id.append(vocab.index(item))
        else:
            testpol_id.append(-1) #not present


#test_vocab,test_polarity
dev_polarity = np.array(dev_polarity)
test_polarity = np.array(test_polarity)


for model in [0,1,2,3,4,5]:

    if(model==0):
            languagemodel = 1
            nonadvtraining = 0
            advtraining = 0
            vattraining = 0
    elif(model ==1):
            languagemodel = 0
            nonadvtraining = 1
            advtraining = 0
            vattraining = 0
    elif(model ==2):
            languagemodel = 0
            nonadvtraining = 0
            advtraining = 1
            vattraining = 0
    elif(model ==3):
            languagemodel = 0
            nonadvtraining = 0
            advtraining = 0
            vattraining = 1
    if(not(os.path.isdir(os.path.join('knearestNeighbourFinalaverage',str(dataset),str(model))))):
        os.mkdir(os.path.join('knearestNeighbourFinalaverage',str(dataset),str(model)))
    file1 = open(os.path.join('knearestNeighbourFinalaverage',str(dataset),str(model),filename+'_test'+'_'+str(model)+'.csv'),'wb')
    file2 = open(os.path.join('knearestNeighbourFinalaverage',str(dataset),str(model),filename+'_train'+'_'+str(model)+'.csv'),'wb')
    file6 = open(os.path.join('knearestNeighbourFinalaverage',str(dataset),str(model),filename+'_generatedpollex'+'_'+str(model)+'.txt'),'wb')

    if(dataset == 1):
        data_vocab = '../WordEmbedding/ABSA/vocab.txt'
        if(languagemodel == 1):
            model_embedding = '../WordEmbedding/ABSA/models/LanguageModel/embedding.pklz'
        elif(nonadvtraining ==1):
            model_embedding = '../WordEmbedding/ABSA/models/ST/embedding.pklz'
        elif(advtraining==1):
            model_embedding = '../WordEmbedding/ABSA/models/AT/embedding.pklz'
        elif(vattraining==1):
            model_embedding = '../WordEmbedding/ABSA/models/VAT/embedding.pklz'
    elif(dataset == 2):
        data_vocab = '../WordEmbedding/yelpabsa/vocab.txt'
        if(languagemodel == 1):
            model_embedding = '../WordEmbedding/yelpabsa/models/LanguageModel/embedding.pklz'
        elif(nonadvtraining ==1):
            model_embedding = '../WordEmbedding/yelpabsa/models/ST/embedding.pklz'
        elif(advtraining==1):
            model_embedding = '../WordEmbedding/yelpabsa/models/AT/embedding.pklz'
        elif(vattraining==1):
            model_embedding = '../WordEmbedding/yelpabsa/models/VAT/embedding.pklz'

    elif(dataset ==3):
        data_vocab = '../WordEmbedding/imdbabsa/vocab.txt'
        if(languagemodel == 1):
            model_embedding = '../WordEmbedding/imdbabsa/models/LanguageModel/embedding.pklz'
        elif(nonadvtraining ==1):
            model_embedding = '../WordEmbedding/imdbabsa/models/ST/embedding.pklz'
        elif(advtraining==1):
            model_embedding = '../WordEmbedding/imdbabsa/models/AT/embedding.pklz'
        elif(vattraining==1):
            model_embedding = '../WordEmbedding/imdbabsa/models/VAT/embedding.pklz'


    elif(dataset ==4):
        data_vocab = '../WordEmbedding/full/vocab.txt'
        if(languagemodel == 1):
            model_embedding = '../WordEmbedding/full/models/LanguageModel/embedding.pklz'
        elif(nonadvtraining ==1):
            model_embedding = '../WordEmbedding/full/models/ST/embedding.pklz'
        elif(advtraining==1):
            model_embedding = '../WordEmbedding/full/models/AT/embedding.pklz'
        elif(vattraining==1):
            model_embedding = '../WordEmbedding/full/models/VAT/embedding.pklz'





    
    if(baseline == 1):
        embedding = np.asarray(embd)
    else:
        f = gzip.open(model_embedding,'rb')
        embedding = pickle.load(f)
        f.close()
        
        embedding = np.vstack((embedding, embd[vocab_not_present]))
        embedding = np.asarray(embedding)







    #train
    Embedding_reduced_train = embedding[devpol_id]
    #dev_vocab,dev_polarity

   
    #train
    Embedding_reduced_test = embedding[testpol_id]

    pss=np.array(np.average(cosine_similarity(Embedding_reduced_train,embedding[posTerms_id]),1))
    ngg=np.array(np.average(cosine_similarity(Embedding_reduced_train,embedding[negTerms_id]),1))
    neutt=np.array(np.average(cosine_similarity(Embedding_reduced_train,embedding[neutTerms_id]),1))
    probability=np.column_stack((pss,ngg,neutt))

    pss=np.array(np.average(cosine_similarity(Embedding_reduced_test,embedding[posTerms_id]),1))
    ngg=np.array(np.average(cosine_similarity(Embedding_reduced_test,embedding[negTerms_id]),1))
    neutt=np.array(np.average(cosine_similarity(Embedding_reduced_test,embedding[neutTerms_id]),1))
    probabiltiy_test=np.column_stack((pss,ngg,neutt))


    pss=np.array(np.average(cosine_similarity(embedding,embedding[posTerms_id]),1))
    ngg=np.array(np.average(cosine_similarity(embedding,embedding[negTerms_id]),1))
    neutt=np.array(np.average(cosine_similarity(embedding,embedding[neutTerms_id]),1))
    probabiltiy_final=np.column_stack((pss,ngg,neutt))


    '''
    train_calc_polarity = list(np.argmax(probability,1))
    test_calc_polarity = list(np.argmax(probabiltiy_test,1))

    for i in range(len(train_calc_polarity)):
        if(train_calc_polarity[i]==1):
            train_calc_polarity[i]=-1
        elif(train_calc_polarity[i]==0):
            train_calc_polarity[i] = 1
        elif(train_calc_polarity[i]==2):
            train_calc_polarity[i]=0

    for i in range(len(test_calc_polarity)):
        if(test_calc_polarity[i]==1):
            test_calc_polarity[i]=-1
        elif(test_calc_polarity[i]==0):
            test_calc_polarity[i] = 1
        elif(test_calc_polarity[i]==2):
            test_calc_polarity[i]=0
    '''
    from sklearn.neighbors import NearestNeighbors
    import warnings
    warnings.filterwarnings("ignore")
    train_calc_polarity = []
    neigh = NearestNeighbors(30, radius =0.5, metric='cosine',algorithm='brute')
    neigh.fit(Embedding_reduced_train)  
    #lshf = LSHForest(random_state=42,n_neighbors=30, radius=0.5)
    #lshf.fit(X_train)  

    for i in range(len(probability)):
        
        nbrs =  neigh.radius_neighbors(Embedding_reduced_train[i], return_distance=False)
        #nbrs = lshf.kneighbors(Embedding_reduced_train[i], return_distance=False)
        tempdd = 0
        for j in range(len(nbrs[0])):
            tempdd = tempdd+probability[nbrs[0][j]]
        tempdd = tempdd/len(nbrs[0])
        probability[i] = tempdd

        inde = probability[i].argsort()[-2:][::-1]
        if((probability[i][inde[0]]-probability[i][inde[1]])<0):
            train_calc_polarity.append(0)
        else:
            if(inde[0]==0):
                train_calc_polarity.append(1)
            elif(inde[0]==1):
                train_calc_polarity.append(-1)
            else:
                train_calc_polarity.append(0)

    neigh = NearestNeighbors(30, radius =0.5, metric='cosine',algorithm='brute')
    neigh.fit(Embedding_reduced_test)  
    #lshf = LSHForest(random_state=42,n_neighbors=30, radius=0.5)
    #lshf.fit(X_train)  

    test_calc_polarity = []
    for i in range(len(probabiltiy_test)):
        nbrs =  neigh.radius_neighbors(Embedding_reduced_test[i], return_distance=False)
        #nbrs = lshf.kneighbors(Embedding_reduced_test[i], return_distance=False)
        tempdd = 0
        for j in range(len(nbrs[0])):
            tempdd = tempdd+probabiltiy_test[nbrs[0][j]]
        tempdd = tempdd/len(nbrs[0])
        probabiltiy_test[i] = tempdd
        inde = probabiltiy_test[i].argsort()[-2:][::-1]
        if((probabiltiy_test[i][inde[0]]-probabiltiy_test[i][inde[1]])<0):
            test_calc_polarity.append(0)
        else:
            if(inde[0]==0):
                test_calc_polarity.append(1)
            elif(inde[0]==1):
                test_calc_polarity.append(-1)
            else:
                test_calc_polarity.append(0)
    
    '''
    from joblib import Parallel, delayed
    import multiprocessing
    final_calc_polarity = []
    import warnings
    warnings.filterwarnings("ignore")
    neigh = NearestNeighbors(30, radius =0.5, metric='cosine',algorithm='brute')
    neigh.fit(embedding)
    import pickle 
    def assignfinal(embedding,probabiltiy_final,i):
        print(i)
        nbrs =  neigh.radius_neighbors(embedding[i], return_distance=False)
        tempdd = 0
        for j in range(len(nbrs[0])):
            tempdd = tempdd+probabiltiy_final[nbrs[0][j]]
        tempdd = tempdd/len(nbrs[0])
        #print(tempdd)
        return tempdd
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(assignfinal)(embedding,probabiltiy_final,i) for i in range(len(probabiltiy_final)))
    pickle.dump( results, open( "results.p", "wb" ) )
    #print(results)
    print('done')
    for i in range(len(probabiltiy_final)):
        print(i)
        probabiltiy_final[i] = list(results[i])
        inde = probabiltiy_final[i].argsort()[-2:][::-1]
        if((probabiltiy_final[i][inde[0]]-probabiltiy_final[i][inde[1]])<0):
            final_calc_polarity.append(0)
        else:
            if(inde[0]==0):
                final_calc_polarity.append(1)
            elif(inde[0]==1):
                final_calc_polarity.append(-1)
            else:
                final_calc_polarity.append(0)

    '''
    file1.write('word'+','+'Actual_Polarity'+','+'Calc_Polarity'+','+'NegativeSimilarity'+','+'NeutralSimilarity'+','+'PositiveSimilarity')
    file1.write('\n')
    import warnings
    warnings.filterwarnings("ignore")
    for count in range(len(testpol_id)):
        file1.write(str(test_vocab[count])+','+str(test_polarity[count])+','+str(test_calc_polarity[count])+','+str(probabiltiy_test[count][0])+','+str(probabiltiy_test[count][1])+','+str(probabiltiy_test[count][2]))
        file1.write('\n')
    file2.write('word'+','+'Actual_Polarity'+','+'Calc_Polarity'+','+'NegativeSimilarity'+','+'NeutralSimilarity'+','+'PositiveSimilarity')
    file2.write('\n')
    import warnings
    warnings.filterwarnings("ignore")
    for count in range(len(devpol_id)):
        file2.write(str(dev_vocab[count])+','+str(dev_polarity[count])+','+str(train_calc_polarity[count])+','+str(probability[count][0])+','+str(probability[count][1])+','+str(probability[count][2]))
        file2.write('\n')
 
    '''
    import warnings
    warnings.filterwarnings("ignore")
    for count in range(len(vocab)):
        if(vocab[count] in vocab_existing):
            if(len(vocab[count])>=2 and all(unicodedata.category(p)=='Ll' for p in vocab[count]) ):
                if(final_calc_polarity[count]==0):
                    pol = 'neut'
                elif(final_calc_polarity[count]==1):
                    pol = 'pos'
                elif(final_calc_polarity[count]==-1):
                    pol = 'neg'
                #neut neg pos
                file6.write(str(vocab[count])+'\t'+str(pol)+'\t'+str(probabiltiy_final[count][2])+'\t'+str(probabiltiy_final[count][1])+'\t'+str(probabiltiy_final[count][0]))
                file6.write('\n')
        
    '''
    from sklearn.metrics import confusion_matrix
    mat1 = confusion_matrix(dev_polarity, train_calc_polarity)
    print(mat1)


    mat2= confusion_matrix(test_polarity, test_calc_polarity)
    print(mat2)
   
    
    acc1 = accuracy_score(dev_polarity,train_calc_polarity)
    acc2 = accuracy_score(test_polarity,test_calc_polarity)

    print(acc1)
    print(acc2)
    TrainAcc.append(acc1)
    TestAcc.append(acc2)




    file3.write(str(mat1)+'\n'+'\n')
    file4.write(str(mat2)+'\n'+'\n')
    print('model done')
    if(baseline ==1 ):
        break

if(baseline != 1):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    X = range(len(TrainAcc))
    plt.plot(X,TrainAcc,'r--',label='Train Accuracy')
    for xy in zip(X, TrainAcc):                                      
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    plt.plot(X,TestAcc,'b--', label='Test Accuracy')
    for xy in zip(X, TestAcc):                                      
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

    plt.grid()
    plt.show()
    fig.savefig(os.path.join('knearestNeighbourFinalaverage',str(dataset),filename+'.png'))




    