requirement:
TensorFlow >= v1.1
keras
gensim
————————————————————————————————————————
cd WordEmbedding
absa_DATA_DIR=ABSA

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1`  python adversarial_text/data/gen_vocab.py --output_dir=$absa_DATA_DIR  --dataset=yelpabsa  --yelpabsa_input_dir=../DatasetCreation/ABSA --lowercase=True

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1`  python adversarial_text/data/gen_data.py --output_dir=$absa_DATA_DIR  --dataset=yelpabsa  --yelpabsa_input_dir=../DatasetCreation/ABSA --lowercase=True --label_gain=False

###################################################################################################################################

Word to vec embedding for the vocabulary..stored as npz file (for initialisation)

word2vecembeddingpath=ABSA

python adversarial_text/data/embedding.py --word2vecembeddingpath=$word2vecembeddingpath
###################################################################################################################################
Language Model Training:

LangModel=ABSA/models/LanguageModel

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/pretrain.py --train_dir=$LangModel --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --num_candidate_samples=1024 --optimizer=adam --batch_size=256 --learning_rate=0.001 --learning_rate_decay_factor=0.9999 --max_steps=8000 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --copt=-msse4.2 --word2vec_initialize=True --word2vecembeddingpath=$word2vecembeddingpath

Test accuracy:

 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evaluate.py --eval_dir=$EVAL_DIR --checkpoint_dir=$LangModel --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$LangModel --normalize_embeddings

embedding:
 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evalute_embedding.py --eval_dir=$EVAL_DIR --checkpoint_dir=$LangModel --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$LangModel --normalize_embeddings

python adversarial_text/test_embedding.py --saveEmbeddingPath=$LangModel --vocabPath=$absa_DATA_DIR

###################################################################################################################################
Normal Sentiment Classification Training:


TRAIN_NON_ADV_DIR=ABSA/models/ST

no pretraining required for language model as Google w2v already really good.

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/train_classifier.py --train_dir=$TRAIN_NON_ADV_DIR --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --cl_num_layers=1 --cl_hidden_size=30  --optimizer=adam --batch_size=64 --learning_rate=0.0005 --learning_rate_decay_facto1550r=0.9998 --max_steps=2000 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --word2vec_initialize=True --word2vecembeddingpath=$word2vecembeddingpath

Test accuracy:

 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evaluate.py --eval_dir=$EVAL_DIR --checkpoint_dir=$TRAIN_NON_ADV_DIR --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$TRAIN_NON_ADV_DIR --normalize_embeddings

embedding:
 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evalute_embedding.py --eval_dir=$EVAL_DIR --checkpoint_dir=$TRAIN_NON_ADV_DIR --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$TRAIN_NON_ADV_DIR --normalize_embeddings

python adversarial_text/test_embedding.py --saveEmbeddingPath=$TRAIN_NON_ADV_DIR --vocabPath=$absa_DATA_DIR



##################################################################################################################################
vat training


train_CL_VAT_DIR=ABSA/models/VAT

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/train_classifier.py --train_dir=$train_CL_VAT_DIR --data_dir=$absa_DATA_DIR --pretrained_model_dir=$TRAIN_NON_ADV_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --cl_num_layers=1 --cl_hidden_size=30 --optimizer=adam --batch_size=64 --learning_rate=0.0005 --learning_rate_decay_factor=0.9998 --max_steps=1500 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --adv_training_method=vat

Test accuracy:

 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evaluate.py --eval_dir=$EVAL_DIR --checkpoint_dir=$train_CL_VAT_DIR --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$train_CL_VAT_DIR --normalize_embeddings

embedding:
 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evalute_embedding.py --eval_dir=$EVAL_DIR --checkpoint_dir=$train_CL_VAT_DIR --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$train_CL_VAT_DIR --normalize_embeddings

python adversarial_text/test_embedding.py --saveEmbeddingPath=$train_CL_VAT_DIR --vocabPath=$absa_DATA_DIR

#####################################

AT training

train_CL_AT_DIR=ABSA/models/AT

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/train_classifier.py --train_dir=$train_CL_AT_DIR --data_dir=$absa_DATA_DIR --pretrained_model_dir=$TRAIN_NON_ADV_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --cl_num_layers=1 --cl_hidden_size=30 --optimizer=adam --batch_size=64 --learning_rate=0.0005 --learning_rate_decay_factor=0.9998 --max_steps=1500 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --adv_training_method=at

Test accuracy:

 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evaluate.py --eval_dir=$EVAL_DIR --checkpoint_dir=$train_CL_AT_DIR --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$train_CL_AT_DIR --normalize_embeddings

embedding:
 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evalute_embedding.py --eval_dir=$EVAL_DIR --checkpoint_dir=$train_CL_AT_DIR --eval_data=test --run_once --num_examples=674 --data_dir=$absa_DATA_DIR --vocab_size=1550 --embedding_dims=300 --rnn_cell_size=512 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$train_CL_AT_DIR --normalize_embeddings

python adversarial_text/test_embedding.py --saveEmbeddingPath=$train_CL_AT_DIR --vocabPath=$absa_DATA_DIR

##################################################################################################################################




Polarity Lexicon Creation:

(the polarity dataset is confidential and unable to run on real dataset)

cd ../LexiconCreation
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

SVM (rbf)  [supervised]

python svmclassifier_rbf.py

>>>>>>>>>>>>>>>>>>>>>>>>>>>

unsupervised [nearest neighbours]

python k-nearestNeighbour_final_average.py


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
code reference (Tensorflow tutorials):
https://github.com/tensorflow/models/tree/master/adversarial_text
















