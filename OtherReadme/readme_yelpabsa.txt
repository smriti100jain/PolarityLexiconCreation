
yelpabsa_DATA_DIR=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa_300/yelpabsa

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/data/gen_vocab.py --output_dir=$yelpabsa_DATA_DIR  --dataset=yelpabsa  --yelpabsa_input_dir=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa/aclyelpabsa   --lowercase=False

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/data/gen_data.py --output_dir=$yelpabsa_DATA_DIR  --dataset=yelpabsa  --yelpabsa_input_dir=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa/aclyelpabsa --lowercase=False --label_gain=False


#################################################################################################################################

Word to vec embedding for the vocabulary..stored as npz file

word2vecembeddingpath=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa_300/yelpabsa/
CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/data/embedding.py --output_dir=$yelpabsa_DATA_DIR  --dataset=yelpabsa  --yelpabsa_input_dir=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa/aclyelpabsa --lowercase=False --label_gain=False --word2vec_initialize=True


##################################################################################################################################

PRETRAIN_DIR=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa_300/yelpabsa/models/yelpabsa_pretrain

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/pretrain.py --train_dir=$PRETRAIN_DIR --data_dir=$yelpabsa_DATA_DIR --vocab_size=15625 --embedding_dims=300 --rnn_cell_size=1024 --num_candidate_samples=1024 --optimizer=adam --batch_size=256 --learning_rate=0.001 --learning_rate_decay_factor=0.9999 --max_steps=100000 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --copt=-msse4.2 --word2vec_initialize=True --word2vecembeddingpath=$word2vecembeddingpath

INFO:tensorflow:step 100000, loss = 0.63 (189.5 examples/sec; 1.351 sec/batch)

_______




TRAIN_NON_ADV_DIR=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa_300/yelpabsa/models/yelpabsa_train_NON_ADV
CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/train_classifier.py --train_dir=$TRAIN_NON_ADV_DIR --pretrained_model_dir=$PRETRAIN_DIR --data_dir=$yelpabsa_DATA_DIR --vocab_size=13299 --embedding_dims=300 --rnn_cell_size=1024 --cl_num_layers=1 --optimizer=adam --batch_size=64 --learning_rate=0.0005 --learning_rate_decay_factor=0.9998 --max_steps=10000 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --word2vec_initialize=True --word2vecembeddingpath=$word2vecembeddingpath



 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evalute_embedding.py --eval_dir=$EVAL_DIR --checkpoint_dir=$TRAIN_NON_ADV_DIR --eval_data=test --run_once --num_examples=25000 --data_dir=$yelpabsa_DATA_DIR --vocab_size=13299 --embedding_dims=256 --rnn_cell_size=1024 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$TRAIN_NON_ADV_DIR --normalize_embeddings




no pretraining:

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/train_classifier.py --train_dir=$TRAIN_NON_ADV_DIR --data_dir=$yelpabsa_DATA_DIR --vocab_size=15625 --embedding_dims=300 --rnn_cell_size=1024 --cl_num_layers=1 --cl_hidden_size=30  --optimizer=adam --batch_size=64 --learning_rate=0.0005 --learning_rate_decay_factor=0.9998 --max_steps=100000 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --word2vec_initialize=True --word2vecembeddingpath=$word2vecembeddingpath

________
____TRAIN VAT over TRAIN_NON_ADV_DIR

train_NON_ADV_THEN_VAT_DIR=../../../../../opt/ADL_db/Users/sjain/NLP/AdversarialTraining/yelpabsa_300/yelpabsa/models/yelpabsa_train_NON_ADV_THEN_VAT

CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/train_classifier.py --train_dir=$train_NON_ADV_THEN_VAT_DIR --pretrained_model_dir=$TRAIN_NON_ADV_DIR --data_dir=$yelpabsa_DATA_DIR --vocab_size=13299 --embedding_dims=256 --rnn_cell_size=1024 --cl_num_layers=1 --cl_hidden_size=30 --optimizer=adam --batch_size=64 --learning_rate=0.0005 --learning_rate_decay_factor=0.9998 --max_steps=15000 --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings --adv_training_method=vat 

______
evaluate pretrain_dir embeddings:


 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evalute_embedding.py --eval_dir=$EVAL_DIR --checkpoint_dir=$train_NON_ADV_THEN_VAT_DIR --eval_data=test --run_once --num_examples=25000 --data_dir=$yelpabsa_DATA_DIR --vocab_size=15625 --embedding_dims=256 --rnn_cell_size=1024 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$train_NON_ADV_THEN_VAT_DIR --normalize_embeddings

test pretrain_dir embeddings

python adversarial_text/test_embedding.py --saveEmbeddingPath=$PRETRAIN_DIR --vocabPath=$yelpabsa_DATA_DIR



evaluate TRAIN_NON_ADV_DIR embeddings:


 CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python adversarial_text/evalute_embedding.py --eval_dir=$EVAL_DIR --checkpoint_dir=$TRAIN_NON_ADV_DIR --eval_data=test --run_once --num_examples=25000 --data_dir=$yelpabsa_DATA_DIR --vocab_size=15625 --embedding_dims=256 --rnn_cell_size=1024 --batch_size=256 --num_timesteps=400 --saveEmbeddingPath=$TRAIN_NON_ADV_DIR --normalize_embeddings

test TRAIN_NON_ADV_DIR embeddings

python adversarial_text/test_embedding.py --saveEmbeddingPath=$TRAIN_NON_ADV_DIR --vocabPath=$yelpabsa_DATA_DIR

