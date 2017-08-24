Trip advisor dataset:
http://times.cs.uiuc.edu/~wang296/Data/


source /opt/Tools/python/2.7-gpu/share/tf-1.0/bin/activate
CUDA_VISIBLE_DEVICES=`/opt/Tools/bin/first-free-gpu|head -1` python -c 'import tensorflow as tf; print(tf.__version__)'


Restaurant Review dataset:

http://tour-pedia.org/about/datasets.html
