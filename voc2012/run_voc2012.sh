:<<BLOCK
lr: learning rate
lrp: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is lr * lrp
batch-size: number of images per batch
image-size: size of the image
epochs: number of training epochs
evaluate: evaluate model on validation set
resume: path to checkpoint
BLOCK

# Train the model
# CUDA_VISIBLE_DEVICES=0 python3 demo_voc2012.py data/voc2012 --image-size 448 --batch-size 8 --lambd 10.0 --beta 0.0001

# Evaluate the trained model
CUDA_VISIBLE_DEVICES=0 python3 demo_voc2012.py data/voc2012 --image-size 448 --batch-size 8 -e --resume checkpoint/voc2012/model_best.pth.tar --lambd 10.0 --beta 0.0001
