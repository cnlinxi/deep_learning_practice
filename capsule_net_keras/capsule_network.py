# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 23:52
# @Author  : MengnanChen
# @FileName: capsule_network.py
# @Software: PyCharm Community Edition

from keras import layers
from keras import backend as K
from keras import models
from keras import callbacks
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from capsule_layer import CapsuleLayer,PrimaryCap,Length,Mask

K.set_image_data_format(data_format='channel_last')

def capsule_net(input_shape,n_class,routings):
    x=layers.Input(shape=input_shape)

    conv1=layers.Conv2D(filters=256,kernel_size=9,strides=1,padding='valid',
                        activation='relu',name='conv1')(x)
    primary_capsule=PrimaryCap(conv1,dim_capsule=8,n_channels=32,
                               kernel_size=9,strides=2,padding='valid')
    digit_capsule=CapsuleLayer(output_dim_capsules=16,routings=routings,
                               output_num_capsules=n_class,name='digit_capsule')(primary_capsule)

    output_capsules=Length(name='capsule_net')(digit_capsule)

    y=layers.Input(shape=(n_class,))
    masked_with_y=Mask()([digit_capsule,y])
    masked=Mask(digit_capsule)

    decoder=models.Sequential(name='decoder')
    decoder.add(layers.Dense(units=512,activation='relu',input_dim=16*n_class))
    decoder.add(layers.Dense(units=1024,activation='relu'))
    decoder.add(layers.Dense(units=np.prod(input_shape),activation='softmax'))
    decoder.add(layers.Reshape(target_shape=input_shape,name='output_reconstruction'))

    train_model=models.Model([x,y],[output_capsules,decoder(masked_with_y)])
    eval_model=models.Model(x,[output_capsules,decoder(masked)])

    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digit_capsule, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

def margin_loss(y_true,y_pred):
    L=y_true*K.square(K.maximum(0.,0.9-y_pred))+\
      0.5*(1-y_true)*K.square(K.maximum(0.,y_pred-0.1))
    return K.mean(K.sum(L,axis=1))

def train(model,data,args):
    (x_train,y_train),(x_test,y_test)=data
    log=callbacks.CSVLogger(args.save_dir+'/log.csv')
    tb=callbacks.TensorBoard(log_dir=args.save_dir+'tensorboard-logs',
                             batch_size=args.batch_size,histogram_freq=int(args.debug))
    checkpoint=callbacks.ModelCheckpoint(args.save_dir+'/weights-{epoch:02d}.h5',monitor='val_capsnet_acc',
                                         save_best_only=True,save_weights_only=True,verbose=1)
    lr_decay=callbacks.LearningRateScheduler(schedule=lambda epoch:args.lr*(args.lr_decay**epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss,'mse'],
                  loss_weight=[1., args.lam_recon],
                  metrics={'capsnet':'accuracy'})
    def train_generator(x,y,batch_size,shift_fraction=0.):
        train_data_generator=ImageDataGenerator(width_shift_range=shift_fraction, # 图片宽度的某个比例，数据提升时图片水平偏移的幅度
                                                height_shift_range=shift_fraction)
        generator=train_data_generator.flow(x,y,batch_size)
        while True:
            x_batch,y_batch=generator.next()
            yield x_batch,y_batch

    model.fit_generator(generator=train_generator(x_train,y_train,args.batch_size,args.shift_fraction),
                        steps_per_epoch=int(x_train.shape[0]/args.batch_size),
                        epochs=args.epoch,
                        validation_data=[[x_test,y_test],[y_test,y_test]],
                        callbacks=[log,tb,checkpoint,lr_decay])
    model.save_weights(args.save_dir+'/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    # plot log

    return model

def load_mnist():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train=x_train.reshape(-1,28,28,1).astype('float32')/255.
    x_test=x_test.reshape(-1,28,28,1).astype('float32')/255.
    y_train=to_categorical(y_train.astype('float32'))
    y_test=to_categorical(y_test.astype('float32'))
    return (x_train,y_train),(x_test,y_test)

if __name__ == '__main__':
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    if os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    (x_train,y_train),(x_test,y_test)=load_mnist()
    train_model,eval_model,manipulate_model=capsule_net(x_train.shape[1:],
                                                        n_class=y_train.shape[1],
                                                        routings=args.routings)
    train_model.summary()

