import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Flatten,Conv2D,MaxPool2D
from keras.models import Model,Sequential
from glob import glob
import os
import argparse
from get_data import get_data
import matplotlib.pyplot as plt
import tensorflow


def train_model(config_file):
    
    config = get_data(config_file)
    train = config['model']['trainable']
    if train == True:

        img_size = config['model']['image_size']
        trn_set = config['model']['train_path']
        te_set = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        batch = config['img_augment']['batch_size']
        class_mode = config['img_augment']['class_mode']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']

        print(type(batch))
    
        

        #Install the Sequential model
        model = Sequential()
#Add the conv2d layer to the model with relu activation function and input sahpe
        model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
#add the another maxpoolind layer to the model
        model.add(MaxPool2D(2,2))
#add another convluation layer to the model
        model.add(Conv2D(64,(3,3),activation='relu'))
#Add the another max pooling layer
        model.add(MaxPool2D(2,2))
# Add another Covlution layer with relu activation function
        model.add(Conv2D(128,(3,3),activation='relu'))
#Add the another max pooling layer
        model.add(MaxPool2D(2,2))
#Add another Covlution layer with relu activation function
        model.add(Conv2D(128,(3,3),activation='relu'))
#Add the another max pooling layer
        model.add(MaxPool2D(2,2))
#Add the flattern layer to the model
        model.add(Flatten())
#Add the dense layer to the model with relu activation function
        model.add(Dense(512,activation='relu'))
#Add the dense layer to the model with sigmoid activation function
        model.add(Dense(1,activation='sigmoid'))


        model.compile(loss = loss ,optimizer = optimizer , metrics = metrics)

        train_gen = ImageDataGenerator(rescale = 1./255,
                            
                                    )

        test_gen = ImageDataGenerator(rescale = 1./255)

        train_set = train_gen.flow_from_directory(trn_set,
                                                target_size = (150,150),
                                                batch_size = batch,
                                                class_mode = class_mode
                                                )

        test_set = test_gen.flow_from_directory(te_set,
                                                target_size = (150,150),
                                                batch_size = batch,
                                                class_mode = class_mode
                                                )

        
        history = model.fit(train_set,
                                 epochs = epochs,
                                validation_data = test_set,
                                
        ) 
                                
        

        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'],label='val_loss')
        plt.legend()
        plt.savefig('reports/train_v_loss')

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'],label='val_acc')
        plt.legend()
        plt.savefig('reports/acc_v_vacc')
        

        model.save('saved_models/trained.h5')
        print('model saved by Xerxez Solutions')
    
    else:
        print('Model not trained by Xerxez Solutions')

    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',default='params.yaml')
    passed_args = args_parser.parse_args()
    train_model(config_file=passed_args.config)