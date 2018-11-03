from keras import layers
from keras import models
from keras import optimizers

import numpy as np
from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=100)

def vectorize_sequences(sequences,dimension=100):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(100,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

from keras import losses
from keras import metrics

#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#model.compile(optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

x_val=x_train[:10]
partial_x_train=x_train[10:]


y_val=y_train[:10]
partial_y_train=y_train[10:]

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(partial_x_train,partial_y_train,epochs=5,batch_size=512,validation_data=(x_val,y_val))

test_loss,test_acc=model.evaluate(x_test,y_test)
print('test_loss=',test_loss)
print('test_acc=',test_acc)

from keras import backend as K
K.clear_session()
