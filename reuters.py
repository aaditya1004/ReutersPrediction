from keras.datasets import reuters 
from keras import models,layers
from keras.utils import to_categorical
import numpy as np 

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words = 10000)
# print(train_data)

# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 
# decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in test_data[3]])
# print(decoded_newswire)

def vectorize_sequence(sequences, dimension = 10000):

	results = np.zeros((len(sequences),dimension))
	for i,sequence in enumerate(sequences):
		results[i,sequence]=1
		return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

one_hot_train_labels=to_categorical(train_labels)
one_hot_test_labels=to_categorical(test_labels)

model=models.Sequential()
model.add(layers.Dense(512, activation ='relu',input_shape=(10000,)))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(46, activation ='softmax'))

model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
prediction = model.predict(x_test)
# for i in prediction:
# 	if(i > 0.5):
# 		print("positive")
# 	else:
# 		print("negative")
print(prediction)
