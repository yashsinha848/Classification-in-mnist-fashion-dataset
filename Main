import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.05):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
#Loading the mnist dataset
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#see how training data looks like
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[100])
print(training_labels[100])
print(training_images[100])

#normalize all data between 0 and 1
training_images  = training_images / 255.0
test_images = test_images / 255.0

#Making the model using relu and softmax activation layers
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#Compiling the model using ada, optimizer and using sparse_categorical_crossentropy as loss function.
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Fitting the images on training dataset
model.fit(training_images, training_labels, epochs=30,callbacks=[callbacks],validation_data=(test_images,test_labels),verbose=1)

#Returns the loss value & metrics values for the model in test mode.
model.evaluate(test_images, test_labels)

#Doing some testing
y_pr=model.predict(test_images)
y_pred=[]
for i in range(len(y_pr)):
  y_pred.append(np.argmax(y_pr[i]))
labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#Printing the test images and their prediction
for i in range(5):
  print('prediction of the image : ',labels[y_pred[i]])
  print('actual image :')
  plt.imshow(test_images[i].reshape(28,28))
  plt.show()
