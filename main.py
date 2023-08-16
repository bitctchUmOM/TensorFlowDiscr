import tensorflow as tf
#creating the callback function.
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        #gathering lgos from callback keras class.
        if (logs.get('accuracy') > 0.95):
            print("\n Reached 95% accuracy so cancelling training!")
            self.model.stop_training = True

#creating object of myCallBack class.
callbacks = myCallBack()
mist = tf.keras.datasets.fashion_mnist

#creating variables to contain of mstnn data.
((training_images, training_labels),
(test_images, test_labels)) = mist.load_data()

#optimizing the images pixels.
training_images = training_images/255.0
test_images = test_images / 255.0

#creating the model.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])


