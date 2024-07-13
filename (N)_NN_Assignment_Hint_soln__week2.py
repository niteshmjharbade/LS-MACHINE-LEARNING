import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Rescaling
from tensorflow.keras import Input

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\nites\Desktop\ML(learner's space)\Machine-Learning-LS-24\week 2 asss\NN\homer_bart",
    image_size=(64, 64),
    label_mode="binary"
)

# Split dataset into training and testing
train_data = dataset.take(8)
test_data = dataset.skip(8)
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# preprocessing layer
preprocess = tf.keras.Sequential([
    Rescaling(1./255)  
])

# Define the model
model = tf.keras.Sequential([
    Input((64, 64, 3)),
    preprocess,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
model.fit(train_data,
          epochs=40,
          batch_size=32,
          verbose=1,
          validation_data=test_data)

test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Accuracy: {test_accuracy}')