import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.optimizers import Adamax # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import warnings
from tensorflow.keras.mixed_precision import set_global_policy # type: ignore
from tensorflow.keras.utils import get_custom_objects # type: ignore
from tensorflow.keras.layers import Layer # type: ignore
import tensorflow as tf

warnings.filterwarnings('ignore')
set_global_policy('mixed_float16')  # Enable mixed precision training

# Define a custom Cast layer class
class CastLayer(Layer):
    def call(self, inputs):
        return tf.cast(inputs, dtype=tf.float32)

# Register the custom layer
get_custom_objects().update({"Cast": CastLayer})

# Load dataset directories
train_dir = r"D:\BrainTumor\dataset\Training"
test_dir = r"D:\BrainTumor\dataset\Testing"

# Function to create dataframe from directory
def create_dataframe(directory):
    image_paths, image_labels = [], []
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        for image in os.listdir(category_path):
            image_paths.append(os.path.join(category_path, image))
            image_labels.append(category)
    return pd.DataFrame({'filepaths': image_paths, 'labels': image_labels})

train_df = create_dataframe(train_dir)
test_df = create_dataframe(test_dir)

# Split training data into train and validation sets
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Image Data Generator
image_gen = ImageDataGenerator(rescale=1/255)

gen_train = image_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                          target_size=(128,128), class_mode='categorical', batch_size=32)
gen_valid = image_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                          target_size=(128,128), class_mode='categorical', batch_size=32)
gen_test = image_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                         target_size=(128,128), class_mode='categorical', batch_size=32, shuffle=False)

# ✅ FIXED: No need to apply prefetch
train_ds = gen_train  
valid_ds = gen_valid
test_ds = gen_test

# Check if model already exists
model_path = 'Optimized_Model.h5'
if os.path.exists(model_path):
    print("\n✅ Loading pre-trained model...")
    model = load_model(model_path, custom_objects={"Cast": CastLayer})
else:
    print("\n🚀 Training new model...")
    # Load Pretrained Model (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    base_model.trainable = False  # Freeze the base model
    
    # Custom Layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)  # Reduced Dropout
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)  # Reduced Dropout
    output_layer = Dense(4, activation='softmax')(x)

    # Compile Model
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Reduced patience

    # Train the Model (Reduced Epochs)
    history = model.fit(train_ds, validation_data=valid_ds, epochs=20, callbacks=[early_stopping])

    # Save Model
    model.save(model_path)
    print("\n✅ Model saved as 'Optimized_Model.h5'")

# Evaluate Model
loss, accuracy = model.evaluate(test_ds)
print(f"\n✅ Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# Predictions
predictions = np.argmax(model.predict(test_ds), axis=1)
print(classification_report(gen_test.classes, predictions))

# Confusion Matrix
sns.heatmap(confusion_matrix(gen_test.classes, predictions), annot=True, fmt='d', cmap='crest')
plt.title('Confusion Matrix')
plt.show()

# 🔥 User Input Image Testing (Loop Until Exit)
class_labels = list(gen_train.class_indices.keys())  # Get class names

def preprocess_image(image_path):
    """Preprocesses an image to match the model's expected input format."""
    img = load_img(image_path, target_size=(128, 128))  # Resize
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch
    return img_array

while True:
    user_input = input("\nEnter image path to test (or type 'exit' to quit): ")
    
    if user_input.lower() == "exit":
        print("Exiting the program... 🚪")
        break
    
    if not os.path.exists(user_input):
        print("❌ Error: File not found! Please enter a valid image path.")
        continue
    
    # Preprocess and predict
    image = preprocess_image(user_input)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    
    plt.imshow(load_img(user_input))
    plt.title(f"Predicted: {class_labels[predicted_class]} \n Confidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()
    
    print(f"✅ Predicted Class: {class_labels[predicted_class]}")
    print(f"🔍 Confidence: {confidence:.2f}%")
