 ==================================================
   Brain Tumor Classification using MobileNetV2
==================================================

📁 PROJECT OVERVIEW:
---------------------
This project uses a deep learning model based on MobileNetV2 for classifying brain tumors into four categories using image data. It includes model training, testing, evaluation, and support for user input to predict tumor class for any custom brain scan image.

🎯 FEATURES:
-------------
- MobileNetV2 with transfer learning and fine-tuning
- Mixed precision training for performance boost
- ImageDataGenerator pipeline for preprocessing
- Evaluation using accuracy, classification report, and confusion matrix
- User input for image prediction with confidence score

📦 DEPENDENCIES:
-----------------
Install dependencies using:
> pip install -r requirements.txt

Main libraries:
- TensorFlow 2.14.0
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Pillow

📂 DATASET STRUCTURE:
----------------------
Make sure your dataset folders follow this structure:

D:\BrainTumor\dataset\
    ├── Training\
    │   ├── Class1\
    │   ├── Class2\
    │   └── ...
    └── Testing\
        ├── Class1\
        ├── Class2\
        └── ...

✍️ HOW TO RUN:
--------------
1. Place your dataset in the path as mentioned above.
2. Run the Python script:
> python brain_tumor_classifier.py

- If a trained model exists (`Optimized_Model.h5`), it will load it.
- If not, the model will be trained and saved.

🖼️ USER IMAGE TESTING:
-----------------------
After the model runs, you'll be prompted to input an image path:
> Enter image path to test (or type 'exit' to quit):

Simply paste the path to your brain scan image (JPG/PNG) and the model will predict the tumor class with confidence.

📈 OUTPUTS:
-----------
- Model accuracy on test data
- Confusion matrix visualization
- Classification report
- Image predictions with confidence

🔐 AUTHOR:
----------
Mannepalli Sasi Kumar
RMK Engineering College - CSE Department
Third Year Student

📬 CONTACT:
-----------
Email: mannepallisasi1234@gmail.com


