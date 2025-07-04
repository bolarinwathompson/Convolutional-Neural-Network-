# Convolutional Neural Network - Fruit Classification for ABC Grocery

## Project Overview:
The **ABC Grocery Fruit Classification** project uses a **Convolutional Neural Network (CNN)** to classify images of fruits into six categories: **apple**, **avocado**, **banana**, **kiwi**, **lemon**, and **orange**. The model is trained using a dataset of fruit images and can predict the category of a fruit in a new image.

## Objective:
The primary goal of this project is to train a CNN that classifies fruit images accurately. This system helps ABC Grocery automate the categorization of fruits for inventory management, product recommendations, and other services based on image input. It aids in streamlining processes and improving operational efficiency.

## Key Features:
- **Image Preprocessing**: Product images are resized and normalized for input into the CNN.
- **Data Augmentation**: Techniques such as **rotation**, **zoom**, **shift**, and **flip** are applied to the training data to enhance model robustness.
- **Transfer Learning**: We use **pre-trained VGG16** to extract features from images, which is then fine-tuned for fruit classification.

## Methods & Techniques:

### **1. Basic CNN for Image Classification**:
We start by implementing a **basic CNN architecture**:
- **Convolutional Layers**: To extract features from images.
- **Fully Connected Layers**: To make predictions.
- **Activation**: **ReLU** activation is applied to introduce non-linearity.
- **Softmax**: The output layer uses **softmax** for multi-class classification.

### **2. Dropout for Regularization**:
Dropout is added in the fully connected layers to prevent overfitting. This technique randomly deactivates a fraction of neurons during training, making the model more generalizable.

### **3. Data Augmentation for Better Generalization**:
To increase the diversity of the dataset and improve the model's generalization, we apply the following data augmentation techniques:
- **Rotation**: Random rotations within a specified range.
- **Zooming**: Zooming in or out to simulate various scales.
- **Shifting**: Shifting the images horizontally and vertically.
- **Flipping**: Random horizontal flipping.
- **Brightness adjustment**: Modifying the brightness of the image.

### **4. Transfer Learning with Pre-trained VGG16**:
To improve feature extraction, we use **VGG16**, a pre-trained CNN model on **ImageNet**. By freezing the layers of VGG16 and training only the top layers, we leverage its powerful feature extraction capabilities without needing to train the model from scratch.

### **5. Hyperparameter Tuning with Keras Tuner**:
The hyperparameters of the CNN are optimized using **Keras Tuner** to explore configurations like the number of filters, kernel size, number of dense layers, and dropout rate. This helps improve the model's performance by selecting the best combination of hyperparameters.

### **6. Feature Extraction**:
We use **VGG16** to extract 512-dimensional feature vectors from each image, which represent high-level characteristics like shape, texture, and color. These features are used for classification.

### **7. Image Classification with Nearest Neighbors**:
For the classification task, we use the extracted feature vectors and compare them using **k-Nearest Neighbors** (k-NN). Cosine similarity is used to measure the similarity between the feature vectors of images, making it efficient for classifying unseen fruit images.

### **8. Visualization of Results**:
The model’s performance is visualized by plotting training and validation **accuracy** and **loss** over epochs. Confusion matrices are used to evaluate how well the model classifies each fruit category.

---

## Technologies Used:
- **Python**: Programming language for implementing the entire classification system.
- **TensorFlow/Keras**: For building and training the CNN model and utilizing **VGG16**.
- **NumPy**: For handling arrays and performing mathematical operations.
- **scikit-learn**: For implementing the **Nearest Neighbors** algorithm and evaluating the model.
- **matplotlib**: For visualizing training performance and results.
- **pickle**: For saving the model and feature vectors for future use.

## Key Results & Outcomes:
- The CNN model successfully classifies fruit images into six categories with high accuracy.
- Data augmentation and **transfer learning** have significantly improved the model's robustness and generalization.
- The model achieved **95% accuracy** on the validation set after fine-tuning with **Keras Tuner**.

## Lessons Learned:
- **Transfer learning** is highly effective for tasks with limited data and helps leverage powerful pre-trained models.
- **Dropout** and **data augmentation** are essential for preventing overfitting and improving the model’s generalization.
- Hyperparameter tuning can significantly boost the performance of deep learning models.

## Future Enhancements:
- **Model Fine-Tuning**: Fine-tuning VGG16 for better accuracy on fruit-specific features.
- **Advanced Models**: Explore advanced architectures like **ResNet** or **Inception** for even better accuracy.
- **Real-Time Deployment**: Deploy the trained model for real-time fruit classification in ABC Grocery’s inventory system.

