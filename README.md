# Convolutional Neural Network (CNN) - Image Search Engine for ABC Grocery

## Project Overview:
The **ABC Grocery Image Search Engine** project utilizes a **Convolutional Neural Network (CNN)**, specifically a pre-trained **VGG16 model**, to create a fruits classification system for ABC Grocery’s product catalog. The goal is to find similar products based on a given query image, allowing for efficient product discovery and recommendation. By leveraging **transfer learning**, the VGG16 model extracts high-level feature vectors that represent the content of product images, enabling the system to compare and retrieve similar products.

## Objective:
The primary goal of this project is to build an **image retrieval system** that helps ABC Grocery customers find similar products based on an image input. This system enhances the shopping experience by allowing customers to visually search for products similar to those they are interested in, improving customer engagement and satisfaction.

## Key Features:
- **Feature Extraction with Pre-trained CNN**: The project leverages **VGG16**, a popular CNN model pre-trained on a large image dataset (ImageNet), as a feature extractor. By excluding the top layers of VGG16 and using its convolutional layers, we extract 512-dimensional feature vectors from each product image in ABC Grocery’s catalog.
- **Image Preprocessing**: 
  - **Resizing**: Product images are resized to a consistent dimension of **224x224 pixels**, which is required by the VGG16 model.
  - **Normalization**: Pixel values are scaled to the range [0, 1] by dividing them by 255, ensuring uniform input for the CNN.
  - **Data Augmentation**: Minor image transformations (such as random rotations and flips) are applied to improve the robustness of the feature extraction process and to simulate diverse customer-uploaded images.
- **Similarity Search**: Using the extracted feature vectors, the project implements a **nearest neighbor** search mechanism to identify the most similar products to a given query image. The **cosine distance metric** is employed to measure the similarity between feature vectors, ensuring an efficient and accurate comparison.

## Methods & Techniques:

### **1. Basic CNN for Image Classification**:
In the first part of the project, we implement a **basic CNN** to classify products into different categories. The architecture includes:
- **Convolutional Layers**: Two convolutional layers with **ReLU** activations followed by **MaxPooling** layers.
- **Fully Connected Layers**: A flattening layer followed by dense layers to output the probability distribution over product categories.
- **Loss Function**: **Categorical Cross-Entropy** loss for multi-class classification.
- **Optimizer**: **Adam** optimizer for efficient model training.

### **2. Dropout for Regularization**:
To mitigate overfitting and improve the generalization ability of the model, we add **dropout** in the fully connected layer. This randomly deactivates a fraction of neurons during training to prevent over-reliance on specific features, improving the model’s robustness.

### **3. Data Augmentation for Better Generalization**:
To further improve model performance and robustness, **data augmentation** techniques are employed:
- **Rotation**, **zoom**, **width/height shift**, **horizontal flip**, and **brightness range** are applied on the training dataset to create more diverse images.
- These augmentations help the model learn invariances in the data, making it less sensitive to variations in the input images.

### **4. Transfer Learning with Pre-trained VGG16**:
To extract better features from images, **transfer learning** is employed using the pre-trained **VGG16 model**, which is already trained on the large **ImageNet dataset**. By using this model without the top layers (i.e., excluding the fully connected layers), we can extract rich feature representations from images without the need for extensive training from scratch.

- **Freezing Layers**: All layers from the pre-trained VGG16 are frozen to retain the learned features, and only the top layers are trained to suit the specific task of fruit classification for ABC Grocery.

### **5. Hyperparameter Tuning with Keras Tuner**:
To optimize the performance of the CNN, **hyperparameter tuning** is performed using **Keras Tuner**. Key parameters such as the number of filters, kernel size, number of layers, and dropout rate are explored to identify the best model configuration for the task.

- **Random Search** is used to search through the hyperparameter space efficiently.
- The best combination of hyperparameters is determined based on validation accuracy.

### **6. Feature Extraction**:
In this step, we used the **VGG16 model** to extract **512-dimensional feature vectors** from each image. These feature vectors represent the high-level characteristics of the products, such as color, shape, and texture. By using these feature vectors, we can compare the similarity of images efficiently.

### **7. Image Search with Nearest Neighbors**:
Once the features are extracted, we implement a **nearest neighbor search**. For a given query image, we calculate its feature vector using the VGG16 model and compare it against the feature vectors of other product images in the catalog using **cosine similarity**. This allows the system to find the most visually similar products.

### **8. Visualization of Results**:
Finally, the **image search engine** visualizes the search results by displaying the most similar product images to the query image along with their **cosine similarity scores**. This provides customers with a clear and intuitive display of the most relevant products based on their input image.

---

## Technologies Used:
- **Python**: Programming language for implementing the entire image search engine.
- **TensorFlow/Keras**: For implementing the **VGG16** model, performing feature extraction, and utilizing CNN layers for feature extraction.
- **NumPy**: For handling arrays and mathematical operations during the feature extraction and comparison process.
- **scikit-learn**: Implements the **Nearest Neighbors** algorithm and the cosine similarity metric for efficient image search.
- **matplotlib**: Used for visualizing the search results and displaying images and similarity scores in a user-friendly format.
- **pickle**: Used for saving and loading the feature vectors and pre-trained models, allowing for easy deployment and reuse.

## Key Results & Outcomes:
- The **Image Search Engine** successfully identifies and ranks similar products based on the input image. By utilizing the **VGG16 CNN model**, the system captures intricate visual features and enables high-accuracy image retrieval.
- The system's performance is validated by testing the similarity search functionality, ensuring that products with similar content are returned first.
- **Cosine similarity** ensures the efficiency of the search, producing relevant and reliable results based on the similarity between feature vectors.

## Lessons Learned:
- **Transfer Learning** with pre-trained models like **VGG16** is highly effective for tasks such as image feature extraction, significantly reducing the need for training a model from scratch.
- **Feature representation** through CNNs captures intricate details that are vital for performing accurate image similarity searches.
- **Nearest Neighbors** is a simple yet powerful technique for searching through high-dimensional spaces, providing a scalable solution for content-based image retrieval.

## Future Enhancements:
- **Model Fine-tuning**: Fine-tuning the VGG16 model by adding custom layers on top of the pre-trained network could improve the accuracy of feature extraction specific to ABC Grocery's product images.
- **Efficient Search Algorithms**: Implementing more efficient algorithms like **Approximate Nearest Neighbors** (ANN) could further improve search speed, especially for large product catalogs.
