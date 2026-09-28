#  AI Product Image Classifier

A deep learning image classification application that identifies clothing, footwear, and fashion-related products from real-world images.

The project uses **MobileNetV2 transfer learning** with TensorFlow/Keras and provides an interactive **Streamlit web application** for real-time predictions.

##  Live Demo

https://ai-appuct-image-classifier.streamlit.app/

##  Features

* Upload JPG, JPEG, or PNG images
* AI-powered product category prediction
* Top 3 predictions with confidence scores
* MobileNetV2 transfer learning
* Data augmentation
* Fine-tuning for improved performance
* Interactive Streamlit interface
* 10 product categories

##  Supported Categories

* Tshirts
* Shirts
* Tops
* Dresses
* Casual Shoes
* Sports Shoes
* Heels
* Watches
* Handbags
* Sunglasses

##  Model

The classifier is based on **MobileNetV2**, a convolutional neural network pretrained on ImageNet.

### Training approach

1. Dataset preparation and category selection
2. Image resizing to 160 × 160 pixels
3. Stratified train/validation/test split
4. Data augmentation
5. Transfer learning using MobileNetV2
6. Fine-tuning of the final layers
7. Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

##  Performance

| Metric           |     Result |
| ---------------- | ---------: |
| Test Accuracy    | **91.17%** |
| Macro F1-score   |   **0.91** |
| Dresses F1-score |   **0.80** |

The final evaluation was performed on **1,336 previously unseen test images**.

### Classification Performance

| Category     | F1-score |
| ------------ | -------: |
| Tshirts      |     0.88 |
| Shirts       |     0.93 |
| Tops         |     0.86 |
| Dresses      |     0.80 |
| Casual Shoes |     0.80 |
| Sports Shoes |     0.85 |
| Heels        |     0.95 |
| Watches      |     1.00 |
| Handbags     |     1.00 |
| Sunglasses   |     1.00 |

##  Dataset

The project uses the **Fashion Product Images (Small)** dataset from Kaggle.

The original dataset contains product metadata and thousands of product images. A subset of 10 categories was selected for this project.

The dataset itself is **not included in this repository** because of its size.

##  Technologies

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pandas
* Scikit-learn
* Pillow
* Matplotlib
* Seaborn
* Streamlit

##  Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd 02-fashion-image-classifier
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

##  Project Structure

```text
02-fashion-image-classifier/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── product_classifier_v2_best.keras
│
├── notebooks/
│   └── real_image_classifier_v2.ipynb
│
├── src/
│
└── data/
    ├── raw/
    └── processed/
```

##  Limitations

The classifier is trained on a specific set of product categories and may produce incorrect predictions for images that are significantly different from the training data.

Some categories contain fewer source images than others, which can affect category-level performance.

##  Future Improvements

* Expand the number of product categories
* Add more diverse real-world images
* Improve performance on visually similar clothing categories
* Experiment with more advanced architectures
* Add image-based product search
* Integrate the classifier into a larger intelligent styling or recommendation system

##  Project

Developed as part of an AI portfolio project exploring computer vision and real-world product classification.
