# Enhanced Digit Recognition CNN (0-10)

A powerful digit recognition application that can identify digits 0-9 and the number "10" using advanced Convolutional Neural Networks. Built with TensorFlow/Keras and tkinter for an interactive user experience.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Features

- **Advanced CNN Architecture**: Multi-layer convolutional neural network with batch normalization and dropout  
- **Multiple Dataset Integration**:  
  - MNIST dataset (70,000 samples)  
  - EMNIST dataset (280,000 additional handwritten digits)  
  - Synthetic digit "10" generation (20,000+ samples)  
- **Interactive Drawing Canvas**: Draw digits with your mouse and get instant predictions  
- **Real-time Training Visualization**: Watch loss and accuracy plots update during training  
- **Confidence Analysis**: Get prediction confidence scores and top-3 predictions  
- **Data Augmentation**: Rotation, shifting, scaling, and shearing for better generalization  
- **Model Management**: Save and load trained models  
- **Professional GUI**: Clean, intuitive interface built with tkinter  

---

## Quick Start

### Prerequisites
- Python 3.8 or higher  
- pip package manager  

### Installation

Install required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python digit_AI.py
```

---

### Quick Start (with Pre-trained Model)

This repository includes a pre-trained model (`digit_model.keras`) so you can start recognizing digits immediately:

1. Clone and install dependencies  
2. Run the application:  

   ```bash
   python digit_AI.py
   ```

3. Click **"Load Model"** and select `digit_model.keras`  
4. Start drawing and predicting right away!  

Or train your own model using **"Train Enhanced Model"**.  

---

## Usage

### Training the Model
1. Launch the application  
2. Configure training parameters:  
   - **Epochs**: Number of training iterations (default: 10)  
   - **Batch Size**: Training batch size (default: 64)  
3. Click **"Train Enhanced Model"** to start training  
4. Watch the real-time training progress in the visualization panel  

> ⏱️ Training typically takes **10–20 minutes** depending on your hardware.  

---

### Making Predictions
- After training is complete, use the drawing canvas on the left  
- Draw any digit (0–9) or the number "10" with your mouse  
- Click **"Predict Digit"** to get the prediction  
- View the results with confidence scores  
- Click **"Clear Canvas"** to draw a new digit  

---

### Model Management
- **Save Model**: Export your trained model for future use  
- **Load Model**: Import a previously trained model  
- **Supported formats**: `.keras`, `.h5`  

---

## Model Architecture

The CNN uses an advanced architecture optimized for digit recognition:

```text
Input Layer (28x28x1)
    ↓
Conv2D (32 filters) + BatchNorm + Conv2D (32) → MaxPool + Dropout
    ↓
Conv2D (64 filters) + BatchNorm + Conv2D (64) → MaxPool + Dropout
    ↓
Conv2D (128 filters) + BatchNorm + Dropout
    ↓
GlobalAveragePooling2D
    ↓
Dense (512) + BatchNorm + Dropout → Dense (256) + BatchNorm + Dropout
    ↓
Output Layer (11 classes: 0-10)
```

**Key Features:**
- Batch normalization for stable training  
- Dropout layers to prevent overfitting  
- Global Average Pooling to reduce parameters  
- Adam optimizer with learning rate scheduling  
- Early stopping to prevent overtraining  

---

## Dataset Information

| Dataset   | Samples   | Description                                  |
|-----------|-----------|----------------------------------------------|
| MNIST     | 70,000    | Standard handwritten digits 0-9              |
| EMNIST    | 280,000   | Extended MNIST with more handwriting styles  |
| Synthetic | 20,000+   | Generated "10" digits with multiple styles   |
| **Total** | 370,000+  | Comprehensive training data                  |

---

## Performance

- **Training Accuracy**: ~99.5%  
- **Validation Accuracy**: ~98.8%  
- **Prediction Speed**: <100ms per digit  
- **Supports**: Digits 0–9 and number "10"  

---

## Technical Details

### Dependencies
- TensorFlow/Keras – Deep learning framework  
- OpenCV – Image processing  
- Matplotlib – Training visualization  
- Pillow (PIL) – Image manipulation  
- NumPy – Numerical computations  
- Scikit-learn – Data preprocessing  
- Requests – Dataset downloading  

### Data Augmentation
- Rotation (±10°)  
- Width/Height shifting (±10%)  
- Shearing transformations  
- Zoom variations  
- Noise injection for synthetic data  

### Synthetic Digit "10" Generation
The application generates diverse "10" digit samples using three styles:
- **Side-by-side**: "1" and "0" positioned horizontally  
- **Overlapped**: Partially overlapping digits  
- **Stacked**: Vertically arranged digits  

Each synthetic sample includes random transformations for variety.  

---

## File Structure

```text
enhanced-digit-recognition-cnn/
├── digit_AI.py                # Main application file
├── digit_model.keras          # Pre-trained model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Directory for saved models (optional)
├── emnist_data/               # EMNIST dataset cache (auto-created)
└── app.log                    # Application logs (auto-created)
```

---

## Troubleshooting

### 1. EMNIST Download Fails
- The app will automatically fall back to MNIST-only training  
- Manual download: Visit EMNIST Official Site  

### 2. Training is Slow
- Reduce batch size or epochs  
- Ensure TensorFlow can access GPU if available  
- Close other resource-intensive applications  

### 3. Canvas Drawing Issues
- Ensure you're drawing with sufficient thickness  
- Center your digit in the canvas  
- Try clearing and redrawing if prediction seems wrong  

### 4. Memory Issues
- Reduce batch size (try 32 instead of 64)  
- Close other applications  
- Ensure at least 8GB RAM available  

---

## Contributing

Contributions are welcome! Here are some ways you can help:
- Report bugs and issues  
- Suggest new features  
- Improve documentation  
- Add support for more digit types  
- Optimize model architecture  
- Add unit tests  

Please feel free to submit pull requests or open issues on GitHub.  

---

## License
This project is licensed under the **MIT License** – see the LICENSE file for details.  

---

## Educational Value

This project demonstrates:
- **Deep Learning**: CNN architecture, training, and optimization  
- **Computer Vision**: Image preprocessing and recognition  
- **GUI Development**: Interactive applications with tkinter  
- **Data Engineering**: Multi-source dataset integration  
- **Software Engineering**: Clean code, error handling, and user experience  

**Perfect for:**
- Machine learning students and enthusiasts  
- Computer vision projects  
- Educational demonstrations  
- Portfolio showcasing  

---

## Support
If you encounter any issues or have questions:
- Check the Issues page  
- Create a new issue with detailed description  
- Include error messages and system information  

---

## Acknowledgments
- MNIST dataset: Yann LeCun et al.  
- EMNIST dataset: NIST Special Database  
- TensorFlow team for the excellent deep learning framework  
- Python community for the amazing ecosystem  
