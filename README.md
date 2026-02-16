# 🦁 AnimalPedia: AI-Powered Animal Recognition & Information System


## 🎯 Overview

**AnimalPedia** is an intelligent web application that combines computer vision and information retrieval to identify animal species from images and provide comprehensive educational information. Built with state-of-the-art deep learning techniques, this project demonstrates the practical application of transfer learning using EfficientNet-B0 architecture for multi-class image classification across 90 distinct animal species.

The system serves dual purposes:
1. **AI-Powered Recognition**: Upload an image to automatically identify the animal species
2. **Educational Resource**: Search and explore detailed information about various animals including their scientific classification, habitat, and fascinating facts

---

## 🔍 Problem Statement

### Challenges Addressed

1. **Wildlife Identification Complexity**: Manual animal identification requires extensive zoological knowledge and can be time-consuming, especially for non-experts, students, and wildlife enthusiasts.

2. **Educational Accessibility**: Traditional resources for learning about animals are often scattered across multiple platforms, making it difficult to access consolidated, reliable information quickly.

3. **Image Recognition at Scale**: Classifying images across numerous similar-looking species (90 different animals) presents a challenging multi-class classification problem with high visual similarity between certain species (e.g., different big cats, various bird species).

4. **Real-world Application Gap**: There's a need for practical, user-friendly tools that bridge the gap between academic deep learning research and everyday wildlife education and conservation efforts.

---

## 💡 Solution

AnimalPedia addresses these challenges through:

### 1. **Advanced Deep Learning Architecture**
- Implements **EfficientNet-B1**, a state-of-the-art convolutional neural network optimized for efficiency and accuracy
- Utilizes **transfer learning** with ImageNet pre-trained weights, leveraging knowledge from millions of images
- Achieves **87% overall accuracy** across 90 animal species with optimized hyperparameters

### 2. **Robust Data Augmentation Pipeline**
```python
- Random horizontal flipping (50% probability)
- Random rotation (±30 degrees)
- Color jittering (brightness, saturation, contrast)
- Random affine transformations
- Random perspective distortion
- Random erasing for improved generalization
```

### 3. **User-Centric Design**
- **Dual-interface approach**: Both image upload and text search capabilities
- **Top-2 predictions**: Provides context when the model is uncertain
- **Rich information display**: Combines AI predictions with curated educational content
- **Responsive web interface**: Built with Streamlit for seamless user experience

### 4. **Practical Implementation**
- **Real-time inference**: Quick image processing and prediction
- **Comprehensive database**: 90 animals with detailed information (scientific names, habitats, fun facts)
- **Scalable architecture**: Modular design allows easy expansion to more species

---

## ✨ Key Features

### 🤖 AI-Powered Recognition
- **Upload & Identify**: Drag-and-drop image upload for instant animal recognition
- **Multi-Prediction Display**: Shows top 2 most likely species with confidence rankings
- **Detailed Information**: Automatically retrieves and displays comprehensive details for predicted species

### 🔍 Text-Based Search
- **Free-Text Query**: Search animals by common name (e.g., "lion", "eagle", "dolphin")
- **Smart Matching**: Case-insensitive search with whitespace handling
- **90+ Species Database**: Extensive collection covering mammals, birds, reptiles, insects, and marine life

### 📚 Educational Content
Each animal profile includes:
- **Common Name & Scientific Name**: Proper taxonomic classification
- **Habitat Information**: Geographic distribution and environmental preferences
- **Fun Facts**: Engaging trivia to enhance learning
- **Visual Reference**: Representative images for educational purposes

### 🎨 User Interface
- **Clean, Modern Design**: Intuitive layout with custom CSS styling
- **Responsive Columns**: Side-by-side display of search and upload features
- **Visual Feedback**: Success/error messages, loading indicators, and emojis for engagement
- **Image Preview**: Display uploaded images before processing

---

## 🛠️ Technology Stack

### Deep Learning & Computer Vision
| Technology | Purpose | Version |
|------------|---------|---------|
| **PyTorch** | Deep learning framework | 2.0+ |
| **torchvision** | Pre-trained models & transforms | Latest |
| **EfficientNet-B1** | CNN architecture | Pre-trained |
| **PIL (Pillow)** | Image processing | Latest |

### Web Application
| Technology | Purpose | Version |
|------------|---------|---------|
| **Streamlit** | Web framework | 1.28+ |
| **Python** | Core language | 3.12+ |

### Data Processing & Utilities
| Technology | Purpose |
|------------|---------|
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **Matplotlib/Seaborn** | Visualization |
| **splitfolders** | Dataset organization |
| **scikit-learn** | Metrics & evaluation |

### Training Optimization
- **Mixed Precision Training** (AMP): Faster training with reduced memory usage
- **Adam Optimizer**: Adaptive learning rate optimization
- **Cross-Entropy Loss**: Multi-class classification objective

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                      (Streamlit App)                         │
├──────────────────────┬──────────────────────────────────────┤
│   Image Upload       │        Text Search                    │
│   Component          │        Component                      │
└──────────┬───────────┴────────────┬─────────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────────┐  ┌─────────────────────────┐
│  Image Processing    │  │   Database Query        │
│  & Preprocessing     │  │   (JSON Lookup)         │
└──────────┬───────────┘  └────────┬────────────────┘
           │                       │
           ▼                       │
┌──────────────────────┐          │
│  EfficientNet-B1     │          │
│  Model Inference     │          │
│  (90 Classes)        │          │
└──────────┬───────────┘          │
           │                       │
           └───────────┬───────────┘
                       ▼
           ┌───────────────────────┐
           │  Information Display   │
           │  - Predictions         │
           │  - Details             │
           │  - Images              │
           └───────────────────────┘
```

### Data Flow

1. **Input Stage**: User provides image or search query
2. **Processing Stage**:
   - **Image Path**: Resize → Normalize → Tensor Conversion → Model Inference
   - **Search Path**: Text normalization → JSON database lookup
3. **Inference Stage**: Model generates top-2 predictions with probability scores
4. **Output Stage**: Display predictions, retrieve details, render comprehensive information

---
### Data Sources
The model was trained on a carefully curated dataset comprising 90 animal species:

-  Images: Kaggle
-  Botanical Information: Wikipedia API

### Data Collection & Preprocessing
Image Acquisition:The dataset was compiled from publicly available sources, including:

- **Open-source image repositories**
- **Wildlife photography databases**
- **Educational and scientific collections**
- **Creative Commons licensed images**

## 📊 Model Performance

### Overall Metrics
```
Accuracy:  87%
Macro F1:  0.87
Weighted F1: 0.87
```

### Training Configuration
- **Dataset Split**: 60% train / 20% validation / 20% test
- **Total Samples**: 5,400 images (3,240 train / 1,080 val / 1,080 test)
- **Batch Size**: 32
- **Input Resolution**: 224×224 pixels
- **Classes**: 90 animal species

### Top Performing Classes (F1-Score: 1.00)
- Chimpanzee, Crab, Fly, Gorilla, Lion, Panda
- Penguin, Sparrow, Woodpecker, Zebra

### Challenging Classes (F1-Score < 0.70)
- **Whale** (0.58): High intra-class variation in underwater/surface poses
- **Dolphin** (0.61): Visual similarity with other marine mammals
- **Cow** (0.64): Confusion with similar livestock (ox, buffalo)
- **Goat** (0.67): Overlap features with sheep and deer
- **Moth** (0.67): High similarity with butterflies in certain poses

### Model Strengths
1. ✅ **Excellent on distinctive species**: Big cats, primates, iconic animals
2. ✅ **Strong insect recognition**: 92%+ accuracy on most insect classes
3. ✅ **Robust to image augmentation**: Handles rotation, lighting, perspective changes
4. ✅ **Marine life classification**: Good performance on aquatic species (except cetaceans)

### Model Weaknesses
1. ⚠️ **Inter-species similarity**: Struggles with visually similar animals (livestock, cetaceans)
2. ⚠️ **Pose variation**: Performance drops with unusual viewing angles
3. ⚠️ **Environmental context**: May confuse animals in similar habitats
4. ⚠️ **Subspecies distinction**: Cannot differentiate between subspecies of the same animal

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.12 or higher
pip (Python package manager)
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/animalpedia.git
cd animalpedia
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model & Data
Ensure the following files are present:
```
├── efficient_best_mine.pth          # Trained model weights
├── animals_info.json                # Animal database
└── animal_images/                   # Reference images
    ├── lion.jpg
    ├── elephant.png
    └── ...
```

### Step 5: Verify Installation
```bash
python -c "import torch; import streamlit; print('Installation successful!')"
```

---

## 📖 Usage

### Running the Application

#### Method 1: Streamlit Run Command
```bash
streamlit run app.py
```

#### Method 2: Python Module
```bash
python -m streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`

### Using the Application

#### 🖼️ Image Recognition Workflow
1. Navigate to the **"Upload Animal Image"** section
2. Click **"Browse files"** or drag-and-drop an image (JPG, JPEG, PNG)
3. Wait for AI processing (~1-2 seconds)
4. Review **top 2 predictions** with detailed information
5. Explore educational content for both predictions

#### 🔍 Search Workflow
1. Go to **"Search Animal by Name"** section
2. Type the animal's common name (e.g., "tiger", "penguin")
3. Click **"Search"** or press Enter
4. View comprehensive species information

### API Usage (For Developers)

```python
from PIL import Image
from app import load_model, predict_animal

# Load the model
model = load_model()

# Open an image
image = Image.open("path/to/animal.jpg")

# Get predictions
predictions = predict_animal(image, model)

# Access results
for i, pred in enumerate(predictions, 1):
    print(f"{i}. {pred['animal']}: {pred['confidence']:.2%}")
```

---

## ⚠️ Limitations

### Technical Limitations

1. **Model Constraints**
   - **Fixed Input Size**: Requires 224×224 pixel images (automatically resized)
   - **Species Coverage**: Limited to 90 pre-trained species
   - **No Subspecies Recognition**: Cannot distinguish between subspecies or breeds
   - **Single Animal Detection**: Works best with images containing one primary animal

2. **Performance Considerations**
   - **CPU Inference**: Currently optimized for CPU; GPU acceleration not implemented
   - **Processing Time**: 1-3 seconds per image on standard hardware
   - **Memory Usage**: ~500MB for model loading
   - **Concurrent Users**: Not optimized for high-traffic production deployment

3. **Data Limitations**
   - **Training Data Bias**: Performance reflects the diversity of training images
   - **Image Quality Dependency**: Best results with clear, well-lit, centered subjects
   - **Background Complexity**: Cluttered backgrounds may reduce accuracy
   - **Occlusion Issues**: Partially hidden animals harder to classify
