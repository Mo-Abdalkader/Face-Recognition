# 🎯 AI Face Recognition System

[![Streamlit Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Deep learning-powered face recognition system with dual interfaces: **Streamlit web app** for instant demos and **CustomTkinter desktop app** for full database management.

---

## 🚀 Quick Start

### Try the Live Demo
**[Launch Streamlit App →](https://face-similarity-recognition.streamlit.app/)**

### Run Locally
```bash
git clone https://github.com/Mo-Abdalkader/face-recognition-system.git
cd face-recognition-system
pip install -r requirements.txt

# Web Interface
streamlit run streamlit_app.py

# Desktop App (requires model training first)
python gui/main_app.py
```

---

## ✨ Key Features

### Web App (Streamlit) - Demo Ready
- 🔍 **Two-Image Comparison** - Compare any two faces with similarity scores
- 📊 **Multi-Face Detection** - Detect & analyze all faces in group photos
- 🔄 **Batch Processing** - Compare one face against multiple images
- 📈 **Visual Analytics** - Interactive heatmaps and embedding visualizations
- ⬇️ **Export Results** - Download comparison images and CSV reports

### Desktop App (CustomTkinter) - Production Ready
- 💾 **SQLite Database** - Store and manage face records permanently
- 👥 **Person Management** - Add/edit profiles with metadata (name, department, role)
- 📜 **Recognition History** - Track all recognition attempts with timestamps
- 📊 **Statistics Dashboard** - Comprehensive analytics and insights
- ⚡ **Optimized Performance** - 50ms inference on GPU, async initialization

---

## 🧠 Technical Architecture

### Model
- **Hybrid Encoder**: GoogleNet (Inception v1) + ResNet-18 fusion
- **Output**: 512-dimensional L2-normalized embeddings
- **Loss Function**: Triplet Loss with margin 0.5
- **Accuracy**: 95%+ on validation set

### Training Pipeline
```
VGGFace2 Dataset → MTCNN Face Detection → Data Augmentation 
→ Hybrid Encoder → Triplet Loss → 50 Epochs → Trained Model
```

### Core Components
| Component | Purpose |
|-----------|---------|
| `models/hybrid_encoder.py` | Neural network architecture |
| `data/face_cropper.py` | MTCNN-based face detection |
| `inference/embedding_generator.py` | Generate face embeddings |
| `inference/face_recognizer.py` | Match faces against database |
| `database/face_database.py` | SQLite data management |
| `streamlit_app.py` | Web interface |
| `gui/main_app.py` | Desktop application |

---

## 📊 Model Training

```bash
# 1. Prepare VGGFace2 dataset
# Download from: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

# 2. Update paths in config.py
TRAIN_DIR = "/path/to/vggface2/train"
VAL_DIR = "/path/to/vggface2/val"

# 3. Train the model
python main_train.py

# 4. Model saved to: files/model.pth
```

**Training Specs:**
- Dataset: VGGFace2 (9,000+ identities)
- Augmentation: Random flip, rotation, color jitter
- Optimizer: AdamW with ReduceLROnPlateau
- Hardware: GPU recommended (3-5 hours on RTX 3090)
- Batch Size: 64

---

## 🎨 How It Works

### 1. Face Detection (MTCNN)
Detects faces with 99%+ accuracy, extracts bounding boxes, filters by confidence.

### 2. Feature Extraction (Hybrid Encoder)
Converts face images into 512-dimensional embedding vectors that capture unique facial characteristics.

### 3. Similarity Calculation (Cosine Distance)
Measures how similar two faces are using cosine similarity on embeddings (0 = different, 1 = identical).

### 4. Recognition (Threshold Matching)
Faces with similarity ≥ threshold (default 0.6) are considered matches.

---

## 📂 Project Structure

```
face-recognition-system/
├── streamlit_app.py          # Web interface (demo)
├── config.py                 # Central configuration
├── requirements.txt          # Dependencies
│
├── models/
│   └── hybrid_encoder.py     # Neural network
│
├── inference/
│   ├── embedding_generator.py
│   └── face_recognizer.py
│
├── data/
│   ├── face_cropper.py       # MTCNN detection
│   ├── transforms.py         # Preprocessing
│   └── dataset.py            # VGGFace2 loader
│
├── database/
│   └── face_database.py      # SQLite manager
│
├── gui/
│   ├── main_app.py           # Desktop app
│   └── panels/               # UI components
│
├── training/
│   ├── trainer.py            # Training pipeline
│   ├── losses.py             # Triplet loss
│   └── metrics.py            # Evaluation metrics
│
└── files/
    └── model.pth             # Trained weights
```

---

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Model
EMBEDDING_DIM = 512
IMAGE_SIZE = 224

# Recognition
RECOGNITION_THRESHOLD = 0.6  # Adjust sensitivity
FACE_MIN_CONFIDENCE = 0.9    # Detection quality

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
```

---

## 🖥️ Desktop vs Web

| Feature | Desktop | Web |
|---------|---------|-----|
| Face Comparison | ✅ | ✅ |
| Database Storage | ✅ | ❌ |
| User Management | ✅ | ❌ |
| History Tracking | ✅ | ❌ |
| Cloud Deployment | ❌ | ✅ |
| Installation Required | ✅ | ❌ |

**Use Web** for: Quick demos, sharing results, cloud access  
**Use Desktop** for: Production systems, employee databases, offline use

---

## 📈 Performance

- **Inference Speed**: 50ms (GPU) / 200ms (CPU)
- **Model Size**: ~100MB
- **Face Detection**: 99%+ accuracy (frontal faces)
- **Recognition Accuracy**: 95%+ (validation set)
- **Database Capacity**: 1000+ faces (tested)

---

## 🛠️ Development

### Add Face to Database
```python
from database.face_database import FaceDatabase
from inference.embedding_generator import FaceEmbeddingGenerator

db = FaceDatabase('face_recognition.db')
db.add_person('EMP001', 'John Doe', department='Engineering')

embedding = embedding_generator.generate_embedding('john.jpg')
db.add_embedding('EMP001', embedding, 'john.jpg')
```

### Recognize Face
```python
from inference.face_recognizer import FaceRecognizer

recognizer = FaceRecognizer(
    db_path='face_recognition.db',
    embedding_generator=embedding_generator,
    threshold=0.6
)

result = recognizer.recognize_face('test.jpg')
if result['recognized']:
    print(f"Match: {result['person_info']['name']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- VGGFace2 dataset team
- PyTorch & Streamlit communities
- MTCNN face detection

---

## 📞 Contact

**Mohamed Abdalkader** - [@your_linkedin](https://linkedin.com/in/Mo-Abdalkader)
---

**⭐ Star this repo if you find it useful!**

---

*Built with PyTorch, Streamlit, and CustomTkinter*
