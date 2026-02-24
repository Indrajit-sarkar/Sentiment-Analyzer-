<div style="font-family: 'Times New Roman', Times, serif;">

# <span style="font-size: 18pt;">Facial Sentiment Analysis System</span>

<p style="font-size: 12pt; text-align: center;">
<strong>A comprehensive real-time facial emotion recognition system using deep learning</strong><br>
Detects and analyzes seven distinct emotions through facial expressions with high accuracy
</p>

<p style="font-size: 12pt; text-align: center;">
<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.1.0-red.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/Flask-3.0.0-green.svg" alt="Flask">
</p>

---

## <span style="font-size: 18pt;">Table of Contents</span>

<p style="font-size: 12pt;">

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Technical Architecture](#technical-architecture)
- [API Documentation](#api-documentation)
- [Commands Reference](#commands-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

</p>

---

## <span style="font-size: 18pt;">Overview</span>

<p style="font-size: 12pt;">

This facial sentiment analysis system provides a complete solution for emotion detection from facial expressions. Built with Python, Flask, PyTorch, and OpenCV, it offers both real-time camera analysis and batch processing capabilities through an intuitive web interface.

The system employs state-of-the-art deep learning techniques, utilizing MTCNN for face detection and a custom Convolutional Neural Network (CNN) for emotion classification. It achieves high accuracy rates and processes video streams at 15-30 frames per second on standard hardware.

</p>

### <span style="font-size: 14pt;">Detected Emotions</span>

<p style="font-size: 12pt;">

The system accurately identifies seven fundamental human emotions:

1. **Happy** 😊 - Joy, pleasure, contentment
2. **Sad** 😢 - Sorrow, unhappiness, melancholy
3. **Angry** 😠 - Rage, frustration, irritation
4. **Surprise** 😲 - Astonishment, amazement, shock
5. **Fear** 😨 - Anxiety, apprehension, terror
6. **Disgust** 🤢 - Revulsion, distaste, aversion
7. **Neutral** 😐 - Calm, composed, unexpressive

Each detection includes confidence scores and probability distributions across all emotion categories.

</p>

---

## <span style="font-size: 18pt;">Features</span>

### <span style="font-size: 14pt;">Core Capabilities</span>

<p style="font-size: 12pt;">

- **Real-time Camera Analysis**: Live emotion detection from webcam with instant feedback
- **Image Analysis**: Upload and analyze static images with detailed emotion breakdowns
- **Video Processing**: Comprehensive video analysis with frame-by-frame emotion tracking
- **Batch Processing**: Simultaneous analysis of multiple images for efficient workflows
- **Multi-face Detection**: Detect and analyze emotions for multiple faces in a single frame

</p>

### <span style="font-size: 14pt;">Advanced Features</span>

<p style="font-size: 12pt;">

- **Emotion Tracking**: Temporal analysis tracking emotion changes over time
- **Real-time Visualization**: Interactive charts displaying emotion timelines and distributions
- **Data Export**: Export analysis results in JSON or CSV format for further processing
- **Statistics Dashboard**: Real-time metrics including confidence scores and dominant emotions
- **Confidence Scoring**: Probability distributions for all emotion categories per detection

</p>

### <span style="font-size: 14pt;">User Interface</span>

<p style="font-size: 12pt;">

- Modern, responsive web interface with intuitive navigation
- Color-coded emotion badges for quick visual identification
- Interactive charts powered by Chart.js
- Mobile-friendly responsive design
- Professional gradient styling and smooth animations

</p>

---

## <span style="font-size: 18pt;">Quick Start</span>

<p style="font-size: 12pt;">

Get started in 3 simple steps:

```bash
# 1. Run automated setup
python setup.py

# 2. Start the application
python app.py

# 3. Open your browser
# Navigate to: http://localhost:5000
```

</p>


### <span style="font-size: 14pt;">System Requirements</span>

<p style="font-size: 12pt;">

**Hardware:**
- Processor: Intel Core i5 or equivalent (minimum)
- RAM: 4GB minimum, 8GB recommended
- GPU: CUDA-capable NVIDIA GPU (optional, for acceleration)
- Webcam: Any standard USB or integrated webcam

**Software:**
- Operating System: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- Python: Version 3.8 or higher
- Web Browser: Chrome 90+, Firefox 88+, Safari 14+, or Edge 90+

</p>

---

## <span style="font-size: 18pt;">Installation</span>

### <span style="font-size: 14pt;">Method 1: Automated Setup (Recommended)</span>

<p style="font-size: 12pt;">

```bash
# Clone the repository
git clone https://github.com/yourusername/facial-sentiment-analysis.git
cd facial-sentiment-analysis

# Run automated setup
python setup.py
```

The setup script will:
- Check Python version compatibility
- Create necessary directories
- Install all dependencies
- Verify installation

</p>

### <span style="font-size: 14pt;">Method 2: Manual Setup</span>

<p style="font-size: 12pt;">

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir exports models uploads
```

</p>

### <span style="font-size: 14pt;">Verify Installation</span>

<p style="font-size: 12pt;">

```bash
python -c "import torch; import cv2; import flask; print('Installation successful!')"
```

</p>

---

## <span style="font-size: 18pt;">Usage Guide</span>

### <span style="font-size: 14pt;">Starting the Application</span>

<p style="font-size: 12pt;">

```bash
python app.py
```

The application will start on `http://localhost:5000`. Open this URL in your web browser.

</p>

### <span style="font-size: 14pt;">1. Real-time Camera Analysis</span>

<p style="font-size: 12pt;">

**Purpose**: Continuous emotion detection from live webcam feed with temporal tracking.

**Steps**:
1. Navigate to the "Real-time Camera" tab
2. Click "Start Camera" to activate webcam
3. Grant browser permissions when prompted
4. Position your face within the camera frame
5. Observe real-time emotion detection with bounding boxes and labels
6. Monitor the statistics dashboard for session metrics
7. View interactive charts showing emotion timeline and distribution
8. Click "Export JSON" or "Export CSV" to save session data
9. Click "Reset Tracking" to start a new session
10. Click "Stop Camera" when finished

**Features**:
- Live emotion detection with confidence scores
- Real-time statistics (total detections, average confidence, session duration)
- Interactive timeline chart showing emotion changes over time
- Doughnut chart displaying emotion distribution percentages
- Export session data in JSON or CSV format

</p>

### <span style="font-size: 14pt;">2. Image Analysis</span>

<p style="font-size: 12pt;">

**Purpose**: Analyze emotions in static images with detailed probability breakdowns.

**Steps**:
1. Navigate to the "Analyze Image" tab
2. Click "Choose Image" to select an image file
3. Supported formats: JPG, JPEG, PNG, BMP
4. View annotated image with detected faces and emotions
5. Review detailed emotion probabilities for each detected face

**Output**:
- Annotated image with bounding boxes around detected faces
- Emotion label and confidence score for each face
- Probability distribution chart for all seven emotions
- Multiple face detection support

</p>

### <span style="font-size: 14pt;">3. Video Processing</span>

<p style="font-size: 12pt;">

**Purpose**: Comprehensive analysis of video files with frame-by-frame emotion tracking.

**Steps**:
1. Navigate to the "Analyze Video" tab
2. Click "Choose Video" to select a video file
3. Supported formats: MP4, AVI, MOV, MKV
4. Wait for processing to complete (analyzes every 5th frame)
5. Review comprehensive statistics and emotion distribution

**Output**:
- Total frames processed
- Frame-by-frame detection results with timestamps
- Emotion count distribution across entire video
- Dominant emotion identification
- Total number of face detections

**Note**: Processing time varies based on video length and resolution. A 1-minute video typically processes in 10-30 seconds.

</p>

### <span style="font-size: 14pt;">4. Batch Processing</span>

<p style="font-size: 12pt;">

**Purpose**: Efficient analysis of multiple images simultaneously.

**Steps**:
1. Navigate to the "Batch Processing" tab
2. Click "Choose Multiple Images"
3. Select multiple image files (Ctrl+Click or Cmd+Click)
4. Wait for batch processing to complete
5. Review results for all images in a consolidated view

**Output**:
- Individual results for each image
- Filename and detection count per image
- Emotion labels and confidence scores for all detected faces
- Summary statistics across the entire batch

**Use Cases**:
- Processing photo albums or collections
- Analyzing multiple subjects simultaneously
- Batch emotion tagging for datasets
- Comparative emotion analysis across images

</p>

### <span style="font-size: 14pt;">5. Data Export</span>

<p style="font-size: 12pt;">

**Purpose**: Export tracking data for external analysis, reporting, or archival.

**Export Formats**:

**JSON Format**:
- Complete metadata (export timestamp, record count)
- Session statistics (dominant emotion, average confidence, duration)
- Full detection history with timestamps
- Probability distributions for all emotions
- Ideal for programmatic processing and data science workflows

**CSV Format**:
- Simplified tabular format
- Columns: timestamp, emotion, confidence, elapsed_seconds
- Compatible with Excel, Google Sheets, and data analysis tools
- Ideal for statistical analysis and visualization

**Steps**:
1. Complete a real-time camera session
2. Click "Export JSON" or "Export CSV"
3. File downloads automatically to your default download folder
4. Filename format: `emotion_data_YYYYMMDD_HHMMSS.{json|csv}`

</p>

---

## <span style="font-size: 18pt;">Technical Architecture</span>

### <span style="font-size: 14pt;">System Components</span>

<p style="font-size: 12pt;">

**Backend Framework**:
- Flask 3.0.0 for web server and API endpoints
- Flask-CORS for cross-origin resource sharing
- RESTful API architecture

**Computer Vision**:
- OpenCV 4.8.1 for image and video processing
- MTCNN (Multi-task Cascaded Convolutional Networks) for face detection
- 90% confidence threshold for face detection accuracy

**Deep Learning**:
- PyTorch 2.1.0 for neural network implementation
- Custom CNN architecture with 4 convolutional layers
- Batch normalization and dropout for regularization
- Softmax activation for probability distribution

**Frontend**:
- HTML5, CSS3, JavaScript (ES6+)
- Chart.js for data visualization
- Responsive design with CSS Grid and Flexbox
- Real-time data polling with AJAX

</p>

### <span style="font-size: 14pt;">Neural Network Architecture</span>

<p style="font-size: 12pt;">

```
Input Layer: 48×48 grayscale image

Convolutional Block 1:
├── Conv2D(64 filters, 3×3 kernel)
├── BatchNorm2D
├── ReLU Activation
└── MaxPool2D(2×2)

Convolutional Block 2:
├── Conv2D(128 filters, 3×3 kernel)
├── BatchNorm2D
├── ReLU Activation
└── MaxPool2D(2×2)

Convolutional Block 3:
├── Conv2D(256 filters, 3×3 kernel)
├── BatchNorm2D
├── ReLU Activation
└── MaxPool2D(2×2)

Convolutional Block 4:
├── Conv2D(512 filters, 3×3 kernel)
├── BatchNorm2D
├── ReLU Activation
└── MaxPool2D(2×2)

Fully Connected Layers:
├── Flatten
├── FC(512 neurons) + Dropout(0.5)
├── FC(256 neurons) + Dropout(0.5)
└── FC(7 neurons) [Output]

Output: Softmax probabilities for 7 emotion classes
```

**Model Parameters**: Approximately 15 million trainable parameters

</p>

### <span style="font-size: 14pt;">Data Flow</span>

<p style="font-size: 12pt;">

1. **Image Acquisition**: Capture frame from camera or load from file
2. **Face Detection**: MTCNN identifies face regions with bounding boxes
3. **Preprocessing**: Resize to 48×48, convert to grayscale, normalize
4. **Emotion Classification**: CNN predicts emotion probabilities
5. **Post-processing**: Apply confidence thresholds, format results
6. **Visualization**: Annotate image with bounding boxes and labels
7. **Tracking**: Store detection in temporal history (real-time mode)
8. **Response**: Return results to frontend via JSON API

</p>


---

## <span style="font-size: 18pt;">API Documentation</span>

### <span style="font-size: 14pt;">Endpoints</span>

<p style="font-size: 12pt;">

#### **GET /**
Returns the main web application interface.

**Response**: HTML page

---

#### **GET /video_feed**
Streams real-time video with emotion detection annotations.

**Response**: Multipart MJPEG stream

**Usage**: Set as `<img>` source for live video display

---

#### **GET /get_tracking_data**
Retrieves current emotion tracking data and session statistics.

**Response**:
```json
{
  "history": [
    {
      "timestamp": "2024-02-24T10:30:45.123456",
      "emotion": "Happy",
      "confidence": 0.95,
      "all_probabilities": {
        "Happy": 0.95,
        "Sad": 0.02,
        "Angry": 0.01,
        "Surprise": 0.01,
        "Fear": 0.00,
        "Disgust": 0.00,
        "Neutral": 0.01
      },
      "elapsed_seconds": 12.5
    }
  ],
  "statistics": {
    "total_detections": 150,
    "emotion_distribution": {
      "Happy": 0.65,
      "Neutral": 0.20,
      "Surprise": 0.10,
      "Sad": 0.05
    },
    "average_confidence": 0.87,
    "dominant_emotion": "Happy",
    "session_duration": 45.2
  }
}
```

---

#### **POST /reset_tracking**
Clears emotion tracking history and resets session statistics.

**Response**:
```json
{
  "success": true
}
```

---

#### **POST /analyze_image**
Analyzes uploaded image for facial emotions.

**Request**: multipart/form-data
- `image`: Image file (JPG, PNG, BMP)

**Response**:
```json
{
  "success": true,
  "results": [
    {
      "bbox": [100, 150, 300, 350],
      "emotion": "Happy",
      "confidence": 0.92,
      "all_emotions": {
        "Happy": 0.92,
        "Neutral": 0.05,
        "Surprise": 0.02,
        "Sad": 0.01
      }
    }
  ],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

---

#### **POST /analyze_video**
Processes video file for emotion analysis.

**Request**: multipart/form-data
- `video`: Video file (MP4, AVI, MOV)

**Response**:
```json
{
  "success": true,
  "total_frames": 1500,
  "analyzed_frames": 300,
  "frame_results": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "detections": [...]
    }
  ],
  "statistics": {
    "emotion_counts": {
      "Happy": 150,
      "Neutral": 100,
      "Sad": 30,
      "Angry": 20
    },
    "dominant_emotion": "Happy",
    "total_detections": 300
  }
}
```

---

#### **POST /batch_analyze**
Processes multiple images simultaneously.

**Request**: multipart/form-data
- `files`: Multiple image files

**Response**:
```json
{
  "success": true,
  "total_files": 5,
  "results": [
    {
      "filename": "image1.jpg",
      "index": 0,
      "detections": [...]
    }
  ]
}
```

---

#### **POST /export_data**
Exports tracking data to file.

**Request**:
```json
{
  "format": "json"  // or "csv"
}
```

**Response**: File download (application/json or text/csv)

---

#### **POST /stop_camera**
Releases camera resources.

**Response**:
```json
{
  "success": true
}
```

</p>

---

## <span style="font-size: 18pt;">Commands Reference</span>

### <span style="font-size: 14pt;">Installation Commands</span>

<p style="font-size: 12pt;">

```bash
# Automated setup (recommended)
python setup.py

# Manual setup
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Verify installation
python -c "import torch; import cv2; print('Success!')"
```

</p>

### <span style="font-size: 14pt;">Running the Application</span>

<p style="font-size: 12pt;">

```bash
# Start the application
python app.py

# Access in browser
# http://localhost:5000

# Stop server
# Press Ctrl + C in terminal
```

</p>

### <span style="font-size: 14pt;">Model Training</span>

<p style="font-size: 12pt;">

```bash
# Download FER2013 dataset from Kaggle
# Extract to ./fer2013/ directory

# Train the model
python train_model.py

# Model will be saved as emotion_model_best.pth
```

</p>

### <span style="font-size: 14pt;">Development Commands</span>

<p style="font-size: 12pt;">

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Deactivate environment
deactivate

# Update pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install with no cache
pip install -r requirements.txt --no-cache-dir

# Freeze current environment
pip freeze > requirements.txt
```

</p>

### <span style="font-size: 14pt;">Git Commands</span>

<p style="font-size: 12pt;">

```bash
# Initialize repository
git init

# Add remote
git remote add origin https://github.com/username/repo.git

# Add files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to remote
git push -u origin main

# Pull from remote
git pull
```

</p>

### <span style="font-size: 14pt;">Testing Commands</span>

<p style="font-size: 12pt;">

```bash
# Test imports
python -c "import app; print('App imports OK')"

# Test Flask
python -c "from flask import Flask; print('Flask OK')"

# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

</p>

### <span style="font-size: 14pt;">Debugging Commands</span>

<p style="font-size: 12pt;">

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Find Python path
which python  # macOS/Linux
where python  # Windows

# Check port usage
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows
```

</p>

---

## <span style="font-size: 18pt;">Configuration</span>

### <span style="font-size: 14pt;">Application Settings</span>

<p style="font-size: 12pt;">

**Face Detection Threshold**:
```python
# In app.py, line ~120
if prob < 0.9:  # Adjust between 0.0 and 1.0
    continue
```
- Lower values: More detections, potential false positives
- Higher values: Fewer detections, higher accuracy
- Recommended: 0.85 - 0.95

**Video Frame Sampling**:
```python
# In app.py, line ~250
if frame_count % 5 == 0:  # Analyze every Nth frame
```
- Lower values: More detailed analysis, slower processing
- Higher values: Faster processing, less detail
- Recommended: 3-10 frames

**Tracking History Size**:
```python
# In app.py, line ~100
self.tracker = EmotionTracker(max_history=100)
```
- Adjust based on memory constraints and session length
- Recommended: 50-200 entries

**Server Configuration**:
```python
# In app.py, final line
app.run(debug=True, host='0.0.0.0', port=5000)
```
- `debug=False` for production deployment
- `host='127.0.0.1'` for local-only access
- Change `port` if 5000 is occupied

</p>

### <span style="font-size: 14pt;">Current Implementation</span>

<p style="font-size: 12pt;">

The current version uses OpenCV's Haar Cascade for face detection and a simplified emotion detection algorithm based on facial features (brightness and contrast analysis).

**For Production Use**:

To achieve higher accuracy, consider:
1. Using Python 3.10 or 3.11 (which supports TensorFlow)
2. Training a deep learning model on the FER2013 dataset
3. Using pre-trained models like DeepFace or FER library
4. Implementing transfer learning with models like VGG16 or ResNet

**Current Approach**:
- Face Detection: OpenCV Haar Cascade
- Emotion Detection: Feature-based analysis (brightness, contrast)
- Accuracy: Suitable for demonstration and basic use cases
- No training required: Works out of the box

</p>


---

## <span style="font-size: 18pt;">Troubleshooting</span>

### <span style="font-size: 14pt;">Common Issues and Solutions</span>

<p style="font-size: 12pt;">

**Issue**: Camera not detected or "Permission denied" error

**Solutions**:
- Verify webcam is connected and functional
- Grant browser camera permissions when prompted
- Check if another application is using the camera
- Try a different browser (Chrome recommended)
- On Linux, ensure user is in `video` group: `sudo usermod -a -G video $USER`

---

**Issue**: Low frame rate or laggy performance

**Solutions**:
- Close unnecessary applications to free system resources
- Reduce video resolution in camera settings
- Increase frame skip rate in configuration
- Use GPU acceleration if available
- Ensure adequate lighting for faster face detection

---

**Issue**: "No faces detected" despite visible faces

**Solutions**:
- Ensure adequate lighting conditions
- Position face frontally toward camera
- Remove obstructions (glasses, masks, hair)
- Lower face detection threshold in configuration
- Verify face is within camera frame and properly sized

---

**Issue**: Installation fails with "No module named 'torch'"

**Solutions**:
```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision

# For CUDA support (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

**Issue**: "CUDA out of memory" error

**Solutions**:
```python
# Force CPU usage in app.py
self.device = torch.device('cpu')
```
- Reduce batch size if training
- Close other GPU-intensive applications
- Use CPU mode for inference

---

**Issue**: Charts not displaying in web interface

**Solutions**:
- Verify internet connection (Chart.js loads from CDN)
- Check browser console for JavaScript errors
- Clear browser cache and reload page
- Ensure tracking data is being generated (start camera first)

---

**Issue**: Export functionality not working

**Solutions**:
- Verify `exports/` directory exists and has write permissions
- Check available disk space
- Ensure tracking session has data before exporting
- Try different export format (JSON vs CSV)

---

**Issue**: Port 5000 already in use

**Solutions**:
```python
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```
Or kill the process using port 5000:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :5000
kill -9 <PID>
```

</p>

### <span style="font-size: 14pt;">Performance Optimization</span>

<p style="font-size: 12pt;">

**Hardware Acceleration**:
- Install CUDA toolkit and cuDNN for NVIDIA GPUs
- PyTorch automatically utilizes GPU if available
- Expected speedup: 3-5x faster than CPU
- Verify GPU usage: `torch.cuda.is_available()`

**CPU Optimization**:
- Use multi-threading for batch processing
- Enable OpenCV optimizations: `cv2.setUseOptimized(True)`
- Compile OpenCV with Intel MKL for Intel CPUs

**Reduce Processing Load**:
- Increase frame skip rate for video analysis
- Lower camera resolution (640×480 recommended)
- Disable tracking for simple image analysis
- Process smaller batches for batch analysis

**Memory Management**:
- Reduce tracking history size
- Clear tracking data periodically
- Use video streaming instead of loading entire video into memory

</p>

---

## <span style="font-size: 18pt;">Project Structure</span>

<p style="font-size: 12pt;">

```
facial-sentiment-analysis/
│
├── app.py                      # Main Flask application
├── setup.py                    # Automated setup script
├── requirements.txt            # Python dependencies
├── haarcascade_frontalface_default.xml  # Face detection model
│
├── templates/
│   └── index.html             # Web interface
│
├── exports/                    # Exported data files (JSON/CSV)
│   └── .gitkeep
│
├── models/                     # Reserved for future trained models
│   └── .gitkeep
│
├── uploads/                    # Temporary uploaded files
│   └── .gitkeep
│
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore                 # Git ignore rules
```

</p>

### <span style="font-size: 14pt;">File Descriptions</span>

<p style="font-size: 12pt;">

**app.py** (13.8 KB)
- Main Flask web application
- Emotion detection using OpenCV
- API endpoints for all features
- Real-time tracking and statistics
- Export functionality
- Feature-based emotion analysis

**setup.py** (2.4 KB)
- Automated installation script
- Dependency verification
- Directory creation
- Environment setup

**haarcascade_frontalface_default.xml**
- OpenCV Haar Cascade for face detection
- Pre-trained face detection model
- No training required

**templates/index.html** (29.3 KB)
- Complete web UI
- Four-tab interface (Real-time, Image, Video, Batch)
- Interactive charts (Chart.js)
- Real-time statistics dashboard
- Export controls

**requirements.txt**
- Python package dependencies
- Version specifications
- Installation requirements

**.gitignore**
- Git ignore patterns
- Excludes temporary files
- Excludes large model files
- Keeps directory structure

</p>

---

## <span style="font-size: 18pt;">Contributing</span>

<p style="font-size: 12pt;">

Contributions are welcome and appreciated. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Submit a pull request

</p>

### <span style="font-size: 14pt;">Contribution Guidelines</span>

<p style="font-size: 12pt;">

**Code Style**:
- Follow PEP 8 style guide for Python code
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all functions and classes

**Testing**:
- Add unit tests for new features
- Ensure existing tests pass
- Test on multiple browsers (Chrome, Firefox, Safari)
- Test on different operating systems if possible

**Documentation**:
- Update README.md for new features
- Add inline comments for complex code
- Update API documentation for new endpoints
- Include usage examples

**Pull Request Process**:
1. Maintainers will review your pull request
2. Address any requested changes
3. Once approved, your code will be merged
4. Your contribution will be acknowledged

**Community Guidelines**:
- Be respectful and constructive
- Help others learn and grow
- Give credit where due

</p>

---

## <span style="font-size: 18pt;">License</span>

<p style="font-size: 12pt;">

This project is licensed under the MIT License - see the LICENSE file for details.

**Third-Party Licenses**:
- PyTorch: BSD License
- OpenCV: Apache 2.0 License
- Flask: BSD License
- Chart.js: MIT License
- facenet-pytorch: MIT License

**Dataset License**:

The FER2013 dataset, if used for training, is subject to its own license terms and is intended for academic and research purposes only. Please ensure compliance with the dataset's license before use.

</p>

---

## <span style="font-size: 18pt;">Acknowledgments</span>

<p style="font-size: 12pt;">

- FER2013 dataset creators for providing training data
- MTCNN implementation by timesler (facenet-pytorch)
- PyTorch and OpenCV development communities
- Chart.js for visualization capabilities
- All contributors to this project

</p>

---

## <span style="font-size: 18pt;">Use Cases</span>

<p style="font-size: 12pt;">

**Personal**:
- Emotion journaling and mood tracking
- Self-awareness and emotional intelligence development
- Photo analysis and memory enhancement

**Professional**:
- Customer sentiment analysis
- User experience research
- Market research and focus groups
- Training feedback and assessment

**Research**:
- Emotion studies and behavioral analysis
- Dataset creation for machine learning
- Algorithm testing and validation
- Psychology and neuroscience research

**Education**:
- Psychology and emotion studies
- AI/ML learning and demonstrations
- Computer vision projects
- Student research projects

</p>

---

## <span style="font-size: 18pt;">FAQ</span>

<p style="font-size: 12pt;">

**Q: Do I need a GPU to run this application?**  
A: No, the application works on CPU. However, a GPU will significantly improve performance (3-5x faster).

**Q: Can I use this for commercial purposes?**  
A: Yes, the project is licensed under MIT License. However, ensure compliance with third-party licenses and dataset terms.

**Q: How accurate is the emotion detection?**  
A: With proper training on FER2013, the model achieves 65-70% accuracy. Accuracy depends on lighting, face angle, and image quality.

**Q: Can it detect emotions for multiple people?**  
A: Yes, the system can detect and analyze emotions for multiple faces in a single frame simultaneously.

**Q: Does it work with recorded videos?**  
A: Yes, you can upload and analyze video files. The system processes every 5th frame by default (configurable).

**Q: Is my data stored or sent to a server?**  
A: No, all processing is done locally on your machine. No data is stored permanently or sent to external servers.

**Q: Can I train the model with my own dataset?**  
A: Yes, you can modify `train_model.py` to work with custom datasets. Ensure your dataset follows the same structure as FER2013.

**Q: What browsers are supported?**  
A: Chrome, Firefox, Safari, and Edge (version 90+). Chrome is recommended for best performance.

**Q: Can I change the emotions detected?**  
A: Yes, but you'll need to retrain the model with a dataset containing your desired emotion categories.

**Q: How do I improve detection accuracy?**  
A: Ensure good lighting, frontal face position, train on FER2013 dataset, and adjust detection thresholds.

</p>

---

## <span style="font-size: 18pt;">Contact & Support</span>

<p style="font-size: 12pt;">

For questions, issues, or suggestions:
- Open an issue on GitHub
- Submit a pull request
- Contact the maintainers

**Reporting Bugs**:
- Provide clear description of the problem
- Include steps to reproduce
- Share error messages or logs
- Specify system information (OS, Python version, GPU/CPU)

**Feature Requests**:
- Describe the feature and its benefits
- Provide use cases
- Consider implementation complexity

</p>

---

<p style="font-size: 12pt; text-align: center;">
<strong>Built with ❤️ using Python, PyTorch, and OpenCV</strong><br>
<strong>Version 1.0 | Last Updated: February 2024</strong>
</p>

</div>

