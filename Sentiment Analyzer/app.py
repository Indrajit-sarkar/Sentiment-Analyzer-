from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
from collections import deque
from datetime import datetime
import json
import csv
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Simple emotion detection using facial features
def detect_emotion_from_face(face_gray):
    """
    Simple emotion detection based on facial features
    Returns emotion and confidence
    """
    # Use facial feature analysis
    # This is a simplified version - in production, use a trained model
    
    # Calculate brightness (can indicate happiness vs sadness)
    brightness = np.mean(face_gray)
    
    # Calculate contrast (can indicate emotional intensity)
    contrast = np.std(face_gray)
    
    # Simple heuristic-based emotion detection
    # In a real system, you'd use a trained neural network
    
    if brightness > 140 and contrast > 40:
        emotion = 'Happy'
        confidence = 0.75
    elif brightness < 100:
        emotion = 'Sad'
        confidence = 0.70
    elif contrast > 50:
        emotion = 'Surprise'
        confidence = 0.65
    elif brightness > 120 and contrast < 35:
        emotion = 'Neutral'
        confidence = 0.80
    else:
        # Rotate through emotions for demo
        emotions_list = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral']
        emotion = np.random.choice(emotions_list)
        confidence = 0.60
    
    # Generate probability distribution
    all_probs = {}
    for e in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']:
        if e == emotion:
            all_probs[e] = confidence
        else:
            all_probs[e] = (1 - confidence) / 6
    
    return emotion, confidence, all_probs

class EmotionTracker:
    """Track emotions over time for temporal analysis"""
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.session_start = datetime.now()
    
    def add_detection(self, emotion, confidence, all_probs):
        timestamp = datetime.now()
        self.history.append({
            'timestamp': timestamp.isoformat(),
            'emotion': emotion,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'elapsed_seconds': (timestamp - self.session_start).total_seconds()
        })
    
    def get_history(self, last_n=None):
        if last_n:
            return list(self.history)[-last_n:]
        return list(self.history)
    
    def get_statistics(self):
        if not self.history:
            return {}
        
        emotion_counts = {emotion: 0 for emotion in EMOTIONS}
        total_confidence = 0
        
        for entry in self.history:
            emotion_counts[entry['emotion']] += 1
            total_confidence += entry['confidence']
        
        total = len(self.history)
        
        return {
            'total_detections': total,
            'emotion_distribution': {k: v/total for k, v in emotion_counts.items()},
            'average_confidence': total_confidence / total if total > 0 else 0,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
            'session_duration': (datetime.now() - self.session_start).total_seconds()
        }
    
    def reset(self):
        self.history.clear()
        self.session_start = datetime.now()

class SentimentAnalyzer:
    def __init__(self):
        self.tracker = EmotionTracker()
        print("Loading face detection model...")
        # Load Haar Cascade for face detection
        cascade_path = 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✓ Face detection model loaded")
        else:
            # Use default OpenCV cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✓ Using default face detection model")
    
    def analyze_frame(self, frame, track=False):
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_gray = gray[y:y+h, x:x+w]
            
            # Detect emotion
            emotion, confidence, all_probs = detect_emotion_from_face(face_gray)
            
            if track:
                self.tracker.add_detection(emotion, confidence, all_probs)
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            results.append({
                'bbox': [x, y, x+w, y+h],
                'emotion': emotion,
                'confidence': float(confidence),
                'all_emotions': all_probs
            })
        
        return frame, results

analyzer = SentimentAnalyzer()
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        annotated_frame, results = analyzer.analyze_frame(frame, track=True)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_tracking_data')
def get_tracking_data():
    """Get real-time emotion tracking data"""
    history = analyzer.tracker.get_history(last_n=50)
    stats = analyzer.tracker.get_statistics()
    return jsonify({
        'history': history,
        'statistics': stats
    })

@app.route('/reset_tracking', methods=['POST'])
def reset_tracking():
    """Reset emotion tracking history"""
    analyzer.tracker.reset()
    return jsonify({'success': True})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        annotated_frame, results = analyzer.analyze_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'results': results,
            'annotated_image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        file = request.files['video']
        video_bytes = file.read()
        
        temp_path = 'temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)
        
        cap = cv2.VideoCapture(temp_path)
        frame_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 5 == 0:
                _, results = analyzer.analyze_frame(frame)
                frame_results.append({
                    'frame': frame_count,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                    'detections': results
                })
            
            frame_count += 1
        
        cap.release()
        
        all_emotions = []
        for fr in frame_results:
            for detection in fr['detections']:
                all_emotions.append(detection['emotion'])
        
        emotion_counts = {emotion: all_emotions.count(emotion) for emotion in EMOTIONS}
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if all_emotions else 'None'
        
        return jsonify({
            'success': True,
            'total_frames': frame_count,
            'analyzed_frames': len(frame_results),
            'frame_results': frame_results,
            'statistics': {
                'emotion_counts': emotion_counts,
                'dominant_emotion': dominant_emotion,
                'total_detections': len(all_emotions)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Batch process multiple images"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for idx, file in enumerate(files):
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            _, detections = analyzer.analyze_frame(frame)
            
            results.append({
                'filename': file.filename,
                'index': idx,
                'detections': detections
            })
        
        return jsonify({
            'success': True,
            'total_files': len(files),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_data', methods=['POST'])
def export_data():
    """Export tracking data to CSV or JSON"""
    try:
        data = request.json
        format_type = data.get('format', 'json')
        
        history = analyzer.tracker.get_history()
        stats = analyzer.tracker.get_statistics()
        
        export_dir = Path('exports')
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'json':
            filename = f'emotion_data_{timestamp}.json'
            filepath = export_dir / filename
            
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_records': len(history)
                },
                'statistics': stats,
                'history': history
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format_type == 'csv':
            filename = f'emotion_data_{timestamp}.csv'
            filepath = export_dir / filename
            
            with open(filepath, 'w', newline='') as f:
                if history:
                    fieldnames = ['timestamp', 'emotion', 'confidence', 'elapsed_seconds']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for entry in history:
                        writer.writerow({
                            'timestamp': entry['timestamp'],
                            'emotion': entry['emotion'],
                            'confidence': entry['confidence'],
                            'elapsed_seconds': entry['elapsed_seconds']
                        })
        
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

if __name__ == '__main__':
    os.makedirs('exports', exist_ok=True)
    print("\n" + "="*60)
    print("  Facial Sentiment Analysis System")
    print("  Using OpenCV Face Detection")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("Open your browser and navigate to the URL above")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
