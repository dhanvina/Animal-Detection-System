import os
import cv2
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from src.utils.detection import AnimalDetector

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)


class AnimalDetectionApp:
    """
    Object-oriented wrapper for the Flask animal detection app.
    Handles initialization, routing, and detection logic.
    """
    def __init__(self, app: Flask):
        self.app = app
        self.detector = None
        self._configure_app()
        self._register_routes()
        self.initialize_detector()  # Always initialize detector at startup

    def _configure_app(self):
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
        self.app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('static/results', exist_ok=True)

    def allowed_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS']

    def initialize_detector(self):
        """Initialize the animal detector on first request"""
        if self.detector is None:
            print("Initializing Animal Detector...")
            self.detector = AnimalDetector(conf_threshold=0.5, iou_threshold=0.45)
            print("Animal Detector initialized!")

    def register(self):
        # No-op for compatibility; initialization is now always done in __init__
        return self

    def _register_routes(self):
        @self.app.route('/')
        def index():
            """Render the main page"""
            return render_template('index.html')

        @self.app.route('/detect', methods=['POST'])
        def detect_animals():
            """Handle image/video upload and run detection"""
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            file = request.files['file']
            file_type = request.form.get('type', 'image')
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if not self.allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_ext = os.path.splitext(file.filename)[1].lower()
                filename = f"{file_type}_{timestamp}{file_ext}"
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                if file_type == 'video':
                    output_filename = f"detected_{filename}"
                    output_path = os.path.join('static', 'results', output_filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cap = cv2.VideoCapture(filepath)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    detections = []
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % 5 == 0:
                            processed_frame, frame_detections = self.detector.process_frame(frame)
                            detections.extend(frame_detections)
                            out.write(processed_frame)
                        else:
                            out.write(frame)
                        frame_count += 1
                    cap.release()
                    out.release()
                    return jsonify({
                        'type': 'video',
                        'detections': detections,
                        'video_url': f"/{output_path}",
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    detections = self.detector.detect_animals(filepath)
                    if detections:
                        img = cv2.imread(filepath)
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            label = f"{det['class']} {det['confidence']:.2f}"
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        result_filename = f"detected_{filename}"
                        result_path = os.path.join('static', 'results', result_filename)
                        cv2.imwrite(result_path, img)
                        result_url = f"/static/results/{result_filename}"
                    else:
                        result_url = f"/{filepath}"
                    return jsonify({
                        'type': 'image',
                        'detections': detections,
                        'image_url': result_url,
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500

        @self.app.route('/realtime')
        def realtime():
            """Render the real-time detection page"""
            return render_template('realtime.html')

        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route for real-time detection"""
            return Response(self.generate_frames(), 
                           mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/static/results/<path:filename>')
        def serve_result(filename):
            """Serve processed files"""
            return send_from_directory('static/results', filename)

    def generate_frames(self):
        """Generate video frames with real-time detection"""
        camera = cv2.VideoCapture(0)
        while True:
            success, frame = camera.read()
            if not success:
                break
            processed_frame, _ = self.detector.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        camera.release()

# Instantiate and register the OOP Flask app

# Only register routes if not already registered (avoid duplicate registration on reload)
if not hasattr(app, '_routes_registered'):
    animal_app = AnimalDetectionApp(app)
    animal_app.register()
    app._routes_registered = True

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_animals():
    """Handle image/video upload and run detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type', 'image')  # Get the type (image or video)
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{file_type}_{timestamp}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process based on file type
        if file_type == 'video':
            # Process video
            output_filename = f"detected_{filename}"
            output_path = os.path.join('static', 'results', output_filename)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Process video frames
            cap = cv2.VideoCapture(filepath)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            detections = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process every 5th frame to save processing time
                if frame_count % 5 == 0:
                    # Detect animals in the frame
                    processed_frame, frame_detections = detector.process_frame(frame)
                    detections.extend(frame_detections)
                    out.write(processed_frame)
                else:
                    out.write(frame)
                    
                frame_count += 1
            
            cap.release()
            out.release()
            
            return jsonify({
                'type': 'video',
                'detections': detections,
                'video_url': f"/{output_path}",
                'timestamp': datetime.now().isoformat()
            })
            
        else:  # Image processing
            # Process image
            detections = detector.detect_animals(filepath)
            
            # Draw detections on image if any
            if detections:
                img = cv2.imread(filepath)
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    label = f"{det['class']} {det['confidence']:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save result
                result_filename = f"detected_{filename}"
                result_path = os.path.join('static', 'results', result_filename)
                cv2.imwrite(result_path, img)
                result_url = f"/static/results/{result_filename}"
            else:
                result_url = f"/{filepath}"
            
            return jsonify({
                'type': 'image',
                'detections': detections,
                'image_url': result_url,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/realtime')
def realtime():
    """Render the real-time detection page"""
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for real-time detection"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate video frames with real-time detection"""
    camera = cv2.VideoCapture(0)  # Use default camera
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame
        processed_frame, _ = detector.process_frame(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/static/results/<path:filename>')
def serve_result(filename):
    """Serve processed files"""
    return send_from_directory('static/results', filename)

if __name__ == '__main__':
    # Initialize detector before running the app
    initialize_detector()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
