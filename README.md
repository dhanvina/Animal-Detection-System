# 🐾 Animal Detection System

An industry-grade web application for real-time animal detection in images and videos, powered by state-of-the-art YOLOv8 deep learning models. Built for wildlife monitoring, research, and smart surveillance, this project features a modern UI, robust backend, and scalable architecture.

---

## 🚀 Features

- **Image & Video Detection:** Upload images or videos and get instant animal detection results with bounding boxes and confidence scores.
- **Modern, Responsive UI:** Professional, mobile-friendly interface with live previews and clear result visualization.
- **YOLOv8 Integration:** Utilizes the latest YOLOv8x model for high-accuracy, real-time animal detection.
- **RESTful API:** Easily extend or integrate with other systems.
- **Security:** File type validation, size limits, and secure upload handling.
- **Extensible:** Modular codebase for easy customization and future enhancements.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML5, Bootstrap 5, JavaScript (ES6+)
- **ML Model:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV
- **Deep Learning:** PyTorch

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- 4GB+ RAM (8GB+ recommended for video)
- Modern web browser

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/animal-detection.git
   cd animal-detection
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python app.py
   ```
5. **Open your browser:**
   - Go to [http://localhost:5000](http://localhost:5000)

## 📁 Project Structure

```
animal-detection/
├── app.py                 # Main Flask application (OOP, industry standard)
├── requirements.txt       # Python dependencies
├── src/
│   └── utils/             # Modular utility code (detection, model loading)
├── static/
│   ├── js/                # JavaScript frontend logic
│   └── results/           # Processed images/videos
├── templates/             # HTML templates (Bootstrap 5, modern UI)
├── uploads/               # Temporary storage for uploads
├── models/                # (Optional) Custom YOLO models
├── LICENSE
├── .gitignore
└── README.md
```

## 🎯 Usage

1. Access the web interface in your browser.
2. Choose image or video upload.
3. Select a file and click "Detect".
4. View results with animal names, emojis, and confidence scores.
5. Download or review processed media.


## � Download YOLOv8 Model Weights

Due to GitHub file size limits, you must manually download the YOLOv8 model weights:

1. Download `yolov8x.pt` and `yolov8n.pt` from [this Google Drive folder](https://drive.google.com/drive/folders/1v7xc6x64NN21tWooYYVaIH1eHxfEFkNe?usp=sharing).
2. Place the files in the project root directory (where `app.py` is located).

If you need a smaller model, you can also download it from the [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics/releases).

**Note:** These files are required for detection to work. They are not included in the repository due to GitHub's 100MB file size limit.

## 🛡️ Security & Best Practices

- File type and size validation (32MB max)
- Secure file handling and input sanitization
- Modular, maintainable, and extensible codebase
- Professional error handling and logging

## 🤝 Contributing

Contributions are welcome! Please:
- Fork the repo and create a feature branch
- Write clear, well-documented code (PEP8, docstrings)
- Submit a pull request with a detailed description

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📧 Contact

For questions, suggestions, or support, open an issue or email [ndhanvina07@gmail.com].

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- OpenCV, PyTorch, Bootstrap, and the open-source community

## API Endpoints

- `GET /`: Main web interface
- `POST /detect`: Upload an image for animal detection
  - Request: Form-data with 'file' field containing the image
  - Response: JSON with detection results

## Model Integration

To use a custom model:

1. Place your model files in the `models/` directory
2. Update the `ModelLoader` class in `utils/model_loader.py` to load your model
3. Configure the detection parameters in `utils/detection.py`

## Project Structure

```
wildlife-monitoring/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── uploads/               # Temporary storage for uploaded images
├── models/                # Trained models
├── templates/             # HTML templates
│   └── index.html         # Main web interface
└── utils/                 # Utility functions
    ├── detection.py       # Animal detection logic
    └── model_loader.py    # Model loading utilities
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
