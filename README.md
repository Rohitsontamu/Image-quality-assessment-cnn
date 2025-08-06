# 📷 Image Quality Assessment with CNN

A deep learning-based image quality assessment system that predicts Mean Opinion Score (MOS) for images using Convolutional Neural Networks. The project includes both a FastAPI backend service and an interactive Streamlit frontend with image enhancement capabilities.

## 🚀 Features

- **Deep Learning Model**: CNN-based architecture for accurate image quality prediction
- **MOS Prediction**: Predicts quality scores on a 1-10 scale based on Mean Opinion Score
- **Image Enhancement**: Manual and automatic image enhancement with brightness, contrast, sharpness, and colorfulness adjustments
- **Quality Metrics**: Real-time calculation of image quality attributes
- **Interactive UI**: User-friendly Streamlit interface for image upload and analysis
- **API Service**: RESTful API endpoints for programmatic access
- **CI/CD Pipeline**: Automated deployment using Jenkins and Docker
- **Containerized Deployment**: Docker support for easy deployment and scaling

## 📁 Project Structure

```
├── app/                          # FastAPI application
├── docker/                       # Docker configuration files
├── images/                       # Sample images
├── models/                       # Trained CNN models
├── temp/                         # Temporary file storage
├── cnn_model2.py                 # CNN model implementation
├── feature_extraction.py         # Feature extraction utilities
├── metrics.py                    # Evaluation metrics (RMSE, R², PLCC, SROCC, KROCC)
├── streamlit_app.py             # Streamlit frontend application
├── main.py                      # FastAPI main application
├── Jenkinsfile                  # CI/CD pipeline configuration
├── requirements.txt             # Python dependencies
├── req_app_env1.txt            # Backend environment requirements
├── req_train_env2.txt          # Frontend environment requirements
└── *.csv                       # Dataset files (KONIQ-10K)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rohitsontamu/Image-quality-assessment-cnn.git
   cd Image-quality-assessment-cnn
   ```

2. **Create virtual environments**
   ```bash
   # For backend (API)
   python -m venv venv1
   source venv1/bin/activate  # On Windows: venv1\Scripts\activate
   pip install -r req_app_env1.txt
   
   # For frontend (Streamlit)
   python -m venv venv2
   source venv2/bin/activate  # On Windows: venv2\Scripts\activate
   pip install -r req_train_env2.txt
   ```

3. **Download/Train the model**
   - Place your trained CNN model in the `models/` directory
   - Update model paths in [`main.py`](main.py) if necessary

## 🚀 Usage

### Running the Application

#### Option 1: Local Development

1. **Start the FastAPI backend**
   ```bash
   source venv1/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Start the Streamlit frontend**
   ```bash
   source venv2/bin/activate
   streamlit run streamlit_app.py --server.port 8501
   ```

3. **Access the application**
   - Frontend: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

#### Option 2: Docker Deployment

```bash
docker-compose up -d --build
```

### API Endpoints

- **POST** `/predict/` - Upload an image and get quality prediction
  - Input: Image file (JPG, JPEG, PNG)
  - Output: MOS prediction and quality classification

### Using the Streamlit Interface

1. **Upload Image**: Choose an image file (JPG, JPEG, PNG)
2. **View Metrics**: See brightness, contrast, sharpness, and colorfulness values
3. **Apply Enhancements**: 
   - Manual: Use sliders to adjust image properties
   - Auto: Automatic enhancement based on image analysis
   - Off: No enhancements applied
4. **Get Prediction**: View the predicted quality score on a 1-10 scale
5. **Download**: Save enhanced images

## 📊 Quality Metrics

The system calculates several image quality attributes:

- **Brightness**: Average luminance of the image
- **Contrast**: Standard deviation of pixel intensities
- **Sharpness**: Gradient-based edge detection measure
- **Colorfulness**: RGB color distribution analysis

## 🎯 Model Performance

The model is evaluated using standard IQA metrics:
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of Determination
- **PLCC**: Pearson Linear Correlation Coefficient
- **SROCC**: Spearman Rank Order Correlation Coefficient
- **KROCC**: Kendall Rank Order Correlation Coefficient

## 📊 Dataset

The project uses the KONIQ-10K dataset for training and evaluation:
- [`koniq10k_cleaned.csv`](koniq10k_cleaned.csv) - Cleaned dataset
- [`koniq10k_indicators.csv`](koniq10k_indicators.csv) - Quality indicators
- [`koniq10k_scores_and_distributions.csv`](koniq10k_scores_and_distributions.csv) - Score distributions

## 🔧 CI/CD Pipeline

The project includes a Jenkins pipeline ([`Jenkinsfile`](Jenkinsfile)) that:
1. Clones the repository
2. Builds Docker images for API and Streamlit services
3. Runs the application using Docker Compose
4. Pushes images to Docker Hub
5. Provides automated deployment workflow

## 🐳 Docker Configuration

- **API Service**: Containerized FastAPI backend
- **Streamlit Service**: Containerized frontend application
- **Docker Compose**: Orchestrates both services with proper networking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

- **Author**: Rohit Sonagara
- **GitHub**: [@Rohitsontamu](https://github.com/Rohitsontamu)
- **Repository**: [Image-quality-assessment-cnn](https://github.com/Rohitsontamu/Image-quality-assessment-cnn)

## 🙏 Acknowledgments

- KONIQ-10K dataset providers
- TensorFlow/Keras community
- FastAPI and Streamlit frameworks
- Docker and Jenkins for deployment tools