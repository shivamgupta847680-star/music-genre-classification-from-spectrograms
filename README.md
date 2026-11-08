# Music Genre Classification from Spectrograms

## Project Overview
This project uses Deep Learning and Convolutional Neural Networks (CNN) to classify music into 10 different genres using spectrogram images.

## Objectives
- Classify music into 10 genres.
- Use spectrogram images as input.
- Train a CNN model for genre prediction.
- Provide a simple web interface using Streamlit.

## Technologies Used
- Python
- TensorFlow and Keras
- CNN (Convolutional Neural Network)
- NumPy
- Pillow
- Streamlit

## Dataset
The project uses the GTZAN Music Genre Dataset containing 10 genres:
1. Blues
2. Classical
3. Country
4. Disco
5. Hip-Hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

## Model Performance
- Model: CNN
- Test Accuracy: 62%
- Test Loss: 1.1329

## Project Structure
```text
music-genre-classification/
├── app.py
├── requirements.txt
├── music_genre_model.keras
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements
- Improve model accuracy.
- Support direct audio file prediction.
- Deploy the application online.

## Author
Shivam Gupta
