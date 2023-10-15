from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

app = FastAPI()

# Configure CORS to allow requests from your Flutter app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to only allow your Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TensorFlow Lite model and labels
model = tf.lite.Interpreter(model_path="model_unquant.tflite")
model.allocate_tensors()

with open("labels.txt", "r") as file:
    labels = [line.strip() for line in file]

@app.post("/upload_audio/")
async def upload_audio_file(file: UploadFile):
    if not file:
        return JSONResponse(content={"error": "No file provided"}, status_code=422)

    if not file.filename.endswith((".mp3", ".wav", ".ogg")):
        return JSONResponse(content={"error": "Invalid audio file format"}, status_code=422)

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    # Save the uploaded audio file.
    with open(file_path, "wb") as f:
        f.write(file.file.read())


    y, sr = librosa.load(file_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, x_axis=None, y_axis=None)
    plt.axis('off')  
    spectrogram_file = os.path.join(upload_dir, 'spectrogram.png')
    plt.savefig(spectrogram_file, bbox_inches='tight', pad_inches=0)
    
    image = tf.keras.preprocessing.image.load_img(spectrogram_file, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])
    predicted_label = labels[np.argmax(predictions)]
    print(predicted_label)
    
    return JSONResponse(content={"predicted_label": predicted_label})


@app.get("/predict_label/")
async def predict_label(spectrogram_path: str):
    # Load the spectrogram image
    image = tf.keras.preprocessing.image.load_img(spectrogram_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Perform inference using the TensorFlow Lite model
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])
    predicted_label = labels[np.argmax(predictions)]
  
    return JSONResponse(content={"predicted_label": predicted_label})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
