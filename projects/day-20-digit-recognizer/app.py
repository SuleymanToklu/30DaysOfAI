import gradio as gr
import joblib
import numpy as np
from PIL import Image, ImageOps

# 1. Load the trained model (trained on the MNIST dataset)
MODEL_FILENAME = 'digit_recognizer_mnist.joblib'
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"Model file '{MODEL_FILENAME}' not found.")
    print("Please run 'python train_model.py' first to train and save the new MNIST model.")
    model = None

# 2. Define the prediction function
def predict_digit(img):
    """
    This function takes a user-drawn image from the Gradio interface,
    processes it to match the MNIST model's input format (28x28), and returns predictions.
    """
    if model is None:
        return {"Hata": 1.0, "Model Yüklenemedi": 1.0}

    # Handle dictionary output from newer Gradio versions
    if isinstance(img, dict):
        img = img['composite']
    
    # Convert the input to a PIL Image and then to grayscale
    gray_image = Image.fromarray(img.astype('uint8'), 'RGB').convert('L')

    # 1. Invert colors (we need a white digit on a black background)
    inverted_image = ImageOps.invert(gray_image)

    # 2. Find the bounding box of the digit to crop the extra whitespace
    bbox = inverted_image.getbbox()
    if bbox is None:
        return {str(i): 0.0 for i in range(10)}
    
    # 3. Crop the image to the bounding box
    cropped_image = inverted_image.crop(bbox)

    # 4. Add padding and center the digit in a new square canvas
    width, height = cropped_image.size
    # Add padding to prevent the digit from touching the edges
    new_size = max(width, height) + 40
    
    square_canvas = Image.new("L", (new_size, new_size), 0)
    
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2
    square_canvas.paste(cropped_image, (paste_x, paste_y))

    # 5. Resize the final, centered image to 28x28 pixels to match MNIST
    resized_image = square_canvas.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 6. Convert to numpy array and scale values from 0-255 to 0-1
    img_array = np.array(resized_image)
    processed_array = img_array / 255.0
    
    # 7. Flatten and reshape for the model (1, 784)
    flattened_array = processed_array.flatten()
    reshaped_array = flattened_array.reshape(1, -1)
    
    # 8. Get prediction probabilities
    prediction_probabilities = model.predict_proba(reshaped_array)

    # Format the output for Gradio's Label component
    confidences = {str(i): prob for i, prob in enumerate(prediction_probabilities[0])}
    
    return confidences

# 3. Create the Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Lütfen bir rakam çizin (0-9)"),
    outputs=gr.Label(num_top_classes=3, label="Tahmin Sonuçları"),
    title="✍️ El Yazısı Rakam Tanıyıcı (MNIST Modeli)",
    description="""
    Yapay zeka modelinin tahminlerini görmek için aşağıdaki alana bir rakam çizin.
    Bu model, endüstri standardı olan MNIST veri seti ile eğitilmiştir.
    """
)

# 4. Launch the app
if __name__ == "__main__":
    if model is not None:
        demo.launch()
    else:
        print("Gradio app cannot start because the model is not loaded.")

