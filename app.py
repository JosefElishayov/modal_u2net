import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os
from flask import Flask, request, send_file, jsonify, after_this_request
from u2net import U2NET
from tempfile import NamedTemporaryFile
import zipfile
# יצירת אפליקציית Flask
app = Flask(__name__)



# הקישור לקובץ ה-ZIP בשרת ה-WordPress
model_url = "http://test.woocomanage.com/wp-content/uploads/2024/11/u2net.zip"
# נתיב שמירת ה-ZIP והקובץ המודחס
zip_path = "saved_models/u2net/u2net.zip"
# נתיב שמירת המודל לאחר החילוץ
model_path = "saved_models/u2net/u2net.pth"

def download_and_extract_model():
    """הורדה וחילוץ קובץ המודל בשרת המקומי."""
    if not os.path.exists(model_path):
        print("Model not found locally. Downloading model...")
        # הגדרת כותרות הבקשה, כולל 'User-Agent'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # בקשת הורדת הקובץ עם הכותרות
        response = requests.get(model_url, headers=headers, stream=True)
        if response.status_code == 200:
            # יצירת התיקייה אם היא לא קיימת
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            # הורדה ושמירה של הקובץ
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded. Extracting...")

            # חילוץ תוכן ה-ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(model_path))

            # מחיקת קובץ ה-ZIP לאחר החילוץ
            os.remove(zip_path)
            print("Model extracted and ready.")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")
    else:
        print("Model already exists locally. Skipping download.")

# קריאה לפונקציה להורדת המודל אם נדרש
download_and_extract_model()

net = U2NET(3, 1)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval()

def download_image(url):
    # הורדת תמונה מהאינטרנט ומחזירה אותה כאובייקט תמונה
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    else:
        raise Exception("Failed to download image")

def remove_background(image):
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = net(image_tensor)[0][0]
        output = output.squeeze().cpu().numpy()

    mask = Image.fromarray((output * 255).astype('uint8')).resize(original_size, Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def combine_with_new_background(foreground_image, mask, background_image):
    background = background_image.convert("RGBA")
    background = background.resize(foreground_image.size)

    foreground = foreground_image.convert("RGBA")
    foreground.putalpha(mask)

    combined = Image.alpha_composite(background, foreground)
    combined = combined.convert("P", palette=Image.ADAPTIVE)
    return combined

# יצירת נקודת קצה (API Endpoint)
@app.route('/replace_background', methods=['POST'])
def replace_background():
    try:
        # מקבל קישורים לתמונות מהבקשה
        data = request.get_json()
        foreground_url = data.get('foreground_url')
        background_url = data.get('background_url')

        # הורדת התמונות
        foreground_image = download_image(foreground_url)
        background_image = download_image(background_url)

        # הסרת רקע מהתמונה הקדמית
        mask = remove_background(foreground_image)

        # שילוב עם הרקע החדש
        result_image = combine_with_new_background(foreground_image, mask, background_image)

        # שמירת התוצאה בקובץ זמני
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            result_image.save(tmp.name)
            tmp_path = tmp.name

        # שימוש ב-after_this_request כדי למחוק את הקובץ לאחר השליחה
        @after_this_request
        def remove_file(response):
            try:
                os.remove(tmp_path)
                print(f"File {tmp_path} deleted successfully.")
            except Exception as e:
                print(f"Error deleting file: {e}")
            return response

        # שליחת הקובץ
        return send_file(tmp_path, mimetype='image/png', as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
