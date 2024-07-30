from flask import Flask, jsonify, request,send_file
from flask_cors import CORS
import base64
from pipeline import get_similar_product
from PIL import Image
import os
import config


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return jsonify(message="Welcome to the Flask app with CORS enabled")

@app.route('/api/upload', methods=['POST'])
def upload():
    
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify(error="Image and text are required"), 400

    image_file = request.files['image']
    text = request.form['text']

    # Open the image file directly from the request
    image = Image.open(image_file.stream)

    out_paths, out_titles=get_similar_product(image,text)


    return jsonify({'titles':out_titles, "img_paths":out_paths }), 200

@app.route('/api/get_image', methods=['GET'])
def get_image():
    image_path = request.args.get('path')
    base_path=config.images_base_pth
    final_pth=os.path.join(base_path,image_path)
    if not final_pth or not os.path.exists(final_pth):
        return jsonify(error="Invalid or missing image path"), 400

    try:
        return send_file(final_pth, mimetype='image/jpeg')
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)