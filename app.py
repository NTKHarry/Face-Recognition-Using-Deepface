from flask import Flask, render_template, request, redirect, url_for
from deepface import DeepFace
import cv2
import os
import uuid  # For generating unique filenames
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
# Create the folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        # Get uploaded images
        img1 = request.files['image1']
        img2 = request.files['image2']
        
        # Save images
        img1_path = os.path.join(UPLOAD_FOLDER, img1.filename)
        img2_path = os.path.join(UPLOAD_FOLDER, img2.filename)
        img1.save(img1_path)
        img2.save(img2_path)
        
        # Perform verification
        result = DeepFace.verify(img1_path, img2_path)

        return render_template('result.html', result=result, img1_path=img1_path, img2_path=img2_path)
    return render_template('verify.html')

@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        img = request.files['image']
        
        # Save the uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        # Analyze the image using DeepFace
        analysis = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)

        # Read the image with OpenCV
        img_cv = cv2.imread(img_path)

        # List to store cropped face file paths and analysis results
        faces_with_info = []

        for face in analysis:
            # Extract the bounding box coordinates
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            # Crop the face from the image
            face_img = img_cv[y:y+h, x:x+w]

            # Generate a unique filename for each cropped face
            face_filename = f"{uuid.uuid4()}.jpg"
            face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)

            # Save the cropped face image
            cv2.imwrite(face_file_path, face_img)

            # Save the face info and path to the cropped face image
            face_info = {
                'image': face_filename,  # Save filename, not the image array
                'age': face['age'],
                'gender': face['gender'],
                'emotion': face['dominant_emotion'],
                'race': face['dominant_race']
            }
            faces_with_info.append(face_info)

        return render_template('info_result.html', faces=faces_with_info)
    return render_template('info.html')
if __name__ == '__main__':
    app.run(debug=True)
