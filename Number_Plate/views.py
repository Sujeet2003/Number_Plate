from django.shortcuts import render, HttpResponse
from django.http import StreamingHttpResponse
import cv2
import easyocr
import threading
from django.conf import settings

# Global variables for threading
output_text = ""
output_image = None

def index(request):
    return render(request, 'index.html')

def gen_frames():
    global output_text, output_image
    harcascade = "model/haarcascade_plate_number.xml"
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    min_area = 500
    count = 0
    reader = easyocr.Reader(['en'])

    while True:
        success, img = cap.read()
        if not success:
            break

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                img_roi = img[y:y+h, x:x+w]
                output_image = img_roi
                cv2.imshow("ROI", img_roi)
                cv2.imwrite(f"static/scanned_img_{count}.jpg", img_roi)
                result = reader.readtext(img_roi)
                delete_old_images()
                if result:
                    output_text = result[0][1]
                count += 1

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

import os
def delete_old_images():
    static_folder = os.path.join(settings.BASE_DIR, 'static')
    images = sorted([f for f in os.listdir(static_folder) if f.startswith('scanned_img_')], key=lambda x: os.path.getmtime(os.path.join(static_folder, x)))
    
    # Keep the latest 3 images, delete the rest
    if len(images) > 3:
        for image in images[:-3]:
            os.remove(os.path.join(static_folder, image))

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_plate_text(request):
    global output_text
    return HttpResponse(output_text)

def get_plate_image(request):
    global output_image
    if output_image is not None:
        _, buffer = cv2.imencode('.jpg', output_image)
        response = HttpResponse(buffer.tobytes(), content_type='image/jpeg')
        return response
    return HttpResponse(status=404)
