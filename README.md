# Driver Drowsiness Detection System

This project aims to detect driver drowsiness in real-time using webcam input and raise an alert when the driver appears to be sleepy. It implements two approaches:

1. **CNN Model (using Haar Cascades)**
2. **Eye Aspect Ratio (EAR) using Dlib**

---

## 🔧 Technologies Used

- Python 3
- OpenCV
- Dlib
- Keras (TensorFlow backend)
- Imutils
- Scipy
- Numpy
- Pygame (for alert sound)

---



## ▶️ How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run either model:

   - CNN:
     ```bash
     python main_cnn_model.py
     ```

   - Dlib EAR:
     ```bash
     python main_dlib_ear_model.py
     ```

3. Press `q` to quit the window.

---
