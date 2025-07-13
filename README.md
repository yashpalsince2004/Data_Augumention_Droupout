
# 🌀 MNIST CNN with Data Augmentation & Dropout

This project demonstrates training a Convolutional Neural Network (CNN) on the MNIST dataset using **data augmentation** and **dropout regularization** to improve generalization and robustness.

It uses TensorFlow & Keras for building the model and Matplotlib for visualization.

---

## 📋 Features

✅ Load and preprocess the MNIST dataset  
✅ Perform data augmentation (rotation, zoom, shifts) using `ImageDataGenerator`  
✅ Visualize augmented images  
✅ Build and train a CNN with dropout layers  
✅ Plot training and validation accuracy & loss curves  
✅ Evaluate the model on test data  
✅ Save the trained model as `mnist_cnn_model.h5`

---

## 🏗️ Model Architecture

- Input: MNIST grayscale images (28x28)
- Reshape layer: to (28,28,1)
- Convolutional & MaxPooling layers
- Dropout layers (0.25 & 0.5) to prevent overfitting
- Dense layers for classification
- Output: 10-class softmax

---

## 📈 Training

The model is trained for **10 epochs** using augmented data in batches of 64.

The augmentation parameters used:
- Rotation range: ±10°
- Zoom range: ±10%
- Width & height shift: ±10%

The training and validation accuracy & loss curves are plotted after training.

---

## 🚀 Results

At the end of training:
- The model achieves competitive test accuracy on MNIST.
- The trained model is saved as `mnist_cnn_model.h5`.

---

## 🖼️ Sample Output

- Grid of augmented MNIST samples
- Accuracy & Loss over epochs
- Final test accuracy printed in console

---

## 📂 Usage

### Prerequisites
- Python ≥ 3.7
- TensorFlow ≥ 2.x
- Matplotlib

### Run the script
```bash
python Day6_Data_Augmentation_Dropout.py
```

This will:
- Train the model
- Show augmentation & training plots
- Save the model file

---

## 📄 File Structure

```
.
├── Day6_Data_Augmentation_Dropout.py   # Main script
├── mnist_cnn_model.h5                  # Saved model (after running)
├── README.md                           # Project documentation
```

---

## 🙋‍♂️ Author

**Yash Pal**  
🎓 Computer Science (AI/ML) Student  
🌐 [LinkedIn](https://www.linkedin.com/in/yash-pal-since2004) | 🧠 Passionate about AI & Deep Learning

---

## 📜 License

This project is open-source and free to use for educational and research purposes.
