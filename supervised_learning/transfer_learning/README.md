# Transfer Learning for CIFAR-10 with ResNet50V2

> **Machine Learning Experiment — Transfer Learning & Image Classification**

![Transfer Learning and Convolutional Neural Network](https://upload.wikimedia.org/wikipedia/commons/9/96/Neural_network.svg)

---

## Table of Contents

- [Abstract](#-abstract)
- [1. Introduction](#1-introduction)
- [2. Materials and Methods](#2-materials-and-methods)
  - [2.1 Dataset](#21-dataset)
  - [2.2 Data Preprocessing](#22-data-preprocessing)
  - [2.3 Transfer Learning Architecture](#23-transfer-learning-architecture)
  - [2.4 Training Configuration](#24-training-configuration)
- [3. Results](#3-results)
- [4. Discussion](#4-discussion)
- [5. Conclusion](#5-conclusion)
- [6. Acknowledgments](#6-acknowledgments)
- [7. Literature Cited](#7-literature-cited)
- [Appendix A](#appendix-a--final-preprocessing-function)
- [Appendix B](#appendix-b--main-experimental-observations)

---

# Abstract

The goal of this experiment was to build a convolutional neural network capable of classifying the **CIFAR-10 dataset** while achieving at least **87% validation accuracy**.

Instead of training a convolutional network from scratch, I used **transfer learning** with the ImageNet-pretrained **ResNet50V2** architecture available through Keras Applications.

CIFAR-10 contains small **32×32 RGB images**, whereas ResNet50V2 expects significantly larger inputs. I therefore resized the images to **160×160** and applied the preprocessing expected by ResNet50V2.

The pretrained ResNet50V2 layers were frozen, and a new classification head was trained for the ten CIFAR-10 classes.

### Final Results

| Metric | Result |
|---|---:|
| Maximum validation accuracy | **87.61%** |
| Final validation accuracy | **87.33%** |
| Test accuracy | **87.33%** |
| Required accuracy | **≥ 87%** |
| Requirement achieved | **Yes** |

The saved model was subsequently evaluated on the CIFAR-10 test set and obtained **87.33% accuracy**.

---

# 1. Introduction

Image classification is one of the fundamental problems in computer vision. Given an image, the objective is to predict which class it belongs to.

In this project, the task was to classify images from the **CIFAR-10 dataset** into ten categories.

CIFAR-10 contains **60,000 colour images** divided into ten classes. Each image is only **32×32 pixels**, which makes the dataset relatively small compared with datasets such as ImageNet.

A straightforward solution would be to design and train a convolutional neural network from scratch. However, this requires considerable computational resources and training time.

I therefore investigated whether a network that had already learned useful visual representations could be adapted to CIFAR-10.

This is the principle behind **transfer learning**. A model trained on a large dataset such as ImageNet has already learned features such as:

- edges;
- textures;
- shapes;
- increasingly complex visual patterns.

Instead of learning all these representations again, I reused them and trained only a new classification head.

For this experiment, I selected **ResNet50V2**. ResNet architectures introduced residual connections that make it easier to train very deep neural networks. He et al. demonstrated that residual learning allows substantially deeper networks to be optimized effectively.

### 🎯 Research Question

> **Can an ImageNet-pretrained ResNet50V2 be adapted to CIFAR-10 and achieve at least 87% validation accuracy?**

---

# 2. Materials and Methods

## 2.1 Dataset

I used the CIFAR-10 dataset provided by Keras:

```python
(X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
```

The dataset contains:

| Dataset | Number of images |
|---|---:|
| Training set | 50,000 |
| Test set | 10,000 |
| **Total** | **60,000** |

Each image has the shape:

```text
32 × 32 × 3
```

where the three channels correspond to RGB.

The original labels are integer values between `0` and `9`.

---

## 2.2 Data Preprocessing

The first challenge was the difference between CIFAR-10's image size and the expected input size of the pretrained network.

I used:

```python
X_p = K.applications.resnet_v2.preprocess_input(X)
```

to apply the preprocessing expected by ResNetV2.

The labels were converted to one-hot encoded vectors:

```python
Y_p = K.utils.to_categorical(Y, 10)
```

### Example

An original label:

```text
3
```

becomes:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

This representation is appropriate for the categorical cross-entropy loss used by the classifier.

---

## 2.3 Transfer Learning Architecture

I loaded ResNet50V2 with ImageNet weights:

```python
model = K.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(160, 160, 3)
)
```

The original ImageNet classification layers were removed using:

```python
include_top=False
```

I then froze the pretrained network:

```python
model.trainable = False
```

This meant that the parameters learned from ImageNet were not modified during training.

### Input Resizing

Because CIFAR-10 images are only `32×32` pixels, I added a Lambda layer to resize them:

```python
resized = K.layers.Lambda(
    lambda img: K.layers.Resizing(160, 160)(img)
)(final_input)
```

### Classification Head

The output of ResNet50V2 was connected to a new classification head:

```python
x = K.layers.GlobalAveragePooling2D()(model_out)
x = K.layers.Dense(512, activation='relu')(x)
x = K.layers.Dropout(0.3)(x)
final_out = K.layers.Dense(10, activation='softmax')(x)
```

### Model Architecture

```text
┌──────────────────────┐
│    CIFAR-10 image    │
│      32 × 32 × 3     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Resize to 160 × 160  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     ResNet50V2       │
│  ImageNet pretrained │
│       FROZEN         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Global Average Pool  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dense(512, ReLU)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dropout(0.3)         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dense(10, Softmax)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   CIFAR-10 class     │
└──────────────────────┘
```

ResNet's residual architecture was originally designed to make very deep networks easier to optimize, and later work on identity mappings further investigated why these connections improve signal propagation and generalization.

---

## 2.4 Training Configuration

I compiled the model using the Adam optimizer:

```python
cifar10_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Model | ResNet50V2 |
| Pretrained weights | ImageNet |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Batch size | 64 |
| Epochs | 10 |
| Dropout | 0.3 |
| Frozen backbone | ResNet50V2 |
| Trainable part | Classification head |
| Number of classes | 10 |

I initially encountered computational limitations when attempting to train the model locally. The process was terminated with:

```text
Killed
```

This suggested that the available resources were insufficient.

I therefore moved the experiment to **Google Colab**.

---

# 3. Results

The model's training history was as follows:

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|:---:|---:|---:|---:|
| 1 | 83.08% | 86.01% | 0.4066 |
| 2 | 87.67% | 86.95% | 0.3908 |
| 3 | 89.75% | 87.22% | 0.3885 |
| **4** | **90.98%** | **87.61%** | **0.3954** |
| 5 | 92.12% | 87.48% | 0.4078 |
| 6 | 93.31% | 86.95% | 0.4545 |
| 7 | 94.43% | 87.53% | 0.4749 |
| 8 | 95.00% | 87.53% | 0.4791 |
| 9 | 95.65% | 87.62% | 0.4878 |
| 10 | 95.94% | **87.33%** | 0.5283 |

The highest validation accuracy was **87.61%**, achieved during epoch 4.

Training accuracy continued to increase after epoch 4, reaching **95.94%**, while validation accuracy remained around 87%.

This indicates that the model was beginning to **overfit the training data**.

### Test Set Evaluation

After training, the saved `cifar10.h5` model was evaluated on the **10,000-image CIFAR-10 test set**.

The final result was:

```text
Loss:     0.5283
Accuracy: 0.8733
```

Therefore:

> ### Test accuracy = **87.33%**

This exceeds the project's required threshold of **87%**.

---

# 4. Discussion

The experiment demonstrates that **transfer learning can be effective even when the target dataset differs significantly from the dataset used to pretrain the network**.

One of the main advantages was that ResNet50V2 already contained useful visual representations learned from ImageNet.

Rather than training hundreds of layers from scratch, I froze the pretrained backbone and only trained a relatively small classification head.

## Overfitting

The results revealed an interesting training behaviour.

Training accuracy increased almost continuously:

```text
83.08% → 95.94%
```

while validation accuracy remained around:

```text
86–88%
```

The best validation result occurred relatively early:

```text
Epoch 4 → 87.61%
```

After that point, increasing training accuracy did not translate into better validation performance.

This is a classic sign that the model was learning the training set increasingly well without obtaining equivalent improvements on unseen data.

---

## Computational Efficiency

Another important observation concerns computational efficiency.

My initial implementation passed the entire dataset through the frozen ResNet50V2 during every epoch.

Although the pretrained layers were frozen, their forward pass was still repeated.

The project hint suggests a more efficient strategy: compute the outputs of the frozen layers once, store those feature representations, and then train the classification head using those features.

### Current approach

```text
Epoch 1  → ResNet → classifier
Epoch 2  → ResNet → classifier
Epoch 3  → ResNet → classifier
...
Epoch 10 → ResNet → classifier
```

### Optimized approach

```text
              ┌───────────┐
              │  ResNet   │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │  Features │
              └─────┬─────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ Classifier training │
        └─────────────────────┘
```

The second approach would significantly reduce the amount of computation required.

---

## Keras Compatibility Issue

Finally, I encountered a compatibility issue when loading the `.h5` file with a modern version of Keras.

The model contained a Python `Lambda` layer, and Keras 3 introduced stricter deserialization behaviour for Lambda functions.

I was nevertheless able to reconstruct the architecture and load the trained weights.

The resulting model produced the expected:

```text
87.33% test accuracy
```

This confirmed that the trained weights were valid.

The issue was therefore **not a failure of the trained model**, but a compatibility problem between the older `.h5` serialization approach and modern Keras.

---

# 5. Conclusion

The objective of the experiment was to create a CIFAR-10 image classifier with at least **87% validation accuracy** using a pretrained Keras application.

## 🏆 Final Performance

| Metric | Result | Target | Status |
|---|---:|---:|:---:|
| Maximum validation accuracy | **87.61%** | ≥ 87% | ✅ |
| Final validation accuracy | **87.33%** | ≥ 87% | ✅ |
| Test accuracy | **87.33%** | — | ✅ |

Therefore, the required performance threshold was successfully reached.

The experiment also demonstrated several practical aspects of machine learning beyond simply training a model:

- selecting a suitable pretrained architecture;
- adapting input dimensions;
- preprocessing data correctly;
- freezing pretrained layers;
- monitoring overfitting;
- dealing with computational limitations;
- handling model compatibility between different versions of Keras.

### 💡 Main Takeaway

> **Transfer learning can turn an expensive image-classification problem into a much more manageable one by reusing representations learned from a much larger dataset.**

---

# 6. Acknowledgments

I used the Keras implementation of **ResNet50V2** and the **CIFAR-10** dataset provided through Keras.

I also used **Google Colab** to provide sufficient computational resources for the experiment.

---

# 7. Literature Cited

1. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**  
   *Deep Residual Learning for Image Recognition.*  
   Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.

2. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**  
   *Identity Mappings in Deep Residual Networks.*  
   European Conference on Computer Vision (ECCV), 630–645.

3. **Keras Applications — ResNet50V2.**  
   Keras documentation and pretrained ImageNet weights.

---

# Appendix A : Final Preprocessing Function

The preprocessing function used in the experiment was:

```python
def preprocess_data(X, Y):
    X_p = K.applications.resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p
```

---

# Appendix B : Main Experimental Observations

- 🟩 A pretrained **ResNet50V2** was selected instead of training a CNN from scratch.
- 🟩 CIFAR-10 images had to be resized before being passed to ResNet50V2.
- 🟩 The pretrained ResNet50V2 layers were frozen.
- 🟩 A new 10-class classification head was trained.
- 🟩 The model reached **87.61% validation accuracy**.
- 🟩 The final saved model achieved **87.33% accuracy** on the CIFAR-10 test set.
- 🟥 Increasing training accuracy after epoch 4 did not improve validation accuracy, suggesting overfitting.
- 🟧 Computing the frozen ResNet features only once would be a more computationally efficient implementation.
