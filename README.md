# face
A comprehensive collection of face AI models with integrated pre and post-processing steps, utilizing NVIDIA Triton Inference Server for seamless inference. This repository aims to provide easy-to-use face detection, recognition, and analysis tools.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Available Models](#available-models)
- [Pre and Post Processing](#pre-and-post-processing)
- [Disclaimer](#disclaimer)
- [License](#license)

## Introduction
This repository contains a suite of face AI models designed for various applications, such as face detection, recognition, and analysis. The models are optimized for performance and ease of use, leveraging NVIDIA Triton Inference Server for scalable and efficient inference.

## Features
- <b>Diverse Collection of Models:</b> Includes models for face detection, recognition, age estimation, feature embedding, and landmark detection.
- <b>Integrated Pre and Post-Processing:</b>  Ensures consistent and accurate results across different models.
- <b>Triton Inference Server Integration:</b>  Facilitates efficient and scalable model deployment.
- <b>Easy-to-Use Interface:</b>  Simple API for quick integration into various applications.

## Installation
1. Clone the Repository:

    ```bash
    git clone https://github.com/dev6699/face.git
    cd face
    ```

2. Open the repository in vscode devcontainer.

3. Download and Prepare Models:
    - Navigate to the [Available Models](#available-models) section to find the download links for each model.
    - Download each model and rename the file to `model.onnx`.
    - Place each `model.onnx` file into its respective directory within the model_repository folder.
    - Example: Setting up the YOLOFace model:

        ```bash
        mkdir -p model_repository/yoloface/1
        wget -O model_repository/yoloface/1/model.onnx <model_url>
        ```
        Ensure to replace <model_url> with the actual URL provided in the [Available Models](#available-models) section.

## Usage
1. Start Triton Inference Server:

    ```bash
    docker-compose up tritonserver
    ```

## Available Models
### Gender and Age Estimation
- Model Name: gender_age
- Description: Detects gender and estimates the age of detected faces.
- Download Link: [Download Gender and Age Estimation Model](https://github.com/facefusion/facefusion-assets/releases/download/models/yoloface_8n.onnx)

    <img src="docs/gender_age.jpg" height=200>

### YOLOFace
- Model Name: yoloface
- Description: Detects face bounding boxes and the 5 key facial landmarks (landmark5) using the YOLO architecture.
- Download Link: [Download YOLOFace Model](https://github.com/facefusion/facefusion-assets/releases/download/models/yoloface_8n.onnx)

    <img src="docs/yoloface.jpg" height=200>

## Pre and Post Processing
### Pre-processing
Each model has specific pre-processing steps to ensure accurate results. Common steps include:

- <b>Resizing:</b> Scaling the input image to the required dimensions.
- <b>Normalization:</b> Adjusting pixel values to a standard range.
- <b>Face Alignment:</b> Aligning faces for consistent feature extraction.

### Post-processing
Post-processing varies by model and typically includes:

- <b>Bounding Box Extraction:</b> Extracting face locations from detection models.
- <b>Feature Extraction:</b> Computing facial features for recognition models.
- <b>Label Assignment:</b> Assigning predicted labels or scores.

## Disclaimer
Model assets are subject to their individual licenses. Ensure that you review and comply with the specific license terms for each model you use. The repository does not grant rights to use third-party models beyond the scope defined in their respective licenses.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
