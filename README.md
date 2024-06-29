# Automated-Image-Captioning-with-Custom-Data-Generator
This project implements an automated image captioning system using deep learning. A custom data generator preprocesses image-caption pairs, handles batching, shuffling, and sequence conversion. The system trains a neural network to generate descriptive captions for images, enhancing image understanding and description tasks.
- Brief Description of the Code
The provided Jupyter Notebook appears to focus on developing an automated image captioning system. This involves using a custom data generator class to prepare image-caption pairs for training a deep learning model. The key functionality includes:

Custom Data Generator: This class is designed to handle the loading and preprocessing of image data and corresponding captions, ensuring that the data is fed to the model in batches.
Initialization: Sets up the parameters including dataframes, columns, batch size, tokenizer, vocabulary size, maximum length of sequences, and image features.
Shuffling: Ensures data shuffling after each epoch to enhance training.
Batch Retrieval: Retrieves batches of data for model training, converting captions into sequences and preparing them for the model.
Key Features:
- Custom data generator for efficient data handling and preprocessing.
- Integration with deep learning models for image captioning.
- Supports batching and shuffling to enhance model training.
