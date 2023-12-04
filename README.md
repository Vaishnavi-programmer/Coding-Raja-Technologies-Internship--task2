# Image Classification using Fine-Tuned ResNet

## Overview

This repository contains code for image classification using a Fine-Tuned ResNet model. The model is trained on a dataset of Indian food images, exploring various aspects such as data preprocessing, model creation, training, evaluation, and visualization of results.

## Requirements

- Python 3
- PyTorch
- scikit-learn
- matplotlib
- seaborn

## Usage

### 1. Data Preparation:

Ensure that the dataset of Indian food images is available at the specified path: `/kaggle/input/indian-food-images-dataset/Indian Food Images/Indian Food Images`. Adjust the path if necessary.

### 2. Training the Model:

The model is a Fine-Tuned ResNet18. The training loop is provided with a specified number of epochs. Adjust the parameters as needed.

```python
# Training loop example
num_epochs = 10  

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples

    scheduler.step()

    # Print average loss and accuracy per epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy * 100:.2f}%")
    
```
### 3. Evaluation

The model is evaluated on a test set, and performance metrics such as accuracy, precision, recall, and the confusion matrix are displayed.

```python
# Model evaluation example
true_labels, predicted_labels = evaluate_model(model, test_loader)
```

### 4. Visualizing Predictions:

The repository includes functions to visualize model predictions on sample images from the test set. The predictions, along with actual labels, are displayed for better understanding.

```python
# Visualization of predictions example
visualize_predictions(model, test_loader, num_images=5)
```

### 5. Exploring Misclassified Images:

Visualize misclassified images to gain insights into areas of improvement for the model. This can help refine the training process.

```python
# Visualization of misclassified images example
visualize_misclassifications(model, test_loader, num_images=5)
```

## Additional Information

The dataset is transformed using random rotation, horizontal flip, resizing, center cropping, and normalization before being fed into the model.

The Fine-Tuned ResNet18 model consists of feature extraction layers from a pre-trained ResNet18 model followed by a fully connected layer for classification.

The code includes a learning rate scheduler and utilizes the SGD optimizer with momentum.

## Results

The code provides insights into the training process, model evaluation, and visualizations to aid in understanding the model's performance on the Indian food image dataset.

Feel free to explore, modify, and adapt the code for your specific image classification tasks. If you encounter any issues or have suggestions for improvements, please let us know!


