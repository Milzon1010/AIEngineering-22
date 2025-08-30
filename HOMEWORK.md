# Homework: Practical Hyperparameter Tuning

## Objective

The goal of this assignment is to gain hands-on experience with hyperparameter tuning for a neural network. You will build a simple classifier and experiment with different hyperparameters to see how they affect the model's performance.

## The Task

You will be working with the **Fashion MNIST** dataset, which is a collection of 70,000 grayscale images of 10 different types of clothing. Your task is to build a neural network that can classify these images.

### Steps:

1.  **Set up your environment:**
    -   You will need Python with libraries such as TensorFlow, Keras, and scikit-learn.
    -   Load the Fashion MNIST dataset. It is available directly from TensorFlow/Keras.
    -   Preprocess the data:
        -   Normalize the pixel values to be between 0 and 1.
        -   Split the data into training, validation, and test sets.

2.  **Build a Baseline Model:**
    -   Create a simple neural network with the following architecture:
        -   A Flatten layer to convert the 28x28 images into a 1D array.
        -   A Dense (fully-connected) hidden layer with 128 neurons and a ReLU activation function.
        -   A Dropout layer with a rate of 0.2.
        -   An output layer with 10 neurons (one for each class) and a Softmax activation function.
    -   Compile the model using the Adam optimizer, a learning rate of 0.001, and the `sparse_categorical_crossentropy` loss function.
    -   Train the model for 10 epochs with a batch size of 32.
    -   Evaluate the model on the test set and record the accuracy. This is your baseline performance.

3.  **Tune the Hyperparameters:**
    -   Now, you will experiment with changing the hyperparameters of your model. You should try at least **three** different experiments. For each experiment, change **one** hyperparameter at a time to observe its effect.
    -   Here are some ideas for experiments:
        -   **Change the learning rate:** Try a smaller learning rate (e.g., 0.0001) and a larger one (e.g., 0.01).
        -   **Change the number of neurons in the hidden layer:** Try a smaller number (e.g., 64) and a larger number (e.g., 256).
        -   **Change the dropout rate:** Try a lower dropout rate (e.g., 0.1) and a higher one (e.g., 0.5).
        -   **Add another hidden layer:** Make your network deeper.
        -   **Change the batch size:** Try a smaller batch size (e.g., 16) and a larger one (e.g., 128).

4.  **Analyze and Report Your Findings:**
    -   For each experiment, record the hyperparameters you used and the resulting test accuracy.
    -   Create a summary table that compares the performance of your different models.
    -   Write a brief report that answers the following questions:
        -   Which hyperparameter had the most significant impact on the model's performance?
        -   How did the changes you made affect the training process (e.g., did the model train faster or slower? Did it overfit?)?
        -   What was the best set of hyperparameters you found, and what was the final test accuracy?

## Submission

Submit your code (e.g., a Jupyter Notebook or Python script) and your report. Your code should be well-commented, and your report should be clear and concise.
