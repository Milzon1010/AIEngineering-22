# Homework: Practical Hyperparameter Tuning

## Dataset
Fashion MNIST (70k grayscale images, 10 classes)

## Baseline Model
- Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
- Optimizer: Adam (lr=0.001)
- Epochs: 10, Batch size: 32
- Test Accuracy: ~87.7%

## Experiment Results

| Name         | LR    | Hidden | Dropout | Extra Layer | Batch | Test Acc | Best Val Acc | Best Train Acc | Overfit Gap |
|--------------|-------|--------|---------|-------------|-------|----------|--------------|----------------|-------------|
| baseline     | 0.001 | 128    | 0.2     | No          | 32    | 0.8766   | 0.8878       | 0.8929         | 0.0051      |
| hidden_64    | 0.001 | 64     | 0.2     | No          | 32    | 0.8757   | 0.8855       | 0.8808         | -0.0047     |
| hidden_256   | 0.001 | 256    | 0.2     | No          | 32    | 0.8792   | 0.8878       | 0.9004         | 0.0125      |
| dropout_0p1  | 0.001 | 128    | 0.1     | No          | 32    | 0.8785   | 0.8870       | 0.9029         | 0.0159      |
| dropout_0p5  | 0.001 | 128    | 0.5     | No          | 32    | 0.8717   | 0.8780       | 0.8665         | -0.0115     |
| batch_16     | 0.001 | 128    | 0.2     | No          | 16    | 0.8768   | 0.8830       | 0.8910         | 0.0080      |
| batch_128    | 0.001 | 128    | 0.2     | No          | 128   | 0.8787   | 0.8850       | 0.8924         | 0.0074      |
| lr_small     | 0.0001| 128    | 0.2     | No          | 32    | 0.8678   | 0.8757       | 0.8766         | 0.0009      |
| lr_large     | 0.01  | 128    | 0.2     | No          | 32    | 0.8439   | 0.8468       | 0.7965         | -0.0503     |
| extra_layer  | 0.001 | 128    | 0.2     | Yes         | 32    | 0.8816   | 0.8872       | 0.9031         | 0.0159      |

## Insights
- ✅ **Best performer**: Extra hidden layer (Test Acc ~88.16%).
- 📊 Increasing neurons (256) helped a bit but not as much as adding an extra layer.
- 🔄 Batch size 128 sped up training (18s vs 50s) with similar accuracy.
- ⚠️ Learning rate 0.01 caused divergence (accuracy drop to 84%).
- 🎯 Overfitting gap stayed small (<2%), meaning regularization (dropout) worked well.
