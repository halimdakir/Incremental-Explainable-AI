# Exploring Incremental Explainable AI

## Author
**Halim Dakir**  
*Department of Computer Science*  
*Blekinge Institute of Technology*  
*Karlskrona, Sweden*

## Project Overview
This project explores the impact of **incremental learning** on **explainability** in deep learning models, specifically investigating **explanation drift**, the phenomenon where model explanations shift unpredictably despite stable accuracy. The study uses **Grad-CAM** to visualize model explanations over multiple training phases and evaluates stability using correlation and **Intersection-over-Union (IoU) metrics**.

## Key Features
- **Incremental Learning Setup**: The model is trained in three phases with progressively increasing class complexity.
- **Explainability with Grad-CAM**: Visualizes CNN feature importance at each phase.
- **Stability Metrics**: Computes correlation and IoU scores to measure explanation consistency across phases.
- **Heatmap Visualizations**: Displays Grad-CAM heatmaps for better interpretability.

## **Workflow Diagram**
Below is a high-level workflow diagram of the incremental XAI experiment:

![diagram](https://github.com/user-attachments/assets/64331338-ab2e-4448-8f57-48286bf47c58)


## Dataset
The project uses the **MNIST** dataset of handwritten digits, partitioned into three incremental phases:
1. **Phase 1**: Classes [0-4], balanced.
2. **Phase 2**: Classes [0-6], imbalanced (over-representing classes 5 and 6).
3. **Phase 3**: Classes [0-9], rebalanced.

The **MNIST dataset**, available at:
[MNIST Dataset - Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## Model Architecture
A **Convolutional Neural Network (CNN)** is used, consisting of:
- **2 Convolutional layers** with ReLU activation
- **Max-pooling layers** for feature extraction
- **Fully connected layers** for classification

## Training Process
- **Phase 1**: Model is trained from scratch on classes [0-4].
- **Phase 2**: New classes [5,6] are introduced and the model is fine-tuned.
- **Phase 3**: All classes [0-9] are incorporated, and the model is further refined.

## Explanation Stability Analysis
- **Grad-CAM Heatmaps**: Generated for each phase to examine shifts in model explanations.
- **IoU and Correlation Metrics**: Used to assess consistency in feature importance across training phases.
- **Comparison of Explanation Drift**: Visualized using histograms and distribution plots.

## Installation
### Dependencies:
Ensure you have the following Python libraries installed:
```bash
pip install torch torchvision captum numpy matplotlib
```

## Running the Project
### Step 1: Load the MNIST dataset
```python
train_images = load_images_idx('data/train-images.idx3-ubyte')
train_labels = load_labels_idx('data/train-labels.idx1-ubyte')
```

### Step 2: Train the CNN Model Incrementally
```python
model_phase1 = SimpleCNN(num_classes=10)
train_model(model_phase1, phase1_loader, epochs=15, lr=0.001)
torch.save(model_phase1.state_dict(), "model_phase1.pth")
```
Repeat for Phase 2 and Phase 3 by loading previous weights and fine-tuning.

### Step 3: Generate Grad-CAM Explanations
```python
hm_p1 = generate_gradcam(model_phase1, img_t, lbl)
hm_p2 = generate_gradcam(model_phase2, img_t, lbl)
hm_p3 = generate_gradcam(model_phase3, img_t, lbl)
show_heatmap(hm_p1, title="Phase 1 Explanation")
```

### Step 4: Compute Stability Metrics
```python
iou_score = compute_iou(hm_p1, hm_p2)
corr_score = heatmap_corr(hm_p1, hm_p2)
print(f"IoU Score: {iou_score}, Correlation Score: {corr_score}")
```

## Results
- **Model Accuracy**: Increases across phases, reaching **98%** in the final phase.
- **Explanation Drift**: Detected via correlation and IoU scores, indicating shifting feature importance.
- **Heatmap Visualization**: Demonstrates how explanations evolve over time, highlighting the need for adaptive XAI methods.

## Conclusion
This project demonstrates that **incremental learning** affects model explanations, making **explanation drift** a critical issue in XAI. The results emphasize the importance of developing **adaptive XAI techniques** that ensure interpretability remains stable as models evolve.

## Future Work
- Extending the approach to **more complex datasets** like ImageNet.
- Investigating explanation drift in **transformer-based models**.
- Implementing **adaptive regularization techniques** to improve stability.

## Contact
For questions or collaboration, please contact **Halim Dakir** at [hada24@student.bth.se].
