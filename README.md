
# Handwritten Mathematical Expression Recognition

This repository contains code for training and testing handwritten mathematical expression recognition models using deep learning techniques.

## Datasets

This project supports multiple datasets for handwritten mathematical expression recognition:

### CROHME Dataset
The Competition on Recognition of Offline Handwritten Mathematical Expressions (CROHME) dataset is a widely-used benchmark for mathematical expression recognition. It contains handwritten mathematical expressions with their corresponding LaTeX representations.

### HME100K Dataset
The HME100K dataset is another comprehensive dataset for handwritten mathematical expression recognition.

## Sample Data

### CROHME Dataset Examples

Below are sample images from the CROHME dataset with their corresponding LaTeX labels:

**Sample 1:**
![Sample 1](path/to/sample1.png)
- **Label:** `x^2 + y^2 = r^2`
- **Description:** Equation of a circle

**Sample 2:**
![Sample 2](path/to/sample2.png)
- **Label:** `\frac{a}{b} + \frac{c}{d} = \frac{ad + bc}{bd}`
- **Description:** Addition of fractions


**Hardware Requirements:**
- GPU with at least 32GB RAM (recommended for default batch size of 8)
- For smaller GPUs, consider reducing the batch size in `config.yaml`

## Training

1. **Configure Training Parameters:**
   - Check and modify the configuration file `config.yaml` as needed
   - Adjust batch size based on your GPU memory capacity
   - Set other hyperparameters according to your requirements

2. **Start Training:**
   ```bash
   python train.py --dataset CROHME
   ```

   **Note:** The default batch size is set to 8, which requires approximately 32GB of GPU RAM. If you have less GPU memory available, reduce the batch size in the config file.

### Training Configuration

Key parameters in `config.yaml`:
- `batch_size`: Number of samples per batch (default: 8)
- `learning_rate`: Learning rate for optimization
- `epochs`: Number of training epochs
- `model_architecture`: Model configuration settings

## Testing/Inference

1. **Prepare for Testing:**
   - Fill in the `checkpoint` parameter in `config.yaml` with the path to your pretrained model
   - Ensure the test dataset is properly located in `datasets/`

2. **Run Inference:**
   ```bash
   python inference.py --dataset CROHME
   ```

## Model Checkpoints

- Trained models will be saved in the `checkpoints/` directory
- Use the best performing checkpoint for inference
- Model checkpoints include both the model weights and configuration


## Troubleshooting

**Common Issues:**

1. **Out of Memory Error:**
   - Reduce batch size in `config.yaml`
   - Use gradient accumulation if available

2. **Dataset Not Found:**
   - Verify dataset paths in `config.yaml`
   - Ensure datasets are extracted in the correct directory structure

3. **Checkpoint Loading Error:**
   - Check the checkpoint path in `config.yaml`
   - Ensure the checkpoint file is not corrupted

## Acknowledgments

- CROHME competition organizers for providing the dataset
- HME100K dataset creators
- Contributors to the open-source community
