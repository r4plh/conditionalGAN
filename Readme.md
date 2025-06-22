# Conditional Face Generation with Pre-trained FaceNet Embeddings


This project demonstrates the rapid development of a high-quality conditional Generative Adversarial Network (cGAN) for generating human faces. The model is conditioned on facial embeddings extracted by a pre-trained FaceNet model, allowing it to generate faces that retain key identity features.

The primary goal was to simulate a one-day development cycle to build a production-viable model, emphasizing efficient design choices, robust tooling, and clear, metric-driven evaluation.


## Design Philosophy & Ingenuity

1.  **Leveraging Pre-trained Encoders (Model Choice):** Instead of training a standard GAN from scratch, we opted for a Conditional GAN. The key ingenuity is using a pre-trained **FaceNet (InceptionResnetV1)** model as a fixed feature extractor. This approach injects powerful, pre-existing knowledge of facial identity into the generator, dramatically accelerating training and improving the quality and coherence of the generated faces. The generator learns not just to create faces, but to create faces that match a given high-level identity vector. This was the embedding (kind of unique fingerprint which was feeded to generator and discriminator) and is analogous to BERT embedding in NLP. Have used a strong pre-tarined model (in eval mode) to feed the embeddings of face, no training of encoder is done in the pipeline (during training, inference and getting score result, all the time encoder was used in eval mode only.)

2.  **Efficient Data Handling (Dataset Choice):** We use the Hugging Face `datasets` library to stream the CelebA dataset (`flwrlabs/celeba`) directly from the hub (`streaming=True`). This production-ready approach is highly efficient as it avoids the need to download and store the entire \~22GB dataset locally, saving time and disk space. The celebA dataset is popular dataset in tasks involving human faces and is a huge dataset with train and test split, I found this dataset most suitable for training and test.

3.  **Robust Tooling & Compute:** The model was trained on an Apple Silicon Mac using the `mps` (Metal Performance Shaders) backend, demonstrating the feasibility of leveraging modern consumer hardware for deep learning tasks. All training and evaluation metrics were logged using `wandb` for reproducibility and clear reporting.


This was my first time, in which I was tracking the progress through weights and Biases platform.

Pictures from wandb after 2 epoch training was completed are below, the provision to make the wandb public is not there in the weights and biases workspace in which I completed this project, so I'd need to invite people in my team for them to see the results of run involved in this project. For that I would need to get the mail ids to give access and invite in the workspace and then it would be visible to them. I am open to give access to mail id, but the project workspace cannot be made public (It's written down in the W&B settings under my workspace.) Comment mail id and I'll send a invite, then it can be visible.

Wandb (Org.) - amannagrawall002-iit-roorkee-org
Wandb (username) - amannagrawall002

The screenshots on the wandb can be seen in the directory wandb screenshots, the model was built from scratch and 2 epochs were run.

## Model Architecture

The architecture consists of three main components:

1.  **FaceNet Encoder:** A pre-trained `InceptionResnetV1` model (trained on VGGFace2) that takes a 128x128 image and produces a 512-dimensional embedding vector. This encoder is frozen during training.

2.  **Generator:** A DCGAN-style architecture that takes a 100-dim noise vector concatenated with the 512-dim face embedding from the encoder. It uses a series of `ConvTranspose2d` layers to upsample this combined vector into a 128x128x3 image.

3.  **Discriminator:** A conditional discriminator that receives both an image (real or fake) and the corresponding 512-dim face embedding. It processes the image to extract features and then combines them with the embedding to make a final real/fake prediction. This forces the generator to produce images that are not only realistic but also consistent with the conditioning embedding.

The functional model is publicaly available here -:
https://huggingface.co/amannagrawall002/generator_epoch_2.pth/tree/main and model architecture is in the script models.py to instantiate the respective model class.

There are 2 notebooks:

1. inference.ipynb -> It includes the inference on train as well as unseen test images.
2. results.ipynb -> It includes the results on the test dataset of celebA (code with result and all details)

## Results & Evaluation

The model was evaluated on the CelebA test set to measure its ability to generate perceptually convincing faces.

### Quantitative Metrics

We used the **Learned Perceptual Image Patch Similarity (LPIPS)** score, which measures the perceptual distance between the real and generated images. It is a more human-aligned metric than traditional measures like MSE. The score was calculated by comparing a real test image to a new image generated using its embedding.

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **LPIPS** | **0.3981** | (Lower is better) Calculated over 200 batches from the test set. |

This LPIPS score indicates a strong perceptual similarity between the input identity and the generated face, confirming the model's effectiveness.

### Final Training Losses

The final losses after 2 full epochs on the training set were:

  * **Generator Loss**: 3.72521
  * **Discriminator Loss**: 0.79712

## Training Details

  * **Compute:** Apple Silicon (M-series) GPU (`mps` device).
  * **Dataset:** CelebA `train` split (\~162k images), streamed from Hugging Face.
  * **Epochs:** 2
  * **Batch Size:** 64
  * **Batches per Epoch:** 2,525
  * **Total Training Time:** Approx. 4-5 hours.
  * **Preprocessing:** Images were resized and center-cropped to 128x128, then normalized to the range `[-1, 1]`.
  * **Checkpoints:** Model weights were saved after each epoch. The final evaluation uses `generator_epoch_2.pth`.

## Setup and Usage

### 1\. Requirements

The requirements are stated in the requirements.txt file and the env could be created by running the command pip install -r requirements.txt (after the python env is being created)


### 2\. Training

To start training the model from scratch or continue from a checkpoint, run the training script. Configure your `wandb` project and other parameters inside the script.
The model was run for 2 epochs , first we trained it for 1 epoch (train.py).
Then loading the model weights and ran it for epoch 2 (train2.py).

```bash
python train.py
```

### 3\. Evaluation

To evaluate a trained generator checkpoint and calculate the FID/LPIPS scores, use the provided Jupyter Notebook.

```bash
jupyter notebook evaluation.ipynb
```

Make sure your checkpoint file (e.g., `generator_epoch_2.pth`) is located in the `./checkpoints` directory.

## Hyperparameters

The following configuration was used for the final training run:

```python
config = {
    "project_name": "conditional-face-gan-pytorch",
    "num_epochs": 2,
    "batch_size": 64,
    "image_size": 128,
    "noise_dim": 100,
    "embedding_dim": 512,
    "g_lr": 0.0002,
    "d_lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "log_interval": 25,
    "sample_interval": 100,
    "checkpoint_dir": "./checkpoints",
}

# Label smoothing for discriminator stability
real_label = 0.9
fake_label = 0.0
```
