import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

def get_celeba_dataloader(batch_size=64, image_size=128):
    """
    Creates a PyTorch DataLoader using the Hugging Face 'flwrlabs/celeba' dataset,
    with a standard and robust batching strategy.
    """
    print("--- Using Hugging Face 'flwrlabs/celeba' dataset in streaming mode ---")
    
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def transform_example(example):
        """Applies transformations to a single example from the dataset."""
        example['image'] = preprocess(example['image'].convert("RGB"))
        return example

    print("Initializing dataset stream...")
    dataset = load_dataset("flwrlabs/celeba", split="train", streaming=True)
    
    shuffled_dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    transformed_dataset = shuffled_dataset.map(transform_example)
    
    final_dataset = transformed_dataset.with_format("torch")


    dataloader = DataLoader(
        final_dataset,
        batch_size=batch_size,
        num_workers=0,  
    )

    print("Hugging Face DataLoader created successfully!")
    return dataloader

if __name__ == '__main__':
    print("\n--- Running verification test ---")
    test_loader = get_celeba_dataloader(batch_size=4)
    first_batch = next(iter(test_loader))
    images = first_batch['image']
    print(f"Success! Shape of one batch is: {images.shape}")
    print("This is now a 4D tensor [Batch, Channels, Height, Width], which is correct.")