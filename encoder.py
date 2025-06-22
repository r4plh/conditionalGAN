import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetEncoder(nn.Module):
    """
    A pure PyTorch encoder using a pre-trained FaceNet model.
    This is much more stable than the dlib-based face_recognition library.
    """
    def __init__(self, device):
        super(FaceNetEncoder, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').to(device)
        self.model.eval() 

    def forward(self, image_batch):
        """
        Args:
            image_batch (torch.Tensor): A batch of images of shape (N, C, H, W)
                                        normalized to [-1, 1].
        
        Returns:
            torch.Tensor: A tensor of face embeddings of shape (N, 512).
        """
        embeddings = self.model(image_batch)
        return embeddings

if __name__ == '__main__':
    # --- Verification Test ---
    print("--- Testing the FaceNetEncoder ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = FaceNetEncoder(device=device)
    dummy_batch = torch.randn(4, 3, 128, 128).to(device)
    print(f"Input batch shape: {dummy_batch.shape}")
    
    embeddings = encoder(dummy_batch)
    
    print(f"Output embeddings shape: {embeddings.shape}") # Should be (4, 512)
    print("\nFaceNetEncoder test successful!")