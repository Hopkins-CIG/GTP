import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from src.utils import get_project_root

def get_sam_encoder(checkpoint_path=None):
    root_dir = get_project_root()
    if checkpoint_path is None:
        checkpoint_path = f"{root_dir}/checkpoints/sam.pth"
    sam_encoder = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam_encoder.eval()
    sam_encoder.image_encoder.requires_grad_(False)
    return sam_encoder

@torch.no_grad()
def sam_encode_img(sam_encoder, img):
    # img has shape (b, 1, H, W) and values in [-1, 1]
    img = (img + 1) / 2  # scale from [-1, 1] to [0, 1]
    img_3c = torch.cat([img] * 3, dim=1)  # (b, 3, H, W)
    img_1024 = F.interpolate(img_3c, size=(1024, 1024), mode='bilinear', align_corners=False)  # (b, 3, 1024, 1024)
    image_embedding = []
    for i in range(img_1024.shape[0]):
        emb = sam_encoder.image_encoder(img_1024[i].unsqueeze(0))
        image_embedding.append(emb)
    image_embedding = torch.vstack(image_embedding)
    image_embedding = F.interpolate(image_embedding, size=(256, 256), mode='bilinear', align_corners=False)
    return F.instance_norm(image_embedding)

def random_group_average(mr_enc, n_groups=8):
    # mr_enc has shape (B, C, H, W)
    n_channels = mr_enc.shape[1]
    indices = torch.randperm(n_channels)
    mr_enc = torch.stack([mr_enc[:, indices[i::n_groups], :, :].mean(1) for i in range(n_groups)], dim=1)
    return mr_enc