import torch

MODEL_PATH = r"D:\models\mobilenetv3-large-0.75-9632d2a8.pth"

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
state_dict = checkpoint.get("state_dict", checkpoint)

# Remove a possible DataParallel prefix
state_dict = {
    key.removeprefix("module."): value
    for key, value in state_dict.items()
}
for key, tensor in state_dict.items():
    if key.startswith("features") or key.startswith("classifier"):
        print(f"{key:55} {tuple(tensor.shape)}")