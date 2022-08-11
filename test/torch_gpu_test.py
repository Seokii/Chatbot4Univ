import torch

print(torch.cuda.is_available())
# False


# nvcc -V 결과는 11.2

# Linux version -> 18.04.6 LTS

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113