import torch
print(torch.__version__)

print("GPU Configurada:", torch.cuda.is_available())
print("Total de GPUs", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Atual:", torch.cuda.current_device())
    print("Device", torch.cuda.device(0))
    print("Device Name", torch.cuda.get_device_name(0))
else:
    print("Nenhuma GPU configurada")
