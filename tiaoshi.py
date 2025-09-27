import torch


def check_cuda():
    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # 获取 CUDA 版本
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")

        # 获取 GPU 设备信息
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")


if __name__ == "__main__":
    check_cuda()
