import torch.backends.quantized

from quantization_utils import *
quantized_model_filepath = "./quantized_models/mobilenetv2_qat_quantized.pt"
cpu_device = "cpu"

# Load quantized model.
quantized_jit_model = load_torchscript_model(
    model_filepath=quantized_model_filepath, device=cpu_device)

data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
val_data_dir = "/workspace/kli/ILSVRC2012/valid"
val_label_file = "./data/val.txt"
images, labels = get_images_and_labels(val_label_file)
test_dataset = valDataset(val_data_dir, images, labels,
                          transform=data_transform_test)
test_sampler = torch.utils.data.SequentialSampler(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

BACKEND = "fbgemm"
torch.backends.quantized.engine = BACKEND
_, int8_eval_accuracy = evaluate_model(
        model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)
print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

