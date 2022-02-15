from nvidia_qat_utils import *
from pytorch_quantization.tensor_quant import QuantDescriptor
from torchvision import models
from quantization_utils import *
from pytorch_quantization import quant_modules
'''for onnx export'''
# quant_nn.TensorQuantizer.use_fb_fake_quant = True

# quant_modules.initialize()

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

# model = models.resnet50(pretrained=True)
model = models.efficientnet_b0(pretrained=True)
model.cuda()




data_transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
data_transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_dataset = torchvision.datasets.ImageFolder(root='/workspace/kli/ILSVRC2012/train',
                                                 transform=data_transform_train)
# train_sampler = torch.utils.data.RandomSampler(train_dataset)
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_data_dir = "/workspace/kli/ILSVRC2012/valid"
val_label_file = "./data/val.txt"
images, labels = get_images_and_labels(val_label_file)
test_dataset = valDataset(val_data_dir, images, labels,
                          transform=data_transform_test)
# test_sampler = torch.utils.data.SequentialSampler(test_dataset)
data_loader_test = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

'''eval fp32 model'''
eval_fp32_flag = False
if eval_fp32_flag:
    _, fp32_eval_accuracy = evaluate_model(
        model=model, test_loader=data_loader_test, device="cuda", criterion=None)
    print("fp32 eval accuracy: {}".format(fp32_eval_accuracy))
    exit()

# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, data_loader, num_batches=32)

# with torch.no_grad():
#     compute_amax(model, method="percentile", percentile=99.99)
#     _, eval_accuracy = evaluate_model(model, data_loader_test, device="cuda")
#     print("calib method: percentile_99.99")
#     print("eval accuracy: {:.3f}".format(eval_accuracy))

# Save the model
export_onnx_flag = False
if export_onnx_flag:
    torch.save(model.state_dict(), "./quant_resnet50-calibrated.pth")
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    input_names = [ "actual_input_1" ]
    output_names = [ "output1" ]
    # enable_onnx_checker needs to be disabled. See notes below.
    torch.onnx.export(
        model, dummy_input, "./quant_resnet50.onnx", verbose=False, opset_version=10, enable_onnx_checker=False)

# with torch.no_grad():
#     compute_amax(model, method="percentile", percentile=99.9)
#     # _, eval_accuracy = evaluate_model(model, data_loader_test, device="cuda")
#     eval_accuracy = 0.713
#     print("calib method: percentile_99.9")
#     print("eval accuracy: {:.3f}".format(eval_accuracy))
#     exit()

# with torch.no_grad():
#     compute_amax(model, method="mse")
#     _, eval_accuracy = evaluate_model(model, data_loader_test, device="cuda")
#     print("calib method: mse")
#     print("eval accuracy: {:.3f}".format(eval_accuracy))

with torch.no_grad():
    compute_amax(model, method="entropy")
    _, eval_accuracy = evaluate_model(model, data_loader_test, device="cuda")
    print("calib method: entropy")
    print("eval accuracy: {:.3f}".format(eval_accuracy))
    exit()

# with torch.no_grad():
#     for method in ["mse", "entropy"]:
#         print(F"{method} calibration")
#         compute_amax(model, method=method)
#         _, eval_accuracy = evaluate_model(model, data_loader_test, device="cuda")
#         print("calib method: {}".format(method))
#         print("eval accuracy: {:.3f}".format(eval_accuracy))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_model_nvidia(model=model, train_loader=data_loader,
                test_loader=data_loader_test, device="cuda", learning_rate=0.001, num_epochs=15)

# Save the model
torch.save(model.state_dict(), "./nvidia_models/quant_efficientnetb0-finetuned.pth")
