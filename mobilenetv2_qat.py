import warnings
warnings.filterwarnings("ignore")
import torch.utils.data
from quantization_utils import *

def main():
    r'''init params'''
    random_seed = 0
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    model_dir = "quantized_models"
    model_path = "./pretrained_models/mobilenet_v2_pretrained_float.pth"
    quantized_model_filename = "mobilenetv2_qat_quantized.pt"
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
    set_random_seeds(random_seed=random_seed)
    print_module_flag = False
    print_child_flag = False

    r'''load pretrained fp32 model and set mode to train'''
    model = MobileNetV2()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(cpu_device)
    model.train()

    r'''display submodules'''
    if print_module_flag:
        display_named_modules(model)
    if print_child_flag:
        display_named_children(model)

    r'''get train_loader and test_loader of imagenet2012 '''
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
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    val_data_dir = "/workspace/kli/ILSVRC2012/valid"
    val_label_file = "./data/val.txt"
    images, labels = get_images_and_labels(val_label_file)
    test_dataset = valDataset(val_data_dir, images, labels,
                              transform=data_transform_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    r'''eval fp32 model: 0.71840'''
    print("eval fp32 model...")
    # _, fp32_eval_accuracy = evaluate_model(
    #     model=model, test_loader=test_loader, device=cuda_device, criterion=None)
    fp32_eval_accuracy = 0.71840
    print("FP32 evaluation accuracy: {:.5f}".format(fp32_eval_accuracy))
    print("---------------------------------------------------------------")

    r'''Move the model to CPU since static quantization does not support CUDA currently.'''
    model.to(cpu_device)

    r'''Make a copy of the model for layer fusion, the model has to be switched to training mode before any layer fusion
    Otherwise the quantization aware training will not work correctly.'''
    print("fuse model...")
    fused_model = copy.deepcopy(model)
    fused_model.train()
    fused_model.fuse_model()
    print("---------------------------------------------------------------")

    r'''Model and fused model should be equivalent.'''
    print("verify fuse equivalence...")
    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device,
                             rtol=1e-03, atol=1e-06,
                             num_tests=100, input_size=(1, 3, 224, 224)), "Fused model is not equivalent to the original model!"
    print("---------------------------------------------------------------")

    r'''Prepare the model for quantization aware training. This inserts observers in the model that will observe 
    activation tensors during calibration. Using un-fused model will fail. Because there is no quantized layer 
    implementation for a single batch normalization layer.'''
    print("prepare qat...")
    quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm")
    fused_model.qconfig = quantization_config
    torch.quantization.prepare_qat(fused_model, inplace=True)
    print("---------------------------------------------------------------")

    r'''qat training'''
    print("Training QAT Model...")
    _, fp32_eval_accuracy = evaluate_model(
        model=fused_model, test_loader=test_loader, device=cuda_device, criterion=None)
    print("FP32 evaluation accuracy: {:.5f}".format(fp32_eval_accuracy))
    train_model(model=fused_model, train_loader=train_loader,
                test_loader=test_loader, device=cuda_device, learning_rate=1e-3, num_epochs=15)
    fused_model.to(cpu_device)
    fused_model.eval()
    print("---------------------------------------------------------------")

    r'''Save qat quantized model.'''
    print("save qat quantized model...")
    quantized_model = torch.quantization.convert(fused_model, inplace=True)
    quantized_model.eval()
    save_torchscript_model(model=quantized_model, model_dir=model_dir,
                           model_filename=quantized_model_filename)
    print("---------------------------------------------------------------")

    r'''verify performance of int8 model'''
    print("verify performance of int8 model...")
    quantized_jit_model = load_torchscript_model(
        model_filepath=quantized_model_filepath, device=cpu_device)
    fp32_eval_accuracy = 0.718
    _, int8_eval_accuracy = evaluate_model(
        model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)
    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(
        model=model, device=cpu_device, input_size=(1, 3, 224, 224), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(
        model=quantized_model, device=cpu_device, input_size=(1, 3, 224, 224), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(
        model=quantized_jit_model, device=cpu_device, input_size=(1, 3, 224, 224), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(
        model=model, device=cuda_device, input_size=(1, 3, 224, 224), num_samples=100)
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))
    print("----------------------------END--------------------------------")

    return 0

if __name__ == "__main__":

    main()
