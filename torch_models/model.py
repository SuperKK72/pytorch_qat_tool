from mobilenet_v2 import *
wt_path = "./mobilenet_v2_pretrained_float.pth"
scripted_float_model_file = 'mobilenet_v2_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_v2_quantization_scripted_quantized.pth'

#load float model
float_model = load_model(wt_path)
float_model.eval()
dummy_input = torch.rand((1,3,224,224))
torch.onnx.export(float_model, dummy_input, "./mobilenet_v2.onnx", opset_version=11)
# print(model)

#fuse float model
float_model.fuse_model()
torch.onnx.export(float_model, dummy_input, "./mobilenet_v2_fused.onnx", opset_version=11)
print_size_of_model(float_model)
torch.jit.save(torch.jit.script(float_model), scripted_float_model_file)

#quant model
float_model.qconfig = torch.quantization.default_qconfig
print(float_model.qconfig)
torch.quantization.prepare(float_model, inplace=True)
torch.quantization.convert(float_model, inplace=True)
print_size_of_model(float_model)
torch.jit.save(torch.jit.script(float_model), scripted_quantized_model_file)