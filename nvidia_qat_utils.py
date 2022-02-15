import torch
import torch.utils.data
from torch import nn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from tqdm import tqdm
import torch.optim as optim


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            # print(F"{name:40}: {module}")
    model.cuda()


def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    batch_idx = 0
    for inputs, labels in test_loader:
        if batch_idx%100==0:
            print("eval batch {}...".format(batch_idx))
        batch_idx += 1
        inputs = inputs.to(device)
        labels = labels.squeeze(dim=1)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    return eval_loss, eval_accuracy

def train_model_nvidia(model, train_loader, test_loader, device, learning_rate=1e-3, num_epochs=15):

    # The training configurations were not carefully selected.
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    # model.eval()
    # eval_loss, eval_accuracy = evaluate_model(
    #     model=model, test_loader=test_loader, device=device, criterion=criterion)
    # print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(-1, eval_loss, eval_accuracy))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0
        running_corrects = 0
        train_batch_idx = 0
        batch_num = int(train_loader.batch_sampler.sampler.num_samples/train_loader.batch_size)
        # revised by kli: fuse relevant parameters
        # model.apply(torch.quantization.disable_observer)
        # model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        for inputs, labels in train_loader:
            if train_batch_idx%100==0:
                print("Epoch {}/{}\tbatch {}/{}...".format(epoch, num_epochs, train_batch_idx, batch_num))
            train_batch_idx += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)
        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(
            model=model, test_loader=test_loader, device=device, criterion=criterion)
        # Set learning rate scheduler
        lr_scheduler.step()
        print("current lr: {}".format(learning_rate))
        #visualize model loss and accuracy per epoch
        print("Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
            epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model