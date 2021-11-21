import torch
import argparse
import torchvision
import torch.nn as nn
import attack_generator as attack
from torchvision import transforms
from models import *

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from SmallCNN,resnet18,NIN")
parser.add_argument('--dataset', type=str, default="svhn", help="choose from cifar10,svhn,cifar100")
parser.add_argument('--model_path', default="./bestpoint.pth.tar", help='model for white-box attack evaluation')
parser.add_argument('--method',type=str,default='dat',help='select attack setting following DAT or TRADES')

args = parser.parse_args()
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

num_c = 10
print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
    num_c = 10
if args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='../data/svhn', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
if args.dataset == "cifar100":
    testset = torchvision.datasets.CIFAR100(root='../data/cifar-100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    num_c = 100

print('==> Load Model')
if args.net == "SmallCNN":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18(num_classes=num_c).cuda()
    net = "NIN"
if args.net == "NIN":
    model = NIN().cuda()

print(net)
print(args.model_path)
model.load_state_dict(torch.load(args.model_path)['state_dict'])
print(torch.load(args.model_path)['epoch'])
print('==> Evaluating Performance under White-box Adversarial Attack')

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))
if args.method == "dat":
    # Evalutions the same as DAT.
    if args.dataset != "svhn":
      loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=8/255, step_size=8/255,loss_fn="cent", category="Madry",random=True)
      print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
      loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=8/255, step_size=2/255,loss_fn="cent", category="Madry", random=True)
      print('PGD20 Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
      loss, cw_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=8/255, step_size=2/255,loss_fn="cw",category="Madry",random=True)
      print('CW Test Accuracy: {:.2f}%'.format(100. * cw_wori_acc))
    else:
      loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=4/255, step_size=1/255,loss_fn="cent", category="Madry",random=True)
      print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
      loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=4/255, step_size=1/255,loss_fn="cent", category="Madry", random=True)
      print('PGD20 Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
      loss, cw_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=4/255, step_size=1/255,loss_fn="cw",category="Madry",random=True)
      print('CW Test Accuracy: {:.2f}%'.format(100. * cw_wori_acc))      