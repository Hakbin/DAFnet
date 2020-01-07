import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def Ascent_MNISTdataset():
    dataset = datasets.ImageFolder(root='C://Users//KIMHAKBIN//Documents//PycharmProjects//Datacloud//AscentMNIST',
                                   transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                   transforms.ToTensor()]))
    return dataset

