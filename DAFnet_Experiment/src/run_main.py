import numpy as np
from AscentMNIST_data import Ascent_MNISTdataset
import os
import VAEcore
from torch import optim
import plot_utils
import glob
import torch
from MNIST_model import LeNet5
import torch.nn.functional as F

"""checking arguments"""
def check_args(args):
    ### --results_path
    try:
        os.mkdir(args['results_path'])
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args['results_path'] + '/*')
    for f in files:
        os.remove(f)
    ### --add_noise
    try:
        assert args['add_noise'] == True or args['add_noise'] == False
    except:
        print('add_noise must be boolean type')
        return None
    ### --dim-z
    try:
        assert args['dim_z'] > 0
    except:
        print('dim_z must be positive integer')
        return None
    ### --n_hidden
    try:
        assert args['n_hidden'] >= 1
    except:
        print('number of hidden units must be larger than one')
    ### --learn_rate
    try:
        assert args['learn_rate'] > 0
    except:
        print('learning rate must be positive')
    ### --num_epochs
    try:
        assert args['num_epochs'] >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    ### --batch_size
    try:
        assert args['batch_size'] >= 1
    except:
        print('batch size must be larger than or equal to one')
    ### --PRR
    try:
        assert args['PRR'] == True or args['PRR'] == False
    except:
        print('PRR must be boolean type')
        return None

    if args['PRR'] == True:
        ### --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args['PRR_n_img_x'] >= 1 and args['PRR_n_img_y'] >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')
        ### --PRR_resize_factor
        try:
            assert args['PRR_resize_factor'] > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')
    ### --PMLR
    try:
        assert args['PMLR'] == True or args['PMLR'] == False
    except:
        print('PMLR must be boolean type')
        return None

    if args['PMLR'] == True:
        try:
            assert args['dim_z'] == 2
        except:
            print('PMLR : dim_z must be two')
        ### --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args['PMLR_n_img_x'] >= 1 and args['PMLR_n_img_y'] >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')
        ### --PMLR_resize_factor
        try:
            assert args['PMLR_resize_factor'] > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')
        ### --PMLR_z_range
        try:
            assert args['PMLR_z_range'] > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')
        ### --PMLR_n_samples
        try:
            assert args['PMLR_n_samples'] > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args


# load pretrained params
def load_checkpoints(model, path, device):
    """
    Load a saved model.
    """
    checkpoint = torch.load(path, map_location=device)
    model_state = checkpoint.get('model_state', None)
    model.load_state_dict(model_state)


"""train function"""


def train_epoch(networks, optimizer, train_loader, batch_size):
    """
    Update generator and decoder networks for a single epoch.
    """
    classifier, AutoEncoder = networks
    list_bs, list_loss, list_corr = [], [], []
    AutoEncoder.train()
    classifier.eval()

    for _, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        data = data.view(-1, 32*32)
        y, KL_loss = AutoEncoder(data)
        KL_loss = KL_loss * 0
        y = y.view(-1, 1, 32, 32)
        y = (y - 0.1307) / 0.3081
        print(y.size())
        output = classifier(y)
        cls_loss = F.cross_entropy(output, target)
        # loss = cls_loss + KL_loss
        loss = cls_loss

        loss.backward()
        optimizer.step()

        corrects = torch.sum(output.argmax(dim=1).eq(target))
        list_bs.append(batch_size)
        list_loss.append((cls_loss.item(), KL_loss.item()))
        list_corr.append(corrects.item())

    loss = np.average(list_loss, axis=0, weights=list_bs)
    accuracy = np.sum(list_corr) / np.sum(list_bs)
    return accuracy, loss


"""main function"""


def main(args):
    """ parameters """
    RESULTS_DIR = args['results_path']

    # network architecture
    ADD_NOISE = args['add_noise']
    n_hidden = args['n_hidden']
    dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image
    dim_z = args['dim_z']

    # train
    n_epochs = args['num_epochs']
    batch_size = args['batch_size']
    learn_rate = args['learn_rate']

    # Plot
    PMLR = args['PMLR']  # Plot Manifold Learning Result
    PMLR_n_img_x = args['PMLR_n_img_x']  # number of images along x-axis in a canvas
    PMLR_n_img_y = args['PMLR_n_img_y']  # number of images along y-axis in a canvas
    PMLR_resize_factor = args['PMLR_resize_factor']  # resize factor for each image in a canvas
    PMLR_z_range = args['PMLR_z_range']  # range for random latent vector
    DEVICE = args['DEVICE'] # set device
    p_zeroed = args['p_zeroed']

    """ prepare Ascent_MNIST data """

    dataset = Ascent_MNISTdataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0)

    # Teacher classifier
    model_path = 'C://Users//KIMHAKBIN//Documents//PycharmProjects//Activation_Maximization//pretrained//mnist.pth.tar'
    classifier = LeNet5()
    load_checkpoints(classifier, model_path, DEVICE)

    # encoder
    AutoEncoder = VAEcore.autoencoder(dim_img, dim_z, n_hidden, p_zeroed)

    # networks
    networks = (classifier, AutoEncoder)

    # optimizer
    optimizer = optim.Adam(AutoEncoder.parameters(), learn_rate)

    # Plot for manifold learning result
    if PMLR and dim_z == 2:
        PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, IMAGE_SIZE_MNIST,
                                                        IMAGE_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)

    # train per epoch
    min_trn_loss = 1e99

    for epoch in range(1, n_epochs + 1):
        trn_acc, trn_loss = train_epoch(networks, optimizer, train_loader, batch_size)
        print('Iteration:', epoch, '\t', 'Accuaracy:', trn_acc, '\t', 'Loss:', trn_loss)

        # if minimum loss is updated or final epoch, plot results
        if min_trn_loss > np.sum(trn_loss) or epoch + 1 == n_epochs:
            min_trn_loss = np.sum(trn_loss)

            # Plot for manifold learning result
            if PMLR and dim_z == 2:
                AutoEncoder.eval() # p_zeroed를 0으로 만들어 줘야함
                z = torch.from_numpy(PMLR.z).float()
                y_PMLR = AutoEncoder.Decoder(z)

                y_PMLR_img = y_PMLR.reshape(PMLR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                y_PMLR_img = y_PMLR_img.detach().numpy()
                PMLR.save_images(y_PMLR_img, name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

    print('Training Finish!!')


if __name__ == '__main__':
    IMAGE_SIZE_MNIST = 32
    args = {}
    # 'File path of output images'
    args['results_path'] = 'results'
    # 'Boolean for adding salt & pepper noise to input image'
    args['add_noise'] = False
    # 'Dimension of latent vector'
    args['dim_z'] = 2
    # 'Number of hidden units in MLP'
    args['n_hidden'] = 500
    # 'Learning rate for Adam optimizer'
    args['learn_rate'] = 1e-3
    # 'The number of epochs to run'
    args['num_epochs'] = 10000
    # 'Batch size'
    args['batch_size'] = 10
    # 'Boolean for plot-reproduce-result'
    args['PRR'] = True
    # 'Number of images along x-axis'
    args['PRR_n_img_x'] = 10
    # 'Number of images along y-axis'
    args['PRR_n_img_y'] = 10
    # 'Resize factor for each displayed image'
    args['PRR_resize_factor'] = 1.0
    # 'Boolean for plot-manifold-learning-result'
    args['PMLR'] = True
    # 'Number of images along x-axis'
    args['PMLR_n_img_x'] = 10
    # 'Number of images along y-axis'
    args['PMLR_n_img_y'] = 10
    # 'Resize factor for each displayed image'
    args['PMLR_resize_factor'] = 1.0
    # 'Range for uniformly distributed latent vector'
    args['PMLR_z_range'] = 2.0
    # 'Number of samples in order to get distribution of labeled data'
    args['PMLR_n_samples'] = 5000
    # 'probability of an element to be zeroed' - dropout
    args['p_zeroed'] = 0.1
    # 'Set Device'
    args['DEVICE'] = torch.device("cpu")

    check_args(args)

    # main
    main(args)
