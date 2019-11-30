import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
#torch.manual_seed(1)
import random
import tqdm
from sklearn.utils import resample
import copy
import pickle
import generate_data

'''You could load data this way:
* X,y = generate_data.return_dataset('dataset_name') - choose dataset name from the list
    * protein
    * credit
    * wine
    * airbnb
    * synthetic (you can set n_features,n_samples - number of features and samples in synthetic dataset
'''


class GeneticOpt:
    """
    Class which make genetic optimization of neural net.

    Parameters
    ----------
    val : tuple, (X_val,Y_val)
       validation data

    train : tuple, (X_train,Y_train)
       training data

    iterations : integer, default 50
        Number of iterations for genetic algorithm

    generation_size : integer, default 100
        Number of samples in obe population

    num_epoch : integer, default 100
        Number epochs for neural network training

    batch_size : integer, default 300
        Size of batch
    """
    def __init__(self, val, train, iterations=50, generation_size=100, num_epoch=100, batch_size=300):
        self.x_val = torch.from_numpy(val[0]).float()
        self.y_val = torch.from_numpy(val[1].reshape(val[0].shape[0], 1)).float()

        self.x = torch.from_numpy(train[0]).float()
        self.y = torch.from_numpy(train[1].reshape(train[0].shape[0], 1)).float()
        self.iterations = iterations
        self.generation_size = generation_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def generation_prob(self, generation, using_net, num_layer, deep=None):

        """
        Calculate probability to become a "parent" for each sample in generation.
        Each sample of generation is a binary vector - we call it "mask".
        This vector makes some neurons in concrete layer equals to zero.
        The less error function of concrete model the more probability.

        Parameters
        ----------
        generation : numpy.ndarray
            Array of samples. Each sample is a model with different parameters equals to zero.

        using_net : torch.nn.modules.container.Sequential
            Network which structure we try to optimize.

        num_layer : integer
            Number of layer in "using_net" we are currently optimizing.

        deep : integer
            Firstly we optimize AE. Cut the last layer. Stack normal network to the end. And freeze layers of AE.
            So the parameter "deep" is not None, when we are optimizing second network.

        Returns
        -------
        survivals : numpy.ndarray
            Array of probabilities of masks to become a "parent"
        errors : numpy.ndarray
            Array of errors of model with concrete mask
        """
        s = 0
        survivals = np.zeros(len(generation))
        errors = np.zeros(len(generation))
        for num, x in enumerate(generation):
            net_curr = pickle.loads(pickle.dumps(using_net))

            for q in np.where(x)[0]:
                if deep is None:
                    net_curr[num_layer].weight[q] = 0
                else:
                    net_curr[num_layer][deep].weight[q] = 0
            curr_error = torch.dist(net_curr(self.x_val), self.y_val)
            survivals[num] = curr_error
            errors[num] = curr_error
            s += 1 / curr_error.item()
        survivals = 1 / survivals
        survivals = survivals / s
        return survivals, errors

    def create_new_generation(self, generation, using_net, num_layer, deep=None):

        """
        From samples with highest probability this function create new sample - "children"

        Parameters
        ----------
        generation : numpy.ndarray
            Array of samples. Each sample is a model with different parameters equals to zero.

        using_net : torch.nn.modules.container.Sequential
            Network which structure we try to optimize.

        num_layer : integer
            Number of layer in "using_net" we are currently optimizing.

        deep : integer
            Firstly we optimize AE. Cut the last layer. Stack normal network to the end. And freeze layers of AE.
            So the parameter "deep" is not None, when we are optimizing second network.

        Returns
        -------
        new_generation : numpy.ndarray
            New masks, generated from parents.
        """
        new_generation = []
        gen_probabilities, _ = self.generation_prob(generation, using_net, num_layer, deep)
        for i in range(self.generation_size // 2):
            mother = random.choices(generation, weights=gen_probabilities)[0]
            father = random.choices(generation, weights=gen_probabilities)[0]
            number_child = random.randint(1, len(mother) - 1)
            mother_changed = np.hstack((mother[:number_child], father[number_child:]))
            father_changed = np.hstack((father[:number_child], mother[number_child:]))
            for i in range(random.randint(0, len(mother))):
                num = random.randint(0, len(mother) - 1)
                mother_changed[num] = 1 - mother_changed[num]
                father_changed[num] = 1 - father_changed[num]
            new_generation.append(mother_changed)
            new_generation.append(father_changed)
        new_generation = np.vstack(new_generation)
        return new_generation

    def optimize_net(self, using_net):

        """
        From samples with highest probability this function create new sample - "children"

        Parameters
        ----------
        using_net : torch.nn.modules.container.Sequential
            Network which structure we try to optimize.

        Returns
        -------
        using_net : torch.nn.modules.container.Sequential
            Optimized network, with some neurons equal to zero.
        torch.dist(using_net(self.x_val), self.y_val) : float
            Error of optimized net on validation data
        all_masks : numpy.ndarray
            Binary mask for each layer in optimized net.
        """

        all_masks = []
        for num_layer, layer in enumerate(using_net):
            if using_net[num_layer]._get_name() == 'ReLU':
                continue
            elif using_net[num_layer]._get_name() != 'ReLU' and num_layer != len(using_net) - 1:
                first_gen = np.random.randint(2, size=(self.generation_size, using_net[num_layer].out_features))
                for iterate in tqdm.trange(self.iterations):
                    if iterate == 0:
                        curr_generation = self.create_new_generation(first_gen, using_net, num_layer)
                    else:
                        curr_generation = self.create_new_generation(curr_generation, using_net, num_layer)
                _, errors = self.generation_prob(curr_generation, using_net, num_layer)
                best_mask = curr_generation[np.argmin(errors)]
                for q in np.where(best_mask)[0]:
                    using_net[num_layer].weight[q] = 0
                print('optimize ' + str(num_layer) + ' layer')
                all_masks.append(best_mask)
            elif num_layer == len(using_net) - 1:
                deep_using_net = copy.deepcopy(using_net[-1])
                all_masks_deep = []
                for num_layer_deep, layer_deep in enumerate(deep_using_net):
                    if deep_using_net[num_layer_deep]._get_name() == 'ReLU':
                        continue
                    elif num_layer_deep == len(deep_using_net) - 1:
                        continue
                    else:
                        first_gen = np.random.randint(2, size=(self.generation_size, \
                                                               using_net[num_layer][num_layer_deep].out_features))
                        for iterate in tqdm.trange(self.iterations):
                            if iterate == 0:
                                curr_generation = self.create_new_generation(first_gen, using_net, num_layer,
                                                                             num_layer_deep)
                            else:
                                curr_generation = self.create_new_generation(curr_generation, using_net, num_layer,
                                                                             num_layer_deep)
                        _, errors = self.generation_prob(curr_generation, using_net, num_layer, num_layer_deep)
                        best_mask = curr_generation[np.argmin(errors)]
                        for q in np.where(best_mask)[0]:
                            using_net[num_layer][num_layer_deep].weight[q] = 0
                        print('optimize ' + str(num_layer_deep) + ' layer')
                        all_masks_deep.append(best_mask)
                all_masks.append(all_masks_deep)
        return using_net, torch.dist(using_net(self.x_val), self.y_val), all_masks

    def get_ae_model(self, neurons, input_shape):

        """

        Create structure of autoencoder.

        Parameters
        ----------
        neurons : list
            List of neurons in each layer. We should't set the number of layers, because it is obvious from length of neurons list.
        input_shape : integer
            Input shape of first layer. Equals to shape of samples.
        Returns
        -------
        net : torch.nn.modules.container.Sequential
            Structure of autoencoder.
        """

        net = torch.nn.Sequential()
        for name, q in enumerate(neurons):
            prev = neurons[name - 1]
            curr = neurons[name]
            if name == 0:
                prev = input_shape
            layer = torch.nn.Linear(prev, curr)
            torch.nn.init.xavier_normal_(layer.weight)
            net.add_module('hidden_' + str(name), layer)
            net.add_module('hidden_act_' + str(name), torch.nn.ReLU())
        return net

    def get_dense_model_second(self, neurons, autoencoder):

        """

        Create structure of full network. Full network is a composition of autoencoder and dense network, which stacked on the top of
        autoencoder.

        Parameters
        ----------
        neurons : list
            List of neurons in each layer of dense network. We should't set the number of layers, because it is obvious from length of neurons list.
        autoencoder : torch.nn.modules.container.Sequential
            Autoencoder model. We stack dense model on the top of it.
        Returns
        -------
        curr_autoencoder : torch.nn.modules.container.Sequential
            Full model

        """

        curr_autoencoder = copy.deepcopy(autoencoder)
        for param in curr_autoencoder.parameters():
            param.requires_grad = False

        input_shape = curr_autoencoder[-2].in_features
        curr_autoencoder = curr_autoencoder[:-1]

        net = torch.nn.Sequential()
        for name, q in enumerate(neurons):
            prev = neurons[name - 1]
            curr = neurons[name]
            if name == 0:
                prev = input_shape

            layer = torch.nn.Linear(prev, curr)
            torch.nn.init.xavier_normal_(layer.weight)
            net.add_module('hidden_grad' + str(name), layer)
            if name == len(neurons) - 1:
                continue
            else:
                net.add_module('hidden_act_grad' + str(name), torch.nn.ReLU())
        curr_autoencoder[-1] = net

        return curr_autoencoder

    def fit_net(self, net, batch_size, num_epochs, X, Y):


        """
        Fit the model.

        Parameters
        ----------
        net : torch.nn.modules.container.Sequential
            Model to fit.
        batch_size :
            See parameter in ``batch_size`` in :class:`GeneticOpt`:func:`__init__`
        num_epochs :
            See parameter in ``num_epochs`` in :class:`GeneticOpt`:func:`__init__`
        X :
            See parameter in ``X_train`` in :class:`GeneticOpt`:func:`__init__`
        Y :
            See parameter in ``Y_train`` in :class:`GeneticOpt`:func:`__init__`

        Returns
        -------
        net : torch.nn.modules.container.Sequential
            Fitted net
        """
        filter(lambda p: p.requires_grad, net.parameters())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, eps=1e-08,
                                     weight_decay=0.0)

        loss_func = torch.nn.MSELoss()
        torch_dataset = Data.TensorDataset(X, Y)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=2)
        for epoch in tqdm.tqdm_notebook(range(num_epochs)):
            for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
                # running_loss = 0
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                prediction = net(b_x)  # input x and predict based on x
                loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
            # running_loss += loss.item()
            # training_losses.append(running_loss)

            # with torch.no_grad():
            # net.eval()
            # test_loss =0
            # preds = net(X_val)
            # loss = loss_func(preds,Y_val)

        return net

    def genetic_optimizer(self,neurons_ae, neurons_dense):
        """
            Run genetic optimizer for the model.

            Parameters
            ----------
            neurons_ae : list
                List of neurons for hidden layers of autoencoder. Like [10,10,10,30]. Last element should be equal to self.x.shape[1].
            neurons_dense : list
                List of neurons for hidden layers of autoencoder. Like [10,10,10,1]. Last element should be equal to 1 (if regression).

            Returns
            -------
            init_net : torch.nn.modules.container.Sequential
                Fitted net, but without structure optimization.
            optimized_net : torch.nn.modules.container.Sequential
                Fitted and optimized net.
            first_error : float
                Error of trained net but without optimization.
            error : float
                Error of optimized net.
            masks : numpy.ndarray
                Binary mask for each layer of optimized net.
        """

        loss_func = torch.nn.MSELoss()
        fitted_ae = self.fit_net(self.get_ae_model(neurons_ae, self.x.shape[1]),
                                 self.batch_size, self.num_epoch, self.x, self.x)

        fitted_dense = self.fit_net(
            self.get_dense_model_second(neurons_dense,
                                        fitted_ae), self.batch_size, self.num_epoch, self.x, self.y)

        first_error = loss_func(fitted_dense(self.x_val), self.y_val)
        init_net = pickle.loads(pickle.dumps(fitted_dense))

        optimized_net, error, masks = self.optimize_net(fitted_dense)
        return init_net, optimized_net, first_error, error, masks






