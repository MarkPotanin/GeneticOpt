from GeneticOpt import *
import generate_data

datasets={i:{'disp':None,'disp_init':None,'masks':None} for i in ['airbnb','protein','credit','wine','synthetic']}

for dataset in datasets.keys():
    print('PROCESS DATASET : ' + dataset)
    X, y = generate_data.return_dataset(dataset)
    X_train, Y_train = X[:int(len(X) * 0.6)], y[:int(len(X) * 0.6)]
    X_val, Y_val = X[int(len(X) * 0.6):int(len(X) * 0.8)], y[int(len(X) * 0.6):int(len(X) * 0.8)]
    X_test, Y_test = X[int(len(X) * 0.8):], y[int(len(X) * 0.8):]
    gen_opt = GeneticOpt((X_val, Y_val), (X_train, Y_train), batch_size=100)
    initial_net, trained_net, first_error, eroor, masks = gen_opt.genetic_optimizer(
        neurons_ae=[X_train.shape[1] // 2, X_train.shape[1]], \
        neurons_dense=[X_train.shape[1] // 2, X_train.shape[1] // 2, X_train.shape[1] // 2, 1])
    indexes = np.arange(len(Y_test))
    disp = []
    disp_init = []
    for t in range(20):
        bootstraped_index = resample(indexes)
        X_test_curr = X_test[bootstraped_index]
        y_test_curr = Y_test[bootstraped_index]
        x_test = torch.from_numpy(X_test_curr).float()
        y_test = torch.from_numpy(y_test_curr.reshape(X_test_curr.shape[0], 1)).float()
        loss_func = torch.nn.L1Loss()
        disp.append(loss_func(trained_net(x_test), y_test).item())
        disp_init.append(loss_func(initial_net(x_test), y_test).item())

    datasets[dataset]['disp'] = disp
    datasets[dataset]['disp_init'] = disp_init
    datasets[dataset]['masks'] = masks

for i in datasets:
    print(i)
    print('Error after reducing '+ str(round(np.mean(datasets[i]['disp']), 7)) + '+/- ' + str(round(np.std(datasets[i]['disp']), 7)) + \
          ' ||| initial error ' + str(round(np.mean(datasets[i]['disp_init']), 7)) \
          + '+/- ' + str(round(np.std(datasets[i]['disp_init']), 7)))
    print('Number of parameters initial ' + str(len(np.hstack(np.vstack(datasets[i]['masks'])))) + \
          ' , after reducing ' + str(np.sum(np.hstack(np.vstack(datasets[i]['masks'])))))
    print('\n')
