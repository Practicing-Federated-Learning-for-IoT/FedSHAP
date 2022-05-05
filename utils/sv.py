import time
from tqdm import tqdm
from models.Update import LocalUpdate, test_inference
import copy
from models.Fed import FedAvg
from utils.utils import exp_details, powersettool, shapley


def exact(args, net_glob, powerset, submodel_dict, fraction, train_dataset, test_dataset, dict_users, test_acc):
    net_glob = net_glob.to(args.device)
    accuracy_dict = {}
    start_time = time.time() # start the timer

    # accuracy for the full dataset is the same as global accuracy
    accuracy_dict[powerset[-1]] = test_acc

    # Federated Exact Algorithm
    for subset in powerset[1:-1]:
        # print("current subset: ", subset) # print check
        train_loss = []
        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            # print(f'\n | Global Training Round {subset} : {epoch+1} |\n') # print check

            submodel_dict[subset].train()
            # note that the keys for train_dataset are [1,2,3,4,5]
            for idx in subset:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=dict_users[idx])
                w, loss = local_model.train(net=copy.deepcopy(net_glob))
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = FedAvg(local_weights, [fraction[i-1] for i in subset])

            # update global weights
            submodel_dict[subset].load_state_dict(global_weights)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, submodel_dict[subset], test_dataset)
        accuracy_dict[subset] = test_acc
    # accuracy for the random model
    test_acc, _ = test_inference(args, submodel_dict[()], test_dataset)
    accuracy_dict[()] = test_acc
    shapley_dict = shapley(accuracy_dict, args.num_users)


    totalRunTime = time.time()-start_time
    print('\n Total Run Time: {0:0.4f}'.format(totalRunTime))  # print total time

    #write information into a file
    accuracy_file = open('../save/ExactFederated_{}_{}_{}_{}_{}users.txt'.format(args.dataset, args.model,
                            args.epochs, args.traindivision, args.num_users), 'a')
    for subset in powerset:
        accuracy_lines = ['Trainset: '+args.traindivision+'_'+''.join([str(i) for i in subset]), '\n',
                'Accuracy: ' +str(accuracy_dict[subset]), '\n',
                '\n']
        accuracy_file.writelines(accuracy_lines)
    for key in shapley_dict:
        shapley_lines = ['Data contributor: ' + str(key),'\n',
                'Shapley value: '+ str(shapley_dict[key]), '\n',
                '\n']
        accuracy_file.writelines(shapley_lines)
    lines = ['Total Run Time: {0:0.4f}'.format(totalRunTime),
             '\n']
    accuracy_file.writelines(lines)
    accuracy_file.close()