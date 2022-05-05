from itertools import chain, combinations
from scipy.special import comb

def shapley(utility, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in range(1,N+1):
        shapley_dict[i] = 0
    for key in utility:
        if key != ():
            for contributor in key:
                # print('contributor:', contributor, key) # print check
                marginal_contribution = utility[key] - utility[tuple(i for i in key if i!=contributor)]
                # print('marginal:', marginal_contribution) # print check
                shapley_dict[contributor] += marginal_contribution /((comb(N-1,len(key)-1))*N)
    return shapley_dict


def powersettool(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Dictionary Number   : {args.traindivision}\n')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    # print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return