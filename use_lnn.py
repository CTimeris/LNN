import os
import time
import torch
import argparse

from lnn import LNN
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m')
parser.add_argument('--train_dir', default='')
parser.add_argument('--batch_size', default=10240, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--reservoir_dim', default=1000, type=int)
parser.add_argument('--leak_rate', default=0.1, type=float)
parser.add_argument('--spectral_radius', default=0.9, type=float)
parser.add_argument('--num_epochs', default=10, type=int)


args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':

    u2i_index, i2u_index, n_users, n_items = build_index(args.dataset)

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    input_dim, output_dim = args.maxlen, n_items
    model = LNN(args, input_dim, output_dim)
    for step in range(num_batch):
        u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray   seq, pos, neg: batch_size x maxlen
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        print(u.shape, seq.shape, pos.shape, neg.shape)
        print(seq[0], pos[0])
        exit(0)
        model.train(seq, pos)




