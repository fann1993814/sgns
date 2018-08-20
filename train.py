import argparse
import sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils import InputData
from model import SkipGramModel

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.manual_seed(1)
torch.set_num_threads(4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help="text file, each line is a sentence splited with space.")
    parser.add_argument('--output', type=str, default='', help="embedding file.")
    parser.add_argument('--dim', type=int, default=100, help="embedding dimension")
    parser.add_argument('--window', type=int, default=5, help="number of context words")
    parser.add_argument('--n_negs', type=int, default=5, help="number of negative samples")
    parser.add_argument('--min_count', type=int, default=5, help="minimal word frequency, words with lower frequency will be filtered.")
    parser.add_argument('--iters', type=int, default=1, help="number of iteration")
    parser.add_argument('--mb', type=int, default=2, help="mini-batch size")
    parser.add_argument('--sample', type=float, default=1e-3, help="subsample threshold")
    parser.add_argument('--lr', type=float, default=0.025, help="learning rate")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    
    return parser.parse_args()

def train(args):
    
    data = InputData(args.input, args.min_count, args.sample)
    output_file_name = args.output
    emb_size = len(data.word2id)
    emb_dimension = args.dim
    batch_size = args.mb
    window_size = args.window
    n_negs = args.n_negs
    iteration = args.iters
    initial_lr = args.lr
    use_cuda = args.cuda
    
    skip_gram_model = SkipGramModel(emb_size, emb_dimension)
    if use_cuda: skip_gram_model = skip_gram_model.cuda()
    
    optimizer = optim.SGD(skip_gram_model.parameters(), lr=initial_lr)
        
    pair_count = data.evaluate_pair_count(window_size)
    batch_count = iteration * pair_count / batch_size
    process_bar = tqdm(range(int(batch_count)))
    
    # skip_gram_model.save_embedding(
    #     data.id2word, 'begin_embedding.txt', use_cuda)

    for i in process_bar:
        pos_pairs = data.get_batch_pairs(batch_size, window_size)
        neg_v = data.get_neg_v_neg_sampling(pos_pairs, n_negs)
        pos_u = [pair[0] for pair in pos_pairs]
        pos_v = [pair[1] for pair in pos_pairs]

        pos_u = torch.LongTensor(pos_u)
        pos_v = torch.LongTensor(pos_v)
        neg_v = torch.LongTensor(neg_v)
        if use_cuda:
            pos_u = pos_u.cuda()
            pos_v = pos_v.cuda()
            neg_v = neg_v.cuda()
        
        optimizer.zero_grad()
        loss = skip_gram_model(pos_u, pos_v, neg_v)
        loss.backward()
        optimizer.step()
        
        process_bar.set_description("\rLoss: %0.8f, lr: %0.6f" %
                                    (loss.item(),
                                     optimizer.param_groups[0]['lr']))
        
        if i * batch_size % 100000 == 0:
            lr = initial_lr * (1.0 - 1.0 * i / batch_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
    skip_gram_model.save_embedding(data.id2word, output_file_name, use_cuda)


if __name__ == '__main__':
    train(parse_args())
