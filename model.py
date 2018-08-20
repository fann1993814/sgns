import torch
import torch.nn as nn

class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.

    Attributes:
        emb_size: Embedding size.
        emb_dimention: Embedding dimention, typically from 50 to 500.
        u_embedding: Embedding for center word.
        v_embedding: Embedding for neibor words.
    """

    def __init__(self, vocab_size, emb_dimension):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight

        Args:
            vocab_size: Vocabulary size.
            emb_dimention: Embedding dimention, typically from 50 to 500.

        Returns:
            None
        """
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        """Initialize embedding weight like word2vec.

        The u_embedding is a uniform distribution in [-0.5/em_size, 0.5/emb_size], and the elements of v_embedding are zeroes.

        Returns:
            None
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """Forward process.

        As pytorch designed, all variables must be batch format, so all input of this method is a list of word id.

        Args:
            pos_u: list of center word ids for positive word pairs.
            pos_v: list of neibor word ids for positive word pairs.
            neg_v: list of neibor word ids for negative word pairs.

        Returns:
            Loss of this process, a pytorch variable.
        """
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        
        batch_size, dim = emb_u.size()
        
        if batch_size > 1:
            score = torch.mul(emb_u, emb_v).squeeze()
            score = torch.sum(score, dim=1)
        else:
            score = score = torch.dot(emb_u.squeeze(), emb_v.squeeze())
            
        score = torch.log(torch.sigmoid(score))
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.log(torch.sigmoid(-1 * neg_score))
        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def save_embedding(self, id2word, file_name, use_cuda):
        """Save all embeddings to file.

        As this class only record word id, so the map from id to word has to be transfered from outside.

        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))