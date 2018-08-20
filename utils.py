import numpy
from collections import OrderedDict
from collections import deque

numpy.random.seed(1)

class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count, sample, cache = 4096):
        self.input_file_name = file_name
        self.sample = sample
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()
        self.cache = cache
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name, encoding = 'utf-8')
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        
        word_frequency = dict(OrderedDict(sorted(word_frequency.items(), key=lambda t: t[1], reverse=True)))
        
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        self.total_words = 0
        
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            self.total_words += c
            wid += 1
        self.word_count = len(self.word2id)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e7
        pow_frequency = numpy.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)
    
    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < self.cache:
            
            sentence = self.input_file.readline()
            
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name, encoding = 'utf-8')
                sentence = self.input_file.readline()
            
            ran_window_size = numpy.random.randint(0, window_size) + 1
            
            
            word_ids = [self.word2id[word] for word in sentence.strip().split(' ') \
                        if word in self.word2id \
                        if (numpy.sqrt(self.word_frequency[self.word2id[word]] / \
                        numpy.float32(self.sample * self.total_words + 1))) * \
                        (self.sample * self.total_words) / self.word_frequency[self.word2id[word]] \
                        > numpy.random.random()]
                        
            self.word_pair_catch += [(u, v) for i, u in enumerate(word_ids) \
                                    for j, v in enumerate(word_ids[max(i - ran_window_size, 0):i + ran_window_size]) \
                                    if i != j]
            
            '''
            word_ids = []
            for word in sentence.strip().split(' '):
                if word in self.word2id:
                    freq = self.word_frequency[self.word2id[word]]
                    ran = (numpy.sqrt(freq / numpy.float32(self.sample * self.total_words + 1))) * (self.sample * self.total_words) / freq
                    if ran < numpy.random.random(): continue
                    word_ids.append(self.word2id[word])
                    
            for i, u in enumerate(word_ids):
                for j, v in enumerate(word_ids[max(i - ran_window_size, 0):i + ran_window_size]):
                    #assert u < self.word_count
                    #assert v < self.word_count
                    if i == j: continue
                    self.word_pair_catch.append((u, v))
            '''
        batch_pairs = [self.word_pair_catch.popleft() for _ in range(batch_size)]
        
        return batch_pairs
        
    # @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size
    