import json
import os
import copy
import time
from theano import config
import numpy
import scipy.io
from collections import defaultdict
import numpy as np
import cPickle as pickle
from sklearn.cross_validation import KFold
import utils

class data_provider:
   
  def load_data(self, batch_size, word_count_threshold):
    print 'Initializing DataProvider'
    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', 'coco')
    self.batch_size = batch_size
    self.word_count_threshold = word_count_threshold
    
    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    t0 = time.time()
    print 'DataProvider: reading %s' % (dataset_path,)
    self.dataset = json.load(open(dataset_path, 'r'))
    print 'DataProvider: loaded %s in %.2fs' % (dataset_path, time.time() - t0)
    
    # load the image features into memory
    t0 = time.time()
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
    print 'DataProvider: reading %s' % (features_path,)
    features_struct = scipy.io.loadmat(features_path)
    self.features = features_struct['feats']
    print 'DataProvider: loaded %s in %.2fs' % (features_path, time.time() - t0)
    
    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)
  
  def get_image_and_sent_tokens(self, sentid):
    # NOTE: imgid is an integer, and it indexes into features
    imgid = self.get_imgid_from_sentid(sentid)
    sents_for_img = self.dataset['images'][imgid]['sentences']
    for s in sents_for_img:
      if s['sentid'] == sentid:
        sent_tokens = s['tokens']
        break
    return self._get_img_from_imgid(imgid), sent_tokens
  
  def get_raw_sentences_from_imgid(self, imgid):
    sents_for_img = self.dataset['images'][imgid]['sentences']
    raw_sentences_unicode = [s['raw'] for s in sents_for_img]
    raw_sentences_string = [utils.unicode_to_string(s).lower().strip() for s in raw_sentences_unicode]
    return raw_sentences_string
  
  def _get_img_from_imgid(self, imgid):
    return self.features[:, imgid]
  
  def get_imgid_from_sentid(self, sentid):
    return self.sentid_to_imgid[sentid]

  def _get_split_size(self, split, ofwhat='sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else:  # assume images
      return len(self.split[split])
  
  def _iter_sentences(self, split='train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield sent
    
  def build_word_vocab(self):
    # count up all word counts so that we can threshold
    # this shouldnt be too expensive of an operation
    print 'DataProvider: preprocessing word counts and creating vocab based on word count threshold %d' % (self.word_count_threshold,)
    t0 = time.time()
    wordCounts = {}
    nsents = 0
    for sent in self._iter_sentences('train'):
      nsents += 1
      for w in sent['tokens']:
        wordCounts[w] = wordCounts.get(w, 0) + 1
    vocab = [w for w in wordCounts if wordCounts[w] >= self.word_count_threshold]
    print 'DataProvider: filtered words from %d to %d in %.2fs' % (len(wordCounts), len(vocab), time.time() - t0)

    # with K distinct words:
    # - there are K+1 possible inputs (START token and all the words)
    # - there are K+1 possible outputs (END token and all the words)
    # we use ix_to_word to take predicted indices and map them to words for output visualization
    # we use word_to_ix to take raw words and get their index in word vector matrix
    self.ix_to_word = {}
    self.ix_to_word[0] = '.'  # period at the end of the sentence. make first dimension be end token
    self.word_to_ix = {}
    self.word_to_ix['#START#'] = 0  # make first vector be the start token
    ix = 1
    for w in vocab:
      self.word_to_ix[w] = ix
      self.ix_to_word[ix] = w
      ix += 1

    # compute bias vector, which is related to the log probability of the distribution
    # of the labels (words) and how often they occur. We will use this vector to initialize
    # the decoder weights, so that the loss function doesn't show a huge increase in performance
    # very quickly (which is just the network learning this anyway, for the most part). This makes
    # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
    # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
    wordCounts['.'] = nsents
    self.biasInitVector = np.array([1.0 * wordCounts[self.ix_to_word[i]] for i in self.ix_to_word])
    self.biasInitVector /= np.sum(self.biasInitVector)  # normalize to frequencies
    self.biasInitVector = np.log(self.biasInitVector)
    self.biasInitVector -= np.max(self.biasInitVector)  # shift to nice numeric range
    
  def get_word_vocab_size(self):
    return len(self.word_to_ix)
    
  def group_train_captions_by_length(self):
    t0 = time.time()
    print 'DataProvider: grouping captions by length'
    self.sentid_to_imgid = {}
    self.sent_length_to_sentids = {}
    for img in self.split['train']:
      for sent in img['sentences']:
        self.sentid_to_imgid[sent['sentid']] = img['imgid']
        sentLength = len(sent['tokens'])
        if sentLength not in self.sent_length_to_sentids:
          self.sent_length_to_sentids[sentLength] = []
        self.sent_length_to_sentids[sentLength].append(sent['sentid'])
    self.train_iterator = data_iterator(self.sent_length_to_sentids, self.batch_size)
    print 'DataProvider: finished grouping captions by length in %.2fs' % (time.time() - t0)
    
  def prepare_train_batch_data(self, batch_sent_ids):
    # caps -> words (i.e. time steps) x #samples
    # caps_mask -> words (i.e. time steps) x #samples
    # imgs -> img_features_dim (i.e. 4096) x #samples
    
    batch_size = len(batch_sent_ids)
    # we add the START token to the beginning of the captions
    max_sent_len = 1 + max([len(self.get_image_and_sent_tokens(s)[1]) for s in batch_sent_ids])

    sents = numpy.zeros((max_sent_len, batch_size)).astype("int64")
    mask = numpy.zeros((max_sent_len, batch_size)).astype(config.floatX)
    imgs = numpy.zeros((self.features.shape[0], batch_size)).astype(config.floatX)
    ground_truth_sents = numpy.zeros((max_sent_len, batch_size)).astype("int64")
    
    for idx, sentid in enumerate(batch_sent_ids):
      img, sent_tokens = self.get_image_and_sent_tokens(sentid)
      imgs[:, idx] = img
      # we add the START token to the beginning of the captions
      word_idx_vector = [self.word_to_ix[w] for w in sent_tokens if w in self.word_to_ix ]
      sents[1:len(word_idx_vector) + 1, idx] = word_idx_vector
      ground_truth_sents[:len(word_idx_vector), idx] = word_idx_vector
      mask[:len(word_idx_vector) + 1, idx] = 1
    
    return sents, mask, imgs, ground_truth_sents
  
  def prepare_valid_or_test_batch_image_data(self, batch_img_idxs, split='val'):
    batch_size = len(batch_img_idxs)
    imgs = numpy.zeros((self.features.shape[0], batch_size)).astype(config.floatX)
    img_ids = [self.split[split][idx]['imgid'] for idx in batch_img_idxs]
    
    for idx, imgid in enumerate(img_ids):
      imgs[:, idx] = self._get_img_from_imgid(imgid)
      
    return imgs, img_ids

class data_iterator:
  def __init__(self, sent_length_to_sentids, batch_size):
    self.sent_length_to_sentids = sent_length_to_sentids
    self.batch_size = batch_size
    self.reset_iter()
    
  def reset_iter(self):
    self.curr_sent_length_to_sentids = copy.deepcopy(self.sent_length_to_sentids)
    self.curr_sent_length_to_sentids_keys = numpy.random.permutation(self.curr_sent_length_to_sentids.keys()).tolist()
    self.len_idx = -1
    for k in self.curr_sent_length_to_sentids.keys():
      self.curr_sent_length_to_sentids[k] = numpy.random.permutation(self.curr_sent_length_to_sentids[k]).tolist()

  def next(self):
    # randomly choose the length
    count = 0
    curr_key = None
    while True:
      self.len_idx = numpy.mod(self.len_idx + 1, len(self.curr_sent_length_to_sentids_keys))
      curr_key = self.curr_sent_length_to_sentids_keys[self.len_idx]
      if len(self.curr_sent_length_to_sentids[curr_key]) > 0:
        break
      count += 1
      if count >= len(self.curr_sent_length_to_sentids_keys):
        curr_key = None
        break
    if curr_key is None:
      self.reset_iter()
      raise StopIteration()
    
    # get the batch size
    curr_batch_size = numpy.minimum(self.batch_size, len(self.curr_sent_length_to_sentids[curr_key]))
    ret_val = self.curr_sent_length_to_sentids[curr_key][:curr_batch_size]
    del self.curr_sent_length_to_sentids[curr_key][:curr_batch_size]
    return ret_val
  
  def __iter__(self):
    return self
  




# dp = data_provider()
# dp.load_data(256, 5)
# tmp = dp.get_raw_sentences_from_imgid(1)
# kf_valid = KFold(len(dp.split['val']), n_folds=len(dp.split['val']) / 100, shuffle=False)
# for _, img_idx in kf_valid:
#  print 'length: %s' % len(img_idx)
#  batch_imgs = dp.prepare_valid_or_test_batch_image_data(img_idx)
#  print 'hello'
  
# dp.build_word_vocab()
# dp.group_train_captions_by_length()

# for e in range(5):
#  print 'epoch ', e
#  i = 0
#  for d in dp.train_iterator:
#    i = i + 1
#    print 'epoch ', e, 'iter ', i
#    if i == 1652:
#      print 'suspect code'
#    
#    sents, sents_mask, imgs, ground_truth_sents = dp.prepare_train_batch_data(d)
#  if(len(d) <= 20):
#    data = [sents, sents_mask, imgs, ground_truth_sents]
#    with open("testTagData.pkl", "wb") as f:
#      pickle.dump(data, f)
  
