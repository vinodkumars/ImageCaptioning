from random import uniform
from collections import OrderedDict
import theano
from theano import config
import numpy as np
import unicodedata
import numpy


def initw(n, d=0):  # initialize matrix of this size
  magic_number = 0.1
  if d == 0:
    return (np.random.rand(n) * 2 - 1) * magic_number  # U[-0.1, 0.1]
  return (np.random.rand(n, d) * 2 - 1) * magic_number  # U[-0.1, 0.1]

def init_tparams(params):
  tparams = OrderedDict()
  for kk, pp in params.iteritems():
      tparams[kk] = theano.shared(params[kk], name=kk)
  return tparams

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

# use the slice to calculate all the different gates in lstm
def _lstm_slice(_x, n, dim):
  if _x.ndim == 3:
    return _x[:, :, n * dim:(n + 1) * dim]
  elif _x.ndim == 2:
    return _x[:, n * dim:(n + 1) * dim]
  return _x[n * dim:(n + 1) * dim]

def convert_idx_to_sentences(ix_to_word, idx_to_convert):
  word_list = []
  for i in idx_to_convert:
    if i == 0:
      break
    word_list.append(ix_to_word[i])
  unicode_data = " ".join(word_list)
  string_data = unicode_to_string(unicode_data)
  return string_data.lower().strip()

def unicode_to_string(unicode_data):
  if type(unicode_data) == unicode:
    return unicodedata.normalize('NFKD', unicode_data).encode('ascii', 'ignore')
  elif type(unicode_data) == str:
    return unicode_data
  else:
    raise Exception('data type neither string nor unicode')

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def load_params(path, params):
  pp = numpy.load(path)
  for kk, vv in params.iteritems():
    if kk not in pp:
      raise Warning('%s is not in the archive' % kk)
    params[kk] = pp[kk]
  checkpoint_save_n = pp['checkpoint_save_n']
  return params, checkpoint_save_n
  
  
      
    
    
      
