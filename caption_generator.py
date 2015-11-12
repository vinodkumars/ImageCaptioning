import os
import json
import time
from theano import config
import numpy
import scipy.io
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import cPickle as pickle
from utils import initw, _lstm_slice, numpy_floatX
import theano
import theano.scan_module as scan_module
import theano.tensor as tensor
from cost_function import negative_log_likelihood
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

class caption_generator:
  
  def init_params(self, model_options):
    self.image_size = 4096  # size of CNN vectors hardcoded here
    self.word_img_embed_hidden_dim = model_options['word_img_embed_hidden_dim']
    self.vocab_size = model_options['vocab_size']
    
    # Dict name (string) -> numpy ndarray
    params = OrderedDict()
    # Ws -> word encoder
    params['Ws'] = initw(self.vocab_size, self.word_img_embed_hidden_dim).astype(config.floatX)
    # We, be -> image encoder
    params['We'] = initw(self.image_size, self.word_img_embed_hidden_dim).astype(config.floatX)
    params['be'] = initw(self.word_img_embed_hidden_dim).astype(config.floatX)    
    self._lstm_init_params(params)
   
    return params
   
  def _lstm_init_params(self, params):
    input_size = hidden_size = output_size = self.word_img_embed_hidden_dim
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal
    params['WLSTM'] = initw(input_size + hidden_size, 4 * hidden_size).astype(config.floatX)
    params['bLSTM'] = np.zeros(4 * hidden_size).astype(config.floatX)
    # Decoder weights (e.g. mapping to vocabulary)
    params['Wd'] = initw(hidden_size, self.vocab_size).astype(config.floatX)
    params['bd'] = np.zeros(self.vocab_size).astype(config.floatX)
    
  def build_model(self, tparams):
    # sents -> word_indices * #batch_size
    sents = tensor.matrix('sents', dtype="int64")
    # mask -> n_word_indices * #batch_size
    mask = tensor.matrix('mask', dtype=config.floatX)
    # imgs -> #4098 * #batch_size
    imgs = tensor.matrix('imgs', dtype=config.floatX)
    # gt_sents -> word_indices * #batch_size
    gt_sents = tensor.matrix('gt_sents', dtype="int64")
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(1.))
        
    with open("testTagData.pkl", "rb") as f:
      sents_tag, mask_tag, imgs_tag, gt_sents_tag = pickle.load(f)
    sents.tag.test_value = sents_tag
    mask.tag.test_value = mask_tag
    imgs.tag.test_value = imgs_tag
    gt_sents.tag.test_value = gt_sents_tag
    
    n_timesteps = sents.shape[0]
    n_samples = sents.shape[1]
    
    # Image encoding
    # Xe -> #batch_size * #image_encoding_size
    x_e = (tensor.dot(imgs.T, tparams['We']) + tparams['be'])
    # sentences (i.e. captions) encoding
    # Xs -> #no_of_words * #batch_size * #word_encoding_size
    x_s = tparams['Ws'][sents.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                self.word_img_embed_hidden_dim])
    
    # Xes has the image vector as the first timestep
    # Xes -> #no_timesteps (no_of_words + 1 (for image)) * #batch_size * #word_image_encoding_size
    x_es = tensor.zeros([n_timesteps + 1, n_samples, self.word_img_embed_hidden_dim], dtype=config.floatX)
    x_es = tensor.set_subtensor(x_es[1:], x_s)
    x_es = tensor.set_subtensor(x_es[0], x_e)
    
    mask_es = tensor.ones([mask.shape[0] + 1, mask.shape[1]], dtype=config.floatX)
    mask_es = tensor.set_subtensor(mask_es[1:], mask)
    
    # pred_softmax -> #batch_size * #no_of_words * #vocab_size
    pred_softmax = self._lstm_build_model(tparams, x_es, mask_es, use_noise)
    cost = negative_log_likelihood(pred_softmax, gt_sents)
    # pred_prob = lstm_output.max(axis=2)
    # pred = lstm_output.argmax(axis=2)
    
    return sents, mask, imgs, gt_sents, use_noise, cost
    
  def _lstm_build_model(self, tparams, state_below, mask, use_noise):
    hidden_size = self.word_img_embed_hidden_dim
    trng = RandomStreams(SEED)
    
    # if we are dealing with a mini-batch
    nsteps1 = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        init_state = tensor.alloc(0., n_samples, hidden_size)
        init_memory = tensor.alloc(0., n_samples, hidden_size)
    # during sampling
    else:
        n_samples = 1
        init_state = tensor.alloc(0., hidden_size)
        init_memory = tensor.alloc(0., hidden_size)
   
    def _step(m_, x_, h_, c_):
      x_and_h = tensor.concatenate([x_, h_], axis=1)
      preact = tensor.dot(x_and_h, tparams["WLSTM"]) + tparams["bLSTM"]

      i = tensor.nnet.sigmoid(_lstm_slice(preact, 0, hidden_size))
      f = tensor.nnet.sigmoid(_lstm_slice(preact, 1, hidden_size))
      o = tensor.nnet.sigmoid(_lstm_slice(preact, 2, hidden_size))
      c = tensor.tanh(_lstm_slice(preact, 3, hidden_size))

      c = f * c_ + i * c
      c = m_[:, None] * c + (1. - m_)[:, None] * c_

      h = o * tensor.tanh(c)
      h = m_[:, None] * h + (1. - m_)[:, None] * h_

      return h, c
    
    rval1, updates1 = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[init_state, init_memory],
                                name='lstm_layer_train',
                                n_steps=nsteps1)
    
    proj = self.dropout_layer(rval1[0], use_noise, trng)
    
    decoder = (tensor.dot(proj[1:, :, :], tparams['Wd']) + tparams['bd']).dimshuffle(1, 0, 2)

    def _softmax(x_):
      tmp1 = tensor.nnet.softmax(x_)
      return tmp1
    
    nsteps2 = decoder.shape[0]
    rval2, updates2 = theano.scan(_softmax,
                                  sequences=[decoder],
                                  name='lstm_layer_softmax',
                                  n_steps=nsteps2)
    
    return rval2

  def dropout_layer(self, state_before, use_noise, trng):
    """
    tensor switch is like an if statement that checks the
    value of the theano shared variable (use_noise), before
    either dropping out the state_before tensor or
    computing the appropriate activation. During training/testing
    use_noise is toggled on and off.
    """
    proj = tensor.switch(use_noise,
                         (state_before * 
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

  def predict(self, tparams):
    # imgs -> #4098 * #batch_size
    imgs_to_predict = tensor.matrix('imgs_to_predict', dtype=config.floatX)
    
    with open("testTagData.pkl", "rb") as f:
      sents_tag, mask_tag, imgs_tag, gt_sents_tag = pickle.load(f)
    imgs_to_predict.tag.test_value = imgs_tag
    
    # Image encoding
    # Xe -> #batch_size * #image_encoding_size
    x_e_predict = (tensor.dot(imgs_to_predict.T, tparams['We']) + tparams['be'])
    # predicted_indices, predicted_prob -> max_word_count * #batch_size
    predicted_indices, predicted_prob = self._lstm_predict(tparams, x_e_predict)
      
    return imgs_to_predict, predicted_indices, predicted_prob
    
  def _lstm_predict(self, tparams, imgs):
    hidden_size = self.word_img_embed_hidden_dim
    n_samples = imgs.shape[0]
    init_state = tensor.alloc(0., n_samples, hidden_size)
    init_memory = tensor.alloc(0., n_samples, hidden_size)
    is_complete = tensor.alloc(0, n_samples)
        
    def _step(x_, h_, c_, is_complete_, n_samples):
      x_and_h = tensor.concatenate([x_, h_], axis=1)
      preact = tensor.dot(x_and_h, tparams["WLSTM"]) + tparams["bLSTM"]

      i = tensor.nnet.sigmoid(_lstm_slice(preact, 0, hidden_size))
      f = tensor.nnet.sigmoid(_lstm_slice(preact, 1, hidden_size))
      o = tensor.nnet.sigmoid(_lstm_slice(preact, 2, hidden_size))
      c = tensor.tanh(_lstm_slice(preact, 3, hidden_size))

      c = f * c_ + i * c
      h = o * tensor.tanh(c)
      
      decoder = tensor.dot(h, tparams['Wd']) + tparams['bd']
      softmax = tensor.nnet.softmax(decoder)
      predicted_prob, predicted_idx = tensor.max_and_argmax(softmax, axis=1)
      predicted_word_vector = tparams['Ws'][predicted_idx]
      
      is_end_reached = predicted_idx <= 0
      is_complete_ = is_complete_ + is_end_reached
      is_complete_sum = tensor.sum(is_complete_)
      
      return (predicted_word_vector, h, c, is_complete_, predicted_idx, predicted_prob), scan_module.until(tensor.eq(is_complete_sum, n_samples))
    
    rval, updates = theano.scan(_step,
                                outputs_info=[imgs, init_state, init_memory, is_complete, None, None],
                                non_sequences=n_samples,
                                n_steps=50,
                                name='lstm_layer_predict')
    return rval[4], rval[5]
    
