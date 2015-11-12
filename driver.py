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
import theano.tensor as tensor
import argparse
from caption_generator import caption_generator
from data_provider import data_provider
import utils
from optimizers import sgd, adadelta, rmsprop
import metrics
from sklearn.cross_validation import KFold
import theano

def prediction(f_pred, f_pred_prob, prepare_valid_or_test_batch_image_data, split, iterator, ix_to_word, get_raw_sentences_from_imgid, model_options, prediction_save_n):
  # prediction_sentences -> imgid to sentence as a list
  prediction_sentences = {}
  # prediction_gt_sents -> imgid to list of ground truth sentences
  prediction_gt_sents = {}
    # prediction_log_prob -> imgid to log_prob
  prediction_log_prob = {}
  
  for _, valid_index in iterator:
    imgs, img_ids = prepare_valid_or_test_batch_image_data(valid_index, split)
    pred = f_pred(imgs)
    pred_prob = f_pred_prob(imgs)
    
    for idx, img_id in enumerate(img_ids):
      prediction_sentences[img_id] = [utils.convert_idx_to_sentences(ix_to_word, pred[:, idx])]
      # TODO: Need to fix log prop
      prediction_log_prob[img_id] = sum(pred_prob[:, idx])
      prediction_gt_sents[img_id] = get_raw_sentences_from_imgid(img_id)
  
  hypo = {idx:x for idx, x in enumerate(prediction_sentences.values())}  
  ref = {idx:x for idx, x in enumerate(prediction_gt_sents.values())}
  
  if numpy.mod(prediction_save_n, model_options['hypo_save_freq']) == 0:
    save_path = os.path.join(model_options['hypo_save_dir'], 'hypo{0}.pkl'.format(prediction_save_n))
    pickle.dump([hypo, ref], open(save_path, 'wb'), -1)
    print 'Saved hypo to ', os.path.abspath(save_path)
  
  scores = metrics.score(ref, hypo)
  return scores
  
optimizers = {'sgd':sgd, 'adadelta':adadelta, 'rmsprop':rmsprop }


def validate_and_save_checkpoint(model_options, dp, params, tparams, f_pred, f_pred_prob, kf_valid, save_n):
  scores = prediction(f_pred, f_pred_prob, dp.prepare_valid_or_test_batch_image_data, 'val', kf_valid, dp.ix_to_word, dp.get_raw_sentences_from_imgid, model_options, save_n['prediction'])
  # saving a checkpoint
  save_path = os.path.join(model_options['checkpoint_save_dir'], "lstm_{0}_{1:.2f}.npz".format(save_n['checkpoint'], scores['Bleu_4'] * 100))
  params = utils.unzip(tparams)
  numpy.savez(save_path, checkpoint_save_n=save_n['checkpoint'], scores=scores, **params)
  pickle.dump(model_options, open('%s.pkl' % save_path, 'wb'), -1)
  print 'Saved checkpoint to', os.path.abspath(save_path)
  
  save_n['checkpoint'] = save_n['checkpoint'] + 1
  save_n['prediction'] = save_n['prediction'] + 1
  
  return scores

def main(model_options):
  
  print 'Loading data'
  dp = data_provider()
  dp.load_data(model_options['batch_size'], model_options['word_count_threshold'])
  dp.build_word_vocab()
  dp.group_train_captions_by_length()
  model_options['vocab_size'] = dp.get_word_vocab_size()

  print 'Building model'  
  # This create the initial parameters as numpy ndarrays.
  generator = caption_generator()
  params = generator.init_params(model_options)
  save_n = {}
  save_n['checkpoint'] = 0
  save_n['prediction'] = 0
  
  # reload a saved checkpoint
  if model_options['reload_checkpoint_path']:
    _, save_n['checkpoint'] = utils.load_params(model_options['reload_checkpoint_path'], params)
    print 'Reloaded checkpoint from', model_options['reload_checkpoint_path']
  
  # This create Theano Shared Variable from the parameters.
  # Dict name (string) -> Theano Tensor Shared Variable
  # params and tparams have different copy of the weights.
  tparams = utils.init_tparams(params)
  
  # use_noise is for dropout
  sents, mask, imgs, gt_sents, use_noise, cost = generator.build_model(tparams)
  grads = tensor.grad(cost, wrt=tparams.values())
  
  lr = tensor.scalar(name='lr')
  f_grad_shared, f_update = optimizers[model_options['optimizer']](lr, tparams, grads, sents, mask, imgs, gt_sents, cost)
  
  imgs_to_predict, predicted_indices, predicted_prob = generator.predict(tparams)
  f_pred = theano.function([imgs_to_predict], predicted_indices, name='f_pred')
  f_pred_prob = theano.function([imgs_to_predict], predicted_prob, name='f_pred_prob')
    
  train_iter = dp.train_iterator
  kf_valid = KFold(len(dp.split['val']), n_folds=len(dp.split['val']) / model_options['batch_size'], shuffle=False)
  
  if model_options['use_dropout'] == 1:
    use_noise.set_value(1.)
  else:
    use_noise.set_value(0.)
     
  print 'Optimization'
  
  uidx = 0
  lrate = model_options['lrate']
  # display print time duration
  dp_start = time.time()
  for eidx in xrange(model_options['max_epochs']):
    print 'Epoch ', eidx
    
    for batch_data in train_iter:
      uidx += 1
      
      # preparing the mini batch data
      pd_start = time.time()
      sents, sents_mask, imgs, gt_sents = dp.prepare_train_batch_data(batch_data)
      pd_duration = time.time() - pd_start
      
      if sents is None:
        print 'Minibatch is empty'
        continue
      
      # training on the mini batch
      ud_start = time.time()
      cost = f_grad_shared(sents, sents_mask, imgs, gt_sents)
      f_update(lrate)
      ud_duration = time.time() - ud_start
      
      # Numerical stability check
      if numpy.isnan(cost) or numpy.isinf(cost):
        print 'NaN detected'
      
      if numpy.mod(uidx, model_options['disp_freq']) == 0:
        dp_duration = time.time() - dp_start
        print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Prepare data ', pd_duration, 'Update data ', ud_duration, '{0}_iter_time {1}'.format(model_options['disp_freq'], dp_duration)
        dp_start = time.time()

      # Log validation loss + checkpoint the model with the best validation log likelihood
      if numpy.mod(uidx, model_options['valid_freq']) == 0:
        scores = validate_and_save_checkpoint(model_options, dp, params, tparams, f_pred, f_pred_prob, kf_valid, save_n)
        print scores
  
  print 'Performing final validation'
  scores = validate_and_save_checkpoint(model_options, dp, params, tparams, f_pred, f_pred_prob, kf_valid, save_n)
  print scores
  print 'Done!!!'
  

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  # global setup settings, and checkpoints
  parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=100, help='number of epochs to train for')
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help='if a word occurs less than this number of times in training data, it is discarded')
  parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
  parser.add_argument('--disp_freq', dest='disp_freq', type=int, default=100, help='iteration frequency to display the training status')
  parser.add_argument('--valid_freq', dest='valid_freq', type=int, default=2000, help='iteration frequency to evaluate/predict on validation status and save a checkpoint')
  parser.add_argument('--checkpoint_save_dir', dest='checkpoint_save_dir', default='checkpoint', help='dirpectory path to evaluate/predict on validation status and save a checkpoint')
  parser.add_argument('--hypo_save_dir', dest='hypo_save_dir', default='hypo', help='dirpectory path to save the hypothesis and references')
  parser.add_argument('--hypo_save_freq', dest='hypo_save_freq', type=int, default=1, help='frequency to save the hypothesis and references')
  parser.add_argument('--reload_checkpoint_path', dest='reload_checkpoint_path', help='path of checkpoint to load')
  
  parser.add_argument('--word_img_embed_hidden_dim', dest='word_img_embed_hidden_dim', type=int, default=600, help='dimension of embedding of each word, the image and the hidden LSTM layer')
  parser.add_argument('--optimizer', dest='optimizer', default='rmsprop', help='the optimizer to use from sgd/adadelta/rmsprop')
  parser.add_argument('--lrate', dest='lrate', type=float, default=0.001, help='the learning rate')
  parser.add_argument('--use_dropout', dest='use_dropout', type=int, default=1, help='switch to use dropout in lstm')
  
  args = parser.parse_args()
  model_options = vars(args)  # convert to ordinary dict
  print 'parsed model_options:'
  print json.dumps(model_options, indent=2)
  main(model_options)
  
