import theano
import theano.tensor as tensor

def negative_log_likelihood(pred_softmax, gt_sents):
  n_word = gt_sents.shape[0]
  n_samples = gt_sents.shape[1]
  n_word_range = tensor.arange(n_word)
  
  def _step(_pred, _gt_idx):
    return _pred[n_word_range, _gt_idx]
  results, updates = theano.scan(_step,
                                 sequences=[pred_softmax, gt_sents.T],
                                 name='cost_func_loop',
                                 n_steps=n_samples)
  
  cost = -tensor.log(results).mean()
  return cost
