import time

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data_engine
import metrics
from config import config
from model_hLSTMat.model import Model
import utils
import os

import theano
import theano.tensor as tensor
import numpy


def test(model_options_file='model_options.pkl',
         model_file='model_best_so_far.npz'):
    from_dir = 'model_files/'
    print 'preparing reload'
    model_options = utils.load_pkl(from_dir+model_options_file)

    print 'Loading data'
    engine = data_engine.Movie2Caption('attention',
                                       model_options['dataset'],
                                       model_options['video_feature'],
                                       model_options['batch_size'],
                                       model_options['valid_batch_size'],
                                       model_options['maxlen'],
                                       model_options['n_words'],
                                       model_options['K'],
                                       model_options['OutOf'])

    print 'init params'
    t0 = time.time()
    model = Model()
    params = model.init_params(model_options)

    model_saved = from_dir + model_file
    assert os.path.isfile(model_saved)
    print "Reloading model params..."
    params = utils.load_params(model_saved, params)
    tparams = utils.init_tparams(params)
    print tparams.keys

    print 'buliding sampler'
    use_noise = theano.shared(numpy.float32(0.))
    use_noise.set_value(0.)
    trng = RandomStreams(1234)
    f_init, f_next = model.build_sampler(tparams, model_options, use_noise, trng)

    print 'start test...'
    blue_t0 = time.time()
    scores, processes, queue, rqueue, shared_params = \
                    metrics.compute_score(
                    model_type='attention',
                    model_archive=params,
                    options=model_options,
                    engine=engine,
                    save_dir=from_dir,
                    beam=5, n_process=5,
                    whichset='both',
                    on_cpu=False,
                    processes=None, queue=None, rqueue=None,
                    shared_params=None, metric=model_options['metric'],
                    one_time=False,
                    f_init=f_init, f_next=f_next, model=model
                    )

    valid_B1 = scores['valid']['Bleu_1']
    valid_B2 = scores['valid']['Bleu_2']
    valid_B3 = scores['valid']['Bleu_3']
    valid_B4 = scores['valid']['Bleu_4']
    valid_Rouge = scores['valid']['ROUGE_L']
    valid_Cider = scores['valid']['CIDEr']
    valid_meteor = scores['valid']['METEOR']
    test_B1 = scores['test']['Bleu_1']
    test_B2 = scores['test']['Bleu_2']
    test_B3 = scores['test']['Bleu_3']
    test_B4 = scores['test']['Bleu_4']
    test_Rouge = scores['test']['ROUGE_L']
    test_Cider = scores['test']['CIDEr']
    test_meteor = scores['test']['METEOR']
    print 'computing meteor/blue score used %.4f sec, '\
          'B@1: %.3f, B@2: %.3f, B@3: %.3f, B@4: %.3f, M: %.3f'%(
    time.time()-blue_t0, test_B1, test_B2, test_B3, test_B4, test_meteor)


if __name__ == '__main__':
    test(model_file='model_10000.npz')
