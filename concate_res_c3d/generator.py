import time

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data_engine
from config import config
from model_hLSTMat.model import Model
import utils
import os

import theano
import numpy


def generate(model_options_file='model_options.pkl',
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

    feat = numpy.load('datas/vid1715.npy')
    ctx = engine.get_sub_frames(feat)
    ctx_mask = engine.get_ctx_mask(ctx)

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

    print 'start generate...'
    g_t0 = time.time()
    sample, sample_score, _, _ = model.gen_sample(None, f_init, f_next, ctx, ctx_mask, model_options,
                                                  None, 5, maxlen=model_options['maxlen'])
    print sample
    # best_one = numpy.argmin(sample_score)
    # sample = sample[best_one]
    for s in sample:
        for kk, ss in enumerate([s]):
            for vv in ss:
                if vv == 0:
                    break
                if vv in engine.word_idict:
                    print engine.word_idict[vv],
                else:
                    print 'UNK',
        print

if __name__ == '__main__':
    generate(model_file='model_10000.npz')
