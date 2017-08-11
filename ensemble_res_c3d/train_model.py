'''
Build a soft-attention-based video caption generator
'''
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import os, sys
import time

import data_engine
import metrics
import utils

sys.path.append('./model_hLSTMat/')
from optimizers import adadelta
from layers import Layers
from model import Model

from config import config
from jobman import DD, expand


def train(random_seed=1234,
          dim_word=256, # word vector dimensionality
          ctx_dim=-1, # context vector dimensionality, auto set
          dim=1000, # the number of LSTM units
          n_layers_out=1,
          n_layers_init=1,
          encoder='none',
          encoder_dim=100,
          prev2out=False,
          ctx2out=False,
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,
          alpha_c=0.,
          alpha_entropy_r=0.,
          lrate=0.01,
          selector=False,
          n_words=100000,
          maxlen=100, # maximum length of the description
          optimizer='adadelta',
          clip_c=2.,
          batch_size = 64,
          valid_batch_size = 64,
          save_model_dir='/data/lisatmp3/yaoli/exp/capgen_vid/attention/test/',
          validFreq=10,
          saveFreq=10, # save the parameters after every saveFreq updates
          sampleFreq=10, # generate some samples after every sampleFreq updates
          metric='blue',
          dataset='youtube2text',
          video_feature='googlenet',
          use_dropout=False,
          reload_=False,
          from_dir=None,
          K1=10,
          K2=10,
          OutOf=240,
          verbose=True,
          debug=True
          ):
    rng_numpy, rng_theano = utils.get_two_rngs()

    model_options = locals().copy()
    model_options_c = locals().copy()
    if 'self' in model_options:
        del model_options['self']
    with open('model_files/model_options.pkl', 'wb') as f:
        pkl.dump(model_options, f)
    with open('model_files/model_options_c3d.pkl', 'wb') as f:
        pkl.dump(model_options_c, f)

    # instance model
    layers = Layers()
    model = Model()
    model_c = Model()

    print 'Loading data'
    engine = data_engine.Movie2Caption('attention', dataset,
                                       video_feature,
                                       batch_size, valid_batch_size,
                                       maxlen, n_words,
                                       K1, K2, OutOf)
    model_options['ctx_dim'] = engine.ctx_dim
    model_options_c['ctx_dim'] = engine.ctx_dim_c
    model_options['n_words'] = engine.n_words
    model_options_c['n_words'] = engine.n_words
    print 'n_words:', model_options['n_words']
    print model_options_c['dim'],model_options_c['ctx_dim']

    # set test values, for debugging
    idx = engine.kf_train[0]
    [x_tv, mask_tv,
     ctx_tv, ctx_mask_tv,
     ctx_tv_c, ctx_mask_tv_c] = data_engine.prepare_data(
        engine, [engine.train[index] for index in idx])

    print 'init params'
    t0 = time.time()
    params = model.init_params(model_options)
    params_c = model_c.init_params(model_options_c)
    # reloading
    model_saved = 'model_files/model_resnet.npz'
    model_saved_c = 'model_files/model_c3d.npz'
    assert os.path.isfile(model_saved)
    print "Reloading model params..."
    params = utils.load_params(model_saved, params)
    params_c = utils.load_params(model_saved_c, params_c)

    tparams = utils.init_tparams(params)
    tparams_c = utils.init_tparams(params_c)

    trng, use_noise, \
          x, mask, ctx, mask_ctx, \
          cost, extra = \
          model.build_model(tparams, model_options)
    alphas = extra[1]
    betas = extra[2]

    trng_c, use_noise_c, \
    x_c, mask_c, ctx_c, mask_ctx_c, \
    cost_c, extra_c = \
        model_c.build_model(tparams_c, model_options_c)


    alphas_c = extra_c[1]
    betas_c = extra_c[2]

    print 'buliding sampler'
    f_init, f_next = model.build_sampler(tparams, model_options, use_noise, trng)
    f_init_c, f_next_c = model_c.build_sampler(tparams_c, model_options_c, use_noise_c, trng_c)
    # before any regularizer
    print 'building f_log_probs'
    f_log_probs = theano.function([x, mask, ctx, mask_ctx], -cost,
                                  profile=False, on_unused_input='ignore')
    f_log_probs_c = theano.function([x_c, mask_c, ctx_c, mask_ctx_c], -cost_c,
                                  profile=False, on_unused_input='ignore')

    bad_counter = 0

    processes = None
    queue = None
    rqueue = None
    shared_params = None

    uidx = 0
    uidx_best_blue = 0
    uidx_best_valid_err = 0
    estop = False
    best_p = utils.unzip(tparams)
    best_blue_valid = 0
    best_valid_err = 999
    alphas_ratio = []
    for eidx in xrange(max_epochs):
        n_samples = 0
        train_costs = []
        grads_record = []
        print 'Epoch ', eidx
        for idx in engine.kf_train:
            tags = [engine.train[index] for index in idx]
            n_samples += len(tags)
            use_noise.set_value(1.)

            pd_start = time.time()
            x, mask, ctx, ctx_mask, ctx_c, ctx_mask_c = data_engine.prepare_data(
                engine, tags)
            #print 'x:',x.shape,'ctx:',ctx.shape,'ctx_c:',ctx_c.shape
            pd_duration = time.time() - pd_start
            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            if numpy.mod(uidx, saveFreq) == 0:
                pass

            if numpy.mod(uidx, sampleFreq) == 0:
                use_noise.set_value(0.)
                print '------------- sampling from train ----------'
                x_s = x
                mask_s = mask
                ctx_s = ctx
                ctx_s_c = ctx_c
                ctx_mask_s = ctx_mask
                ctx_mask_s_c = ctx_mask_c
                model.sample_execute_ensemble(engine, model_options,model_options_c, tparams,tparams_c,
                                          f_init,f_init_c, f_next,f_next_c, x_s, ctx_s,
                                          ctx_mask_s, ctx_s_c, ctx_mask_s_c, trng)
                print '------------- sampling from valid ----------'
                idx = engine.kf_valid[numpy.random.randint(1, len(engine.kf_valid) - 1)]
                tags = [engine.valid[index] for index in idx]
                x_s, mask_s, ctx_s, mask_ctx_s, ctx_s_c,mask_ctx_s_c = data_engine.prepare_data(engine, tags)
                model.sample_execute_ensemble(engine, model_options,model_options_c, tparams,tparams_c,
                                          f_init, f_init_c, f_next, f_next_c, x_s, ctx_s,
                                     mask_ctx_s, ctx_s_c, mask_ctx_s_c, trng)

            if validFreq != -1 and numpy.mod(uidx, validFreq) == 0:
                current_params = utils.unzip(tparams)

                use_noise.set_value(0.)
                train_err = -1
                train_perp = -1
                valid_err = -1
                valid_perp = -1
                test_err = -1
                test_perp = -1

                mean_ranking = 0
                blue_t0 = time.time()
                scores, processes, queue, rqueue, shared_params = \
                    metrics.compute_score_ensemble(
                    model_type='attention',
                    model_archive=current_params,
                    options=model_options,
                    options_c=model_options_c,
                    engine=engine,
                    save_dir=save_model_dir,
                    beam=5, n_process=5,
                    whichset='both',
                    on_cpu=False,
                    processes=processes, queue=queue, rqueue=rqueue,
                    shared_params=shared_params, metric=metric,
                    one_time=False,
                    f_init=f_init, f_init_c=f_init_c, f_next=f_next, f_next_c= f_next_c, model=model
                    )
                '''
                 {'blue': {'test': [-1], 'valid': [77.7, 60.5, 48.7, 38.5, 38.3]},
                 'alternative_valid': {'Bleu_3': 0.40702270203174923,
                 'Bleu_4': 0.29276570520368456,
                 'CIDEr': 0.25247168210607884,
                 'Bleu_2': 0.529069629270047,
                 'Bleu_1': 0.6804308797115253,
                 'ROUGE_L': 0.51083584331688392},
                 'meteor': {'test': [-1], 'valid': [0.282787550236724]}}
                '''

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
                  'blue score: %.1f, meteor score: %.1f'%(
                time.time()-blue_t0, valid_B4, valid_meteor)


                if test_B4>0.52 and test_meteor>0.32:
                    print 'Saving to %s...'%save_model_dir,
                    numpy.savez(
                        save_model_dir+'model_'+str(uidx)+'.npz',
                         **current_params)

                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, \
                  'best valid err so far',best_valid_err
                print 'valid took %.2f sec'%(time.time() - t0_valid)
                # end of validatioin
                sys.exit()
            if debug:
                break
        if estop:
            break
        if debug:
            break

        # end for loop over minibatches
        print 'This epoch has seen %d samples, train cost %.2f'%(
            n_samples, numpy.mean(train_costs))
    # end for loop over epochs
    print 'Optimization ended.'
    if best_p is not None:
        utils.zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = 0
    test_err = 0
    if not debug:
        #if valid:
        valid_err, valid_perp = model.pred_probs(
            engine, 'valid', f_log_probs,
            verbose=model_options['verbose'])
        #if test:
        #test_err, test_perp = self.pred_probs(
        #    'test', f_log_probs,
        #    verbose=model_options['verbose'])


    print 'stopped at epoch %d, minibatch %d, '\
      'curent Train %.2f, current Valid %.2f, current Test %.2f '%(
          eidx,uidx,numpy.mean(train_err),numpy.mean(valid_err),numpy.mean(test_err))
    params = copy.copy(best_p)
    numpy.savez(save_model_dir+'model_best.npz',
                train_err=train_err,
                valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                **params)

    if history_errs != []:
        history = numpy.asarray(history_errs)
        best_valid_idx = history[:,6].argmin()
        numpy.savetxt(save_model_dir+'train_valid_test.txt', history, fmt='%.4f')
        print 'final best exp ', history[best_valid_idx]

    return train_err, valid_err, test_err


def train_from_scratch(config, state, channel):
    # Model options
    save_model_dir = config[config.model].save_model_dir
    if save_model_dir == 'current':
        config[config.model].save_model_dir = './'
        save_model_dir = './'
        # to facilitate the use of cluster for multiple jobs
        save_path = './model_config.pkl'
    else:
        # run locally, save locally
        save_path = save_model_dir + 'model_config.pkl'
    print 'current save dir ', save_model_dir
    utils.create_dir_if_not_exist(save_model_dir)

    reload_ = config[config.model].reload_
    if reload_:
        print 'preparing reload'
        save_dir_backup = config[config.model].save_model_dir
        from_dir_backup = config[config.model].from_dir
        # never start retrain in the same folder
        assert save_dir_backup != from_dir_backup
        print 'save dir ', save_dir_backup
        print 'from_dir ', from_dir_backup
        print 'setting current model config with the old one'
        model_config_old = utils.load_pkl('model_files/model_config.pkl')
        utils.set_config(config, model_config_old)
        config[config.model].save_model_dir = save_dir_backup
        config[config.model].from_dir = from_dir_backup
        config[config.model].reload_ = True
    if config.erase_history:
        print 'erasing everything in ', save_model_dir
        os.system('rm %s/*' % save_model_dir)
    # for stdout file logging
    # sys.stdout = Unbuffered(sys.stdout, state.save_model_path + 'stdout.log')
    print 'saving model config into %s' % save_path
    utils.dump_pkl(config, save_path)
    # Also copy back from config into state.
    for key in config:
        setattr(state, key, config[key])
    model_type = config.model
    print 'Model Type: %s' % model_type
    print 'Command: %s' % ' '.join(sys.argv)

    t0 = time.time()
    print 'training an attention model'
    train(**state.attention)
    if channel:
        channel.save()
    print 'training time in total %.4f sec' % (time.time() - t0)


def main(state, channel=None):
    utils.set_config(config, state)
    train_from_scratch(config, state, channel)


if __name__ == '__main__':
    args = {}
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            args[k] = v
    except:
        print 'args must be like a=X b.c=X'
        exit(1)

    state = expand(args)
    sys.exit(main(state))
