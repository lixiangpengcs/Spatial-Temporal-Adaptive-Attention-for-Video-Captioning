import numpy
import theano
import theano.tensor as tensor
from layers import Layers
from collections import OrderedDict
import utils
import copy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
from data_engine import prepare_data


class Model(object):

    def __init__(self):
        self.layers = Layers()

    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        params['Wemb'] = utils.norm_weight(options['n_words'], options['dim_word'])

        ctx_dim = options['ctx_dim']

        params = self.layers.get_layer('ff')[0](
            options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'])
        params = self.layers.get_layer('ff')[0](
            options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'])
        # decoder: LSTM
        params = self.layers.get_layer('lstm_cond')[0](options, params, prefix='bo_lstm',
                                                       nin=options['dim_word'], dim=options['dim'],
                                                       dimctx=ctx_dim)
        params = self.layers.get_layer('lstm')[0](params, nin=options['dim'], dim=options['dim'],
                                                  prefix='to_lstm')

        # readout
        params = self.layers.get_layer('ff')[0](
            options, params, prefix='ff_logit_bo',
            nin=options['dim'], nout=options['dim_word'])
        if options['ctx2out']:
            params = self.layers.get_layer('ff')[0](
                options, params, prefix='ff_logit_ctx',
                nin=ctx_dim, nout=options['dim_word'])
            params = self.layers.get_layer('ff')[0](
                options, params, prefix='ff_logit_to',
                nin=options['dim'], nout=options['dim_word'])

        params = self.layers.get_layer('ff')[0](
            options, params, prefix='ff_logit',
            nin=options['dim_word'], nout=options['n_words'])
        return params

    def build_model(self, tparams, options):
        trng = RandomStreams(1234)
        use_noise = theano.shared(numpy.float32(0.))
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype='float32')
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype='float32')
        mask_ctx = tensor.matrix('mask_ctx', dtype='float32')
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time
        emb = tparams['Wemb'][x.flatten()].reshape(
            [n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
        counts = mask_ctx.sum(-1).dimshuffle(0, 'x')

        ctx_ = ctx

        ctx0 = ctx_
        ctx_mean = ctx0.sum(1) / counts

        # initial state/cell
        init_state = self.layers.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
        init_memory = self.layers.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')
        # decoder
        bo_lstm = self.layers.get_layer('lstm_cond')[1](tparams, emb, options,
                                                        prefix='bo_lstm',
                                                        mask=mask, context=ctx0,
                                                        one_step=False,
                                                        init_state=init_state,
                                                        init_memory=init_memory,
                                                        trng=trng,
                                                        use_noise=use_noise)
        to_lstm = self.layers.get_layer('lstm')[1](tparams, bo_lstm[0],
                                                   mask=mask,
                                                   one_step=False,
                                                   prefix='to_lstm'
                                                   )

        bo_lstm_h = bo_lstm[0]
        to_lstm_h = to_lstm[0]
        alphas = bo_lstm[2]
        ctxs = bo_lstm[3]
        betas = bo_lstm[4]
        if options['use_dropout']:
            bo_lstm_h = self.layers.dropout_layer(bo_lstm_h, use_noise, trng)
            to_lstm_h = self.layers.dropout_layer(to_lstm_h, use_noise, trng)

        # compute word probabilities
        logit = self.layers.get_layer('ff')[1](
            tparams, bo_lstm_h, options, prefix='ff_logit_bo', activ='linear')
        if options['prev2out']:
            logit += emb
        if options['ctx2out']:
            to_lstm_h *= (1-betas[:, :, None])
            ctxs_beta = self.layers.get_layer('ff')[1](
                tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
            ctxs_beta += self.layers.get_layer('ff')[1](
                tparams, to_lstm_h, options, prefix='ff_logit_to', activ='linear')
            logit += ctxs_beta

        logit = utils.tanh(logit)
        if options['use_dropout']:
            logit = self.layers.dropout_layer(logit, use_noise, trng)

        # (t,m,n_words)
        logit = self.layers.get_layer('ff')[1](
            tparams, logit, options, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape
        # (t*m, n_words)
        probs = tensor.nnet.softmax(
            logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))
        # cost
        x_flat = x.flatten()  # (t*m,)
        cost = -tensor.log(probs[tensor.arange(x_flat.shape[0]), x_flat] + 1e-8)

        cost = cost.reshape([x.shape[0], x.shape[1]])
        cost = (cost * mask).sum(0)
        extra = [probs, alphas, betas]
        return trng, use_noise, x, mask, ctx, mask_ctx, cost, extra

    def build_sampler(self, tparams, options, use_noise, trng, mode=None):
        # context: #annotations x dim
        ctx0 = tensor.matrix('ctx_sampler', dtype='float32')
        # ctx0.tag.test_value = numpy.random.uniform(size=(50,1024)).astype('float32')
        ctx_mask = tensor.vector('ctx_mask', dtype='float32')
        # ctx_mask.tag.test_value = numpy.random.binomial(n=1,p=0.5,size=(50,)).astype('float32')


        ctx_ = ctx0
        counts = ctx_mask.sum(-1)

        ctx = ctx_
        ctx_mean = ctx.sum(0) / counts
        # ctx_mean = ctx.mean(0)
        ctx = ctx.dimshuffle('x', 0, 1)
        # initial state/cell
        bo_init_state = self.layers.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
        bo_init_memory = self.layers.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')
        to_init_state = tensor.alloc(0., options['dim'])
        to_init_memory = tensor.alloc(0., options['dim'])
        init_state = [bo_init_state, to_init_state]
        init_memory = [bo_init_memory, to_init_memory]

        print 'Building f_init...',
        f_init = theano.function(
            [ctx0, ctx_mask],
            [ctx0] + init_state + init_memory, name='f_init',
            on_unused_input='ignore',
            profile=False, mode=mode)
        print 'Done'

        x = tensor.vector('x_sampler', dtype='int64')
        init_state = [tensor.matrix('bo_init_state', dtype='float32'),
                      tensor.matrix('to_init_state', dtype='float32')]
        init_memory = [tensor.matrix('bo_init_memory', dtype='float32'),
                       tensor.matrix('to_init_memory', dtype='float32')]

        # if it's the first word, emb should be all zero
        emb = tensor.switch(x[:, None] < 0, tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                            tparams['Wemb'][x])

        bo_lstm = self.layers.get_layer('lstm_cond')[1](tparams, emb, options,
                                                        prefix='bo_lstm',
                                                        mask=None, context=ctx,
                                                        one_step=True,
                                                        init_state=init_state[0],
                                                        init_memory=init_memory[0],
                                                        trng=trng,
                                                        use_noise=use_noise,
                                                        mode=mode)
        to_lstm = self.layers.get_layer('lstm')[1](tparams, bo_lstm[0],
                                                   mask=None,
                                                   one_step=True,
                                                   init_state=init_state[1],
                                                   init_memory=init_memory[1],
                                                   prefix='to_lstm'
                                                   )
        next_state = [bo_lstm[0], to_lstm[0]]
        next_memory = [bo_lstm[1], to_lstm[0]]

        bo_lstm_h = bo_lstm[0]
        to_lstm_h = to_lstm[0]
        alphas = bo_lstm[2]
        ctxs = bo_lstm[3]
        betas = bo_lstm[4]
        if options['use_dropout']:
            bo_lstm_h = self.layers.dropout_layer(bo_lstm_h, use_noise, trng)
            to_lstm_h = self.layers.dropout_layer(to_lstm_h, use_noise, trng)

        logit = self.layers.get_layer('ff')[1](
            tparams, bo_lstm_h, options, prefix='ff_logit_bo', activ='linear')
        if options['prev2out']:
            logit += emb
        if options['ctx2out']:
            to_lstm_h *= (1-betas[:, None])
            ctxs_beta = self.layers.get_layer('ff')[1](
                tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
            ctxs_beta += self.layers.get_layer('ff')[1](
                tparams, to_lstm_h, options, prefix='ff_logit_to', activ='linear')
            logit += ctxs_beta
        logit = utils.tanh(logit)
        if options['use_dropout']:
            logit = self.layers.dropout_layer(logit, use_noise, trng)

        logit = self.layers.get_layer('ff')[1](
            tparams, logit, options, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape
        next_probs = tensor.nnet.softmax(logit)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # next word probability
        print 'building f_next...'
        f_next = theano.function(
            [x, ctx0, ctx_mask] + init_state + init_memory,
            [next_probs, next_sample] + next_state + next_memory,
            name='f_next', profile=False, mode=mode, on_unused_input='ignore')
        print 'Done'
        return f_init, f_next

    def gen_sample(self, tparams, f_init, f_next, ctx0, ctx_mask, options,
                   trng=None, k=1, maxlen=30, stochastic=False,
                   restrict_voc=False):
        '''
        ctx0: (26,1024)
        ctx_mask: (26,)

        restrict_voc: set the probability of outofvoc words with 0, renormalize
        '''

        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'

        sample = []
        sample_score = []
        if stochastic:
            sample_score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []
        hyp_memories = []

        # [(26,1024),(512,),(512,)]
        rval = f_init(ctx0, ctx_mask)
        ctx0 = rval[0]

        next_state = []
        next_memory = []
        n_layers_lstm = 2

        for lidx in xrange(n_layers_lstm):
            next_state.append(rval[1 + lidx])
            next_state[-1] = next_state[-1].reshape([live_k, next_state[-1].shape[0]])
        for lidx in xrange(n_layers_lstm):
            next_memory.append(rval[1 + n_layers_lstm + lidx])
            next_memory[-1] = next_memory[-1].reshape([live_k, next_memory[-1].shape[0]])
        next_w = -1 * numpy.ones((1,)).astype('int64')
        # next_state: [(1,512)]
        # next_memory: [(1,512)]
        for ii in xrange(maxlen):
            # return [(1, 50000), (1,), (1, 512), (1, 512)]
            # next_w: vector
            # ctx: matrix
            # ctx_mask: vector
            # next_state: [matrix]
            # next_memory: [matrix]
            rval = f_next(*([next_w, ctx0, ctx_mask] + next_state + next_memory))
            next_p = rval[0]
            if restrict_voc:
                raise NotImplementedError()
            next_w = rval[1]  # already argmax sorted
            next_state = []
            for lidx in xrange(n_layers_lstm):
                next_state.append(rval[2 + lidx])
            next_memory = []
            for lidx in xrange(n_layers_lstm):
                next_memory.append(rval[2 + n_layers_lstm + lidx])
            if stochastic:
                sample.append(next_w[0])  # take the most likely one
                sample_score += next_p[0, next_w[0]]
                if next_w[0] == 0:
                    break
            else:
                # the first run is (1,50000)
                cand_scores = hyp_scores[:, None] - numpy.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k - dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size  # index of row
                word_indices = ranks_flat % voc_size  # index of col
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
                new_hyp_states = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_states.append([])
                new_hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_memories.append([])

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    for lidx in xrange(n_layers_lstm):
                        new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                    for lidx in xrange(n_layers_lstm):
                        new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                for lidx in xrange(n_layers_lstm):
                    hyp_states.append([])
                hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    hyp_memories.append([])

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_states[lidx].append(new_hyp_states[lidx][idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
                hyp_scores = numpy.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = []
                for lidx in xrange(n_layers_lstm):
                    next_state.append(numpy.array(hyp_states[lidx]))
                next_memory = []
                for lidx in xrange(n_layers_lstm):
                    next_memory.append(numpy.array(hyp_memories[lidx]))

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        return sample, sample_score, next_state, next_memory

    def pred_probs(self, engine, whichset, f_log_probs, verbose=True):

        probs = []
        n_done = 0
        NLL = []
        L = []
        if whichset == 'train':
            tags = engine.train
            iterator = engine.kf_train
        elif whichset == 'valid':
            tags = engine.valid
            iterator = engine.kf_valid
        elif whichset == 'test':
            tags = engine.test
            iterator = engine.kf_test
        else:
            raise NotImplementedError()
        n_samples = numpy.sum([len(index) for index in iterator])
        for index in iterator:
            tag = [tags[i] for i in index]
            x, mask, ctx, ctx_mask = prepare_data(engine, tag)
            pred_probs = f_log_probs(x, mask, ctx, ctx_mask)
            L.append(mask.sum(0).tolist())
            NLL.append((-1 * pred_probs).tolist())
            probs.append(pred_probs.tolist())
            n_done += len(tag)
            if verbose:
                sys.stdout.write('\rComputing LL on %d/%d examples' % (
                    n_done, n_samples))
                sys.stdout.flush()
        print
        probs = utils.flatten_list_of_list(probs)
        NLL = utils.flatten_list_of_list(NLL)
        L = utils.flatten_list_of_list(L)
        perp = 2 ** (numpy.sum(NLL) / numpy.sum(L) / numpy.log(2))
        return -1 * numpy.mean(probs), perp

    def sample_execute(self, engine, options, tparams, f_init, f_next, x, ctx, ctx_mask, trng):
        stochastic = False
        for jj in xrange(numpy.minimum(10, x.shape[1])):
            sample, score, _, _ = self.gen_sample(tparams, f_init, f_next, ctx[jj], ctx_mask[jj],
                                                  options, trng=trng, k=5, maxlen=30, stochastic=stochastic)
            if not stochastic:
                best_one = numpy.argmin(score)
                sample = sample[best_one]
            else:
                sample = sample
            print 'Truth ', jj, ': ',
            for vv in x[:, jj]:
                if vv == 0:
                    break
                if vv in engine.word_idict:
                    print engine.word_idict[vv],
                else:
                    print 'UNK',
            print
            for kk, ss in enumerate([sample]):
                print 'Sample (', jj, ') ', ': ',
                for vv in ss:
                    if vv == 0:
                        break
                    if vv in engine.word_idict:
                        print engine.word_idict[vv],
                    else:
                        print 'UNK',
            print
