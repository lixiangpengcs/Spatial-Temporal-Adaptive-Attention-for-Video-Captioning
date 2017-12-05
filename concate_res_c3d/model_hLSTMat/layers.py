import theano
import numpy
import theano.tensor as tensor
from utils import _p, norm_weight, ortho_weight, tanh, linear

class Layers(object):

    def __init__(self):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('self.param_init_fflayer', 'self.fflayer'),
            'lstm': ('self.param_init_lstm', 'self.lstm_layer'),
            'lstm_cond': ('self.param_init_lstm_cond', 'self.lstm_cond_layer'),
            }

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return (eval(fns[0]), eval(fns[1]))

    # dropout
    def dropout_layer(self, state_before, use_noise, trng):
        proj = tensor.switch(use_noise,
                             state_before *
                             trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                             state_before * 0.5)
        return proj

    def param_init_fflayer(self, options, params, prefix='ff', nin=None, nout=None):
        if nin == None:
            nin = options['dim_proj']
        if nout == None:
            nout = options['dim_proj']
        params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
        return params

    def fflayer(self, tparams, state_below, options,
                prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
        return eval(activ)(tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[
            _p(prefix, 'b')])

    # LSTM layer
    def param_init_lstm(self, params, nin, dim, prefix='lstm'):
        assert prefix is not None
        # Stack the weight matricies for faster dot prods
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U
        params[_p(prefix, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

        return params

    # This function implements the lstm fprop
    def lstm_layer(self, tparams, state_below, mask=None, init_state=None, init_memory=None,
                   one_step=False, prefix='lstm', **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling

        if one_step:
            assert init_memory, 'previous memory must be provided'
            assert init_state, 'previous state must be provided'

        n_steps = state_below.shape[0]
        dim = tparams[_p(prefix, 'U')].shape[0]

        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        # mask
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        # initial/previous state
        if init_state == None:
            init_state = tensor.alloc(0., n_samples, dim)
        # initial/previous memory
        if init_memory == None:
            init_memory = tensor.alloc(0., n_samples, dim)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            elif _x.ndim == 2:
                return _x[:, n * dim:(n + 1) * dim]
            return _x[n * dim:(n + 1) * dim]

        U = tparams[_p(prefix, 'U')]
        b = tparams[_p(prefix, 'b')]

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, U)
            preact += x_

            i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
            o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
            c = tensor.tanh(_slice(preact, 3, dim))

            c = f * c_ + i * c
            h = o * tensor.tanh(c)
            if m_.ndim == 0:
                # when using this for minibatchsize=1
                h = m_ * h + (1. - m_) * h_
                c = m_ * c + (1. - m_) * c_
            else:
                h = m_[:, None] * h + (1. - m_)[:, None] * h_
                c = m_[:, None] * c + (1. - m_)[:, None] * c_
            return h, c

        state_below = tensor.dot(
            state_below, tparams[_p(prefix, 'W')]) + b

        if one_step:
            rval = _step(mask, state_below, init_state, init_memory)
        else:
            rval, updates = theano.scan(_step,
                                        sequences=[mask, state_below],
                                        outputs_info=[init_state, init_memory],
                                        name=_p(prefix, '_layers'),
                                        n_steps=n_steps,
                                        profile=False
                                        )
        return rval

    # Conditional LSTM layer with Attention
    def param_init_lstm_cond(self, options, params,
                             prefix='lstm_cond', nin=None, dim=None, dimctx=None):
        if nin == None:
            nin = options['dim']
        if dim == None:
            dim = options['dim']
        if dimctx == None:
            dimctx = options['dim']
        # input to LSTM
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W

        # LSTM to LSTM
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U

        # bias to LSTM
        params[_p(prefix, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

        # context to LSTM
        # Wc = norm_weight(dimctx, dim * 4)
        # params[_p(prefix, 'Wc')] = Wc

        # attention: context -> hidden
        Wc_att = norm_weight(dimctx, ortho=False)
        params[_p(prefix, 'Wc_att')] = Wc_att

        # attention: LSTM -> hidden
        Wd_att = norm_weight(dim, dimctx)
        params[_p(prefix, 'Wd_att')] = Wd_att

        # attention: hidden bias
        b_att = numpy.zeros((dimctx,)).astype('float32')
        params[_p(prefix, 'b_att')] = b_att

        # attention:
        U_att = norm_weight(dimctx, 1)
        params[_p(prefix, 'U_att')] = U_att
        c_att = numpy.zeros((1,)).astype('float32')
        params[_p(prefix, 'c_tt')] = c_att

        if options['selector']:
            # attention: selector
            W_sel = norm_weight(dim, 1)
            params[_p(prefix, 'W_sel')] = W_sel
            b_sel = numpy.float32(0.)
            params[_p(prefix, 'b_sel')] = b_sel

        return params

    def lstm_cond_layer(self, tparams, state_below, options, prefix='lstm',
                        mask=None, context=None, one_step=False,
                        init_memory=None, init_state=None,
                        trng=None, use_noise=None, mode=None,
                        **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling
        # mask (t, m)
        # context (m, f, dim_ctx), or (f, dim_word) in sampling
        # init_memory, init_state (m, dim)
        assert context, 'Context must be provided'

        if one_step:
            assert init_memory, 'previous memory must be provided'
            assert init_state, 'previous state must be provided'

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        # mask
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        dim = tparams[_p(prefix, 'U')].shape[0]

        # initial/previous state
        if init_state == None:
            init_state = tensor.alloc(0., n_samples, dim)
        # initial/previous memory
        if init_memory == None:
            init_memory = tensor.alloc(0., n_samples, dim)

        # projected context
        pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + tparams[
            _p(prefix, 'b_att')]
        if one_step:
            # tensor.dot will remove broadcasting dim
            pctx_ = tensor.addbroadcast(pctx_, 0)
        # projected x
        state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[
            _p(prefix, 'b')]

        Wd_att = tparams[_p(prefix, 'Wd_att')]
        U_att = tparams[_p(prefix, 'U_att')]
        c_att = tparams[_p(prefix, 'c_tt')]
        if options['selector']:
            W_sel = tparams[_p(prefix, 'W_sel')]
            b_sel = tparams[_p(prefix, 'b_sel')]
        else:
            W_sel = tensor.alloc(0., 1)
            b_sel = tensor.alloc(0., 1)
        U = tparams[_p(prefix, 'U')]
        # Wc = tparams[_p(prefix, 'Wc')]

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_,  # sequences
                  h_, c_,  # outputs_info
                  pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U,  # non_sequences
                  dp_=None, dp_att_=None):

            preact = tensor.dot(h_, U)
            preact += x_
            # preact += tensor.dot(ctx_, Wc)

            i = _slice(preact, 0, dim)
            f = _slice(preact, 1, dim)
            o = _slice(preact, 2, dim)
            if options['use_dropout']:
                i = i * _slice(dp_, 0, dim)
                f = f * _slice(dp_, 1, dim)
                o = o * _slice(dp_, 2, dim)
            i = tensor.nnet.sigmoid(i)
            f = tensor.nnet.sigmoid(f)
            o = tensor.nnet.sigmoid(o)
            c = tensor.tanh(_slice(preact, 3, dim))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            # attention
            pstate_ = tensor.dot(h, Wd_att)
            pctx_ = pctx_ + pstate_[:, None, :]
            pctx_ = tanh(pctx_)

            alpha = tensor.dot(pctx_, U_att) + c_att
            alpha_pre = alpha
            alpha_shp = alpha.shape
            alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0], alpha_shp[1]]))  # softmax
            ctx_ = (context * alpha[:, :, None]).sum(1)  # (m, ctx_dim)
            if options['selector']:
                sel_ = tensor.nnet.sigmoid(tensor.dot(h_, W_sel) + b_sel)
                sel_ = sel_.reshape([sel_.shape[0]])
                ctx_ = sel_[:, None] * ctx_

            rval = [h, c, alpha, ctx_, sel_, pstate_, pctx_, i, f, o, preact, alpha_pre]
            return rval

        if options['use_dropout']:
            _step0 = lambda m_, x_, dp_, h_, c_, \
                            pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U: _step(
                m_, x_, h_, c_,
                pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, dp_)
            dp_shape = state_below.shape
            if one_step:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], 3 * dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], 3 * dim))
            else:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], dp_shape[1], 3 * dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], dp_shape[1], 3 * dim))
        else:
            _step0 = lambda m_, x_, h_, c_, \
                            pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U: _step(
                m_, x_, h_, c_, pctx_, context,
                Wd_att, U_att, c_att, W_sel, b_sel, U)

        if one_step:
            if options['use_dropout']:
                rval = _step0(
                    mask, state_below, dp_mask, init_state, init_memory,
                    pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U)
            else:
                rval = _step0(mask, state_below, init_state, init_memory,
                              pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U)
        else:
            seqs = [mask, state_below]
            if options['use_dropout']:
                seqs += [dp_mask]
            rval, updates = theano.scan(
                _step0,
                sequences=seqs,
                outputs_info=[init_state,
                              init_memory,
                              None, None, None,
                              None, None, None, None, None, None, None],
                non_sequences=[pctx_, context,
                               Wd_att, U_att, c_att, W_sel, b_sel, U],
                name=_p(prefix, '_layers'),
                n_steps=nsteps, profile=False, mode=mode, strict=True)

        return rval