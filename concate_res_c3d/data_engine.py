import cPickle as pkl
import gzip
import os, socket, shutil
import sys, re
import time
from collections import OrderedDict
import numpy
# import tables
import theano
import utils
import config

from multiprocessing import Process, Queue, Manager

hostname = socket.gethostname()
                
class Movie2Caption(object):
            
    def __init__(self, model_type, signature, video_feature,
                 mb_size_train, mb_size_test, maxlen, n_words,
                 n_frames=None, outof=None
                 ):
        self.signature = signature
        self.model_type = model_type
        self.video_feature = video_feature
        self.maxlen = maxlen
        self.n_words = n_words
        self.K = n_frames
        self.OutOf = outof

        self.mb_size_train = mb_size_train
        self.mb_size_test = mb_size_test
        self.non_pickable = []
        
        self.load_data()
        
    def _filter_googlenet(self, vidID):
        feat = numpy.load(os.path.join(self.FEAT_ROOT, vidID + '.npy'))
        feat_c = numpy.load(os.path.join(self.FEAT_ROOT_C, vidID + '.npy'))
        #feat = self.FEAT[vidID]
        feat = self.get_sub_frames(feat)
        feat_c = self.get_sub_frames(feat_c)
        feat = numpy.concatenate((feat,feat_c),axis=1)
        return feat
    
    def get_video_features(self, vidID):
        if self.video_feature == 'googlenet':
            y = self._filter_googlenet(vidID)
        else:
            raise NotImplementedError()
        return y

    def pad_frames(self, frames, limit, jpegs):
        # pad frames with 0, compatible with both conv and fully connected layers
        last_frame = frames[-1]
        if jpegs:
            frames_padded = frames + [last_frame]*(limit-len(frames))
        else:
            padding = numpy.asarray([last_frame * 0.]*(limit-len(frames)))
            frames_padded = numpy.concatenate([frames, padding], axis=0)
        return frames_padded
    
    def extract_frames_equally_spaced(self, frames, how_many):
        # chunk frames into 'how_many' segments and use the first frame
        # from each segment
        n_frames = len(frames)
        splits = numpy.array_split(range(n_frames), self.K)
        idx_taken = [s[0] for s in splits]
        sub_frames = frames[idx_taken]
        return sub_frames
    
    def add_end_of_video_frame(self, frames):
        if len(frames.shape) == 4:
            # feat from conv layer
            _,a,b,c = frames.shape
            eos = numpy.zeros((1,a,b,c),dtype='float32') - 1.
        elif len(frames.shape) == 2:
            # feat from full connected layer
            _,b = frames.shape
            eos = numpy.zeros((1,b),dtype='float32') - 1.
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        frames = numpy.concatenate([frames, eos], axis=0)
        return frames
    
    def get_sub_frames(self, frames, jpegs=False):
        # from all frames, take K of them, then add end of video frame
        # jpegs: to be compatible with visualizations
        if self.OutOf:
            raise NotImplementedError('OutOf has to be None')
            frames_ = frames[:self.OutOf]
            if len(frames_) < self.OutOf:
                frames_ = self.pad_frames(frames_, self.OutOf, jpegs)
        else:
            if len(frames) < self.K:
                #frames_ = self.add_end_of_video_frame(frames)
                frames_ = self.pad_frames(frames, self.K, jpegs)
            else:

                frames_ = self.extract_frames_equally_spaced(frames, self.K)
                #frames_ = self.add_end_of_video_frame(frames_)
        if jpegs:
            frames_ = numpy.asarray(frames_)
        return frames_

    def prepare_data_for_blue(self, whichset):
        # assume one-to-one mapping between ids and features
        feats = []
        feats_mask = []
        if whichset == 'valid':
            ids = self.valid_ids
        elif whichset == 'test':
            ids = self.test_ids
        elif whichset == 'train':
            ids = self.train_ids
        for i, vidID in enumerate(ids):
            feat = self.get_video_features(vidID)
            feats.append(feat)
            feat_mask = self.get_ctx_mask(feat)
            feats_mask.append(feat_mask)
        return feats, feats_mask
    
    def get_ctx_mask(self, ctx):
        if ctx.ndim == 3:
            rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 2:
            rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 5 or ctx.ndim == 4:
            assert self.video_feature == 'oxfordnet_conv3_512'
            # in case of oxfordnet features
            # (m, 26, 512, 14, 14)
            rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        
        return rval
        
    def load_data(self):
        print 'loading youtube2text %s features'%self.video_feature
        dataset_path = config.RAB_DATASET_BASE_PATH
        feature_path = config.RAB_FEATURE_BASE_PATH
        feature_path_c = config.RAB_FEATURE_BASE_PATH_C
        self.train = utils.load_pkl(dataset_path + 'train.pkl')
        self.valid = utils.load_pkl(dataset_path + 'valid.pkl')
        self.test = utils.load_pkl(dataset_path + 'test.pkl')
        self.CAP = utils.load_pkl(dataset_path + 'CAP.pkl')
        self.FEAT_ROOT = feature_path
        self.FEAT_ROOT_C = feature_path_c
        if self.signature == 'youtube2text':
            self.train_ids = ['vid%s' % i for i in range(1, 1201)]
            self.valid_ids = ['vid%s' % i for i in range(1201, 1301)]
            self.test_ids = ['vid%s' % i for i in range(1301, 1971)]
            self.worddict = utils.load_pkl(dataset_path + 'worddict.pkl')
        elif self.signature == 'msr-vtt':
            self.train_ids = ['video%s' % i for i in range(0, 6513)]
            self.valid_ids = ['video%s' % i for i in range(6513, 7010)]
            self.test_ids = ['video%s' % i for i in range(7010, 10000)]
            self.worddict = utils.load_pkl(dataset_path + 'worddict_large.pkl')
        else:
            raise NotImplementedError()

        self.word_idict = dict()
        # wordict start with index 2
        for kk, vv in self.worddict.iteritems():
            self.word_idict[vv] = kk
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = 'UNK'

        # if len(self.word_idict) < self.n_words:
        self.n_words = len(self.word_idict)
        
        if self.video_feature == 'googlenet':
            self.ctx_dim = 6144
        else:
            raise NotImplementedError()
        self.kf_train = utils.generate_minibatch_idx(
            len(self.train), self.mb_size_train)
        self.kf_valid = utils.generate_minibatch_idx(
            len(self.valid), self.mb_size_test)
        self.kf_test = utils.generate_minibatch_idx(
            len(self.test), self.mb_size_test)
        
def prepare_data(engine, IDs):
    seqs = []
    feat_list = []
    def get_words(vidID, capID):
        caps = engine.CAP[vidID]
        rval = None
        for cap in caps:
            if str(cap['cap_id']) == capID:
                rval = cap['tokenized'].split(' ')
                rval = [w for w in rval if w != '']
                break
        assert rval is not None
        return rval
    
    for i, ID in enumerate(IDs):
        #print 'processed %d/%d caps'%(i,len(IDs))
        # load GNet feature
        vidID, capID = ID.split('_')
        feat = engine.get_video_features(vidID)
        feat_list.append(feat)
        words = get_words(vidID, capID)
        seqs.append([engine.worddict[w]
                     if w in engine.worddict and engine.worddict[w] < engine.n_words else 1 for w in words])

    lengths = [len(s) for s in seqs]
    if engine.maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        new_caps = []
        for l, s, y, c in zip(lengths, seqs, feat_list, IDs):
            # sequences that have length >= maxlen will be thrown away 
            if l < engine.maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
                new_caps.append(c)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs
        if len(lengths) < 1:
            return None, None, None, None
    
    y = numpy.asarray(feat_list)
    y_mask = engine.get_ctx_mask(y)
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.
    
    return x, x_mask, y, y_mask
    
def test_data_engine():
    from sklearn.cross_validation import KFold
    video_feature = 'googlenet' 
    out_of = None
    maxlen = 100
    mb_size_train = 64
    mb_size_test = 128
    maxlen = 50
    n_words = 30000 # 25770 
    signature = 'youtube2text' #'youtube2text'
    engine = Movie2Caption('attention', signature, video_feature,
                           mb_size_train, mb_size_test, maxlen,
                           n_words,
                           n_frames=26,
                           outof=out_of)
    i = 0
    t = time.time()
    for idx in engine.kf_train:
        t0 = time.time()
        i += 1
        ids = [engine.train[index] for index in idx]
        x, mask, ctx, ctx_mask = prepare_data(engine, ids)
        print 'seen %d minibatches, used time %.2f '%(i,time.time()-t0)
        if i == 10:
            break
            
    print 'used time %.2f'%(time.time()-t)
if __name__ == '__main__':
    test_data_engine()


