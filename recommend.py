# Functions needed for recommender tool
import data, midi, patterns, cmm
import os, cPickle, glob
from decimal import Decimal as fixed
from decimal import ROUND_HALF_DOWN
import types, random

class Recommender(object):
    '''
    A Recommender object has the following attributes:
    self.type       - the genre
    self.pieces     - the training pieces it has seen
    self.mm         - markov model trained on self.pieces
    self.rev_mm     - markov model trained on reversed self.pieces
    '''

    def __init__(self, type, pieces):
        '''
        type is a str describing the category that the training pieces are in
        pieces is a set of the pieces in that category
        '''
        self.type = type
        self.pieces = set(pieces)

        mm = cmm.Markov()
        reverse_mm = cmm.Markov()

        c = patterns.fetch_classifier()
        segmentation = False
        all_keys = False

        for p in self.pieces:
            print(p)
            musicpiece = data.piece(p)
            # segmentation off by default, all_keys off by default
            # this should automatically transpose everything to C major
            _mm = cmm.piece_to_markov_model(musicpiece, c, segmentation, all_keys)
            mm = mm.add_model(_mm)

            # now reverse the state chains to get the reverse mm
            b = _mm.state_chains
            rev_chains = [ chain[::-1] for chain in _mm.state_chains ]
            _mm.state_chains = rev_chains
            reverse_mm = reverse_mm.add_model(_mm)

        self.mm = mm
        self.rev_mm = reverse_mm

    def save(self):
        '''
        Pickles the Markov model.
        :return: None
        '''
        # hash the names of the pieces contained in the model
        hashnum = hash(frozenset(self.pieces))
        name = 'cached/rec/rec-{}-{}.pkl'.format(self.type, str(hashnum))
        if os.path.isfile(name):
            print("This Recommender object already exists, not saving.")
        else:
            with open(name, "wb") as fh:
                cPickle.dump(self, fh)

    def add_piece(self, piece):
        '''
        Add a new piece to the model.
        :param piece: location of midi
        :return: None
        '''
        if piece in self.pieces:
            print piece + " already in model."
            return

        c = patterns.fetch_classifier()
        segmentation = False
        all_keys = True

        musicpiece = data.piece(piece)
        _mm = cmm.piece_to_markov_model(musicpiece, c, segmentation, all_keys)
        self.mm = self.mm.add_model(_mm)

        b = _mm.state_chains
        rev_chains = [chain[::-1] for chain in _mm.state_chains]
        _mm.state_chains = rev_chains
        self.rev_mm = self.rev_mm.add_model(_mm)

    def find_best_seed(self, seed, rev=False):
        '''
        Find the existing state which is most similar to the given seed
        :param seed: start state for the chain
        :param rev: use the reverse markov model
        :return:
        '''
        # seed is almost guaranteed to not be in the markov dict, so find the most similar key and use that as the seed
        model = self.mm
        if rev:
            model = self.rev_mm

        if len(model.markov.get(seed, [])) == 0:
            # Find states in the same key and starting position
            same_key = set()
            same_start = set()
            for key in model.markov.keys():
                ns = key[0]
                if type(ns) == types.StringType: continue   # start and stop tokens
                if ns.chord == seed.chord:
                    same_key.add(ns)
                    if ns.bar_pos.quantize(fixed('0.01'), ROUND_HALF_DOWN) == seed.bar_pos.quantize(fixed('0.01'), ROUND_HALF_DOWN):
                        same_start.add(ns)
            # If there are no states in the same key, try something else
            # For now don't care about this
            best_keys = same_start
            if len(best_keys) == 0:
                best_keys = same_key

            # The best seed will have a similar tempo and number of notes, and also the chord should not
            # be too far away in frequency
            max_dist = float('inf')
            best = None
            for key in best_keys:
                dist = distance(seed, key)
                if dist < max_dist:
                    best = key
                    max_dist = dist

            seed = best
        else:
            seed = model.markov.get(seed)

        return seed

    def concurrent_generate(self, start, end, length=50):
        '''
        Does mm.generate forwards and backwards concurrently, stopping when the same state is reached
        by both, or when the two states are similar enough.
        :param start: list of NoteStates
        :param end: list of NoteStates
        :param length: int
        :return: state chain
        '''
        fwd_chain = [state for state in start]
        fwd_buf = self.mm.get_start_buffer(start)

        rev_chain = [state for state in end]
        rev_buf = self.rev_mm.get_start_buffer(end)

        count = 0
        score = 5
        chain = []
        while count < length/2:
            next_fwd = self.mm.generate_next_state(fwd_buf)
            next_rev = self.rev_mm.generate_next_state(rev_buf)
            if next_fwd is next_rev:
                fwd_chain.append(next_fwd)
                fwd_chain.extend(rev_chain[::-1])
                return fwd_chain
            elif isinstance(next_fwd, basestring) or isinstance(next_rev, basestring):
                return chain
            elif distance(next_fwd, next_rev) < score:
                chain = []
                chain.extend(fwd_chain)
                chain.append(next_fwd)
                chain.extend(rev_chain[::-1])
                score = distance(next_fwd, next_rev)

            fwd_chain.append(next_fwd)
            rev_chain.append(next_rev)
            fwd_buf = self.mm.shift_buffer(fwd_buf, next_fwd)
            rev_buf = self.rev_mm.shift_buffer(rev_buf, next_rev)

        return chain

    def recommend(self, seed=[], length=8, end=[]):
        '''
        Recommend continuation of seed in the style of this category
        :param seed: the starter piece for the recommender, represented as a state chain
        :param end: optional, the state to end on (use if bridging two pieces is desired)
        :param length: the maximum state chain length of the recommended piece
        :return: a short state chain representing the recommendation
        '''

        # 1. preceding mode
        if not len(seed):
            if not len(end):
                print("Can't run in preceding mode without end state!")
                return 0
            seed = self.find_best_seed(end[0], rev=True)
            rec = self.rev_mm.generate([seed], length)
        else:
            # 2. bridging mode in progress
            if len(end):
                seed = self.find_best_seed(seed[0])
                end = self.find_best_seed(end[0], rev=True)
                rec = self.concurrent_generate([seed], [end])
            else:
                # 3. normal mode (not bridge or preceding)
                seed = self.find_best_seed(seed[0])
                rec = self.mm.generate([seed], length)
        return rec


def distance(state, other):
    '''
    Generates a similarity measure between NoteStates.
    :param state:
    :param other:
    :return:
    '''
    # compare num of notes
    dist = abs(len(state.notes) - len(other.notes))
    # compare state duration
    dist += abs(state.state_duration - other.state_duration) / state.state_duration
    # compare pitch delta
    dist += (sum([n.pitch for n in state.notes]) / len(state.notes)) - \
            (sum([n.pitch for n in other.notes]) / len(other.notes))
    return dist

def recommend(piece1, style, training, typ, num_recs=4, piece2=None):
    '''
    The handler for an API call
    '''

    # check if this already exists
    name = "cached/rec/rec-{}-{}.pkl".format(style, hash(frozenset(training)))
    if os.path.isfile(name):
        with open(name, "rb") as fh:
            rec = cPickle.load(fh)
    else:
        rec = Recommender(style, training)
        rec.save()

    # get the incomplete piece
    piece1 = data.piece(piece1)
    # label the piece by chords, determine the length of the seed bars
    use_chords = True
    key_sig, unshifted_state_chain = cmm.NoteState.piece_to_state_chain(piece1, use_chords)
    offset = cmm.get_key_offset(key_sig[0], 'C')
    state_chain1 = [s.transpose(offset) for s in unshifted_state_chain]

    if piece2 is not None:
        piece2 = data.piece(piece2)
        key_sig, unshifted_state_chain = cmm.NoteState.piece_to_state_chain(piece2, use_chords)
        offset = cmm.get_key_offset(key_sig[0], 'C')
        state_chain2 = [s.transpose(offset) for s in unshifted_state_chain]

    # modes: preceding, bridging, and following
    if typ is 'pre':
        seed = []
        end = [state_chain1[0]]
    elif piece2 is not None and typ is 'bridge':
        seed = [state_chain1[-1]]
        end = [state_chain2[0]]
    elif typ is 'post':
        seed = [state_chain1[-1]]
        end = []
    else:   # this shouldn't happen
        print "Error: Second piece not given"
        return 0

    # generate new states by providing the seed bars
    # do this several times and see if we get a different result
    results = []
    for i in range(num_recs):
        res = rec.recommend(seed, 100, end)
        print [g.origin + ('-' if g.chord else '') + g.chord for g in res]
        if res not in results:
            results.append(res)

    # write out the 'best' result as a midi piece (for now, just pick the first one)
    result = results[0]

    if typ is 'pre':
        result.extend(state_chain1)
        music = cmm.NoteState.state_chain_to_notes(result, piece1.bar)
    elif typ is 'post':
        state_chain1.extend(result)
        music = cmm.NoteState.state_chain_to_notes(state_chain1, piece1.bar)
    else:
        state_chain1.extend(result)
        state_chain1.extend(state_chain2)
        music = cmm.NoteState.state_chain_to_notes(state_chain1, piece1.bar)

    song = [piece1.meta]
    song.append([n.note_event() for n in music])

    midi.write('rec.mid', song)
    #cmm.generate_score('rec.mid')

if __name__ == '__main__':

    # sample settings

    # get the genre (aka what pieces to recommend from)
    # train a markov model on the 'genre'
    dir = "./mid/Enya_*"
    training = glob.glob(dir)
    style = "enya"

    # can be pre, bridge, or post
    typ = "pre"
    num_recs = 4

    recommend("mid/twinkle_twinkle.mid", style, training, typ, num_recs, piece2=None)