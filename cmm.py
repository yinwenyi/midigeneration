# Markov Model thingy
import random, sys, glob
import data, midi, experiments, patterns, chords
from decimal import Decimal as fixed
from decimal import ROUND_HALF_DOWN
from IPython import embed
import pickle
import copy

class Markov(object):

    '''
    Generic object for a Markov model

    trains state and state transitions by reading statechains
    statechain: a list of states
    state: a concrete class derived from the abstract class State

    '''
    START_TOKEN = 'start_token'
    STOP_TOKEN = 'stop_token'

    def __init__(self, chain_length=1):
        self.chain_length = chain_length
        self.markov = {}
        self.states = set()
        self.state_chains = [[]]

    def add(self, chain):
        '''
        add a statechain to the markov model (i.e. perform training)

        '''
        self.state_chains.append(chain)
        buf = [Markov.START_TOKEN] * self.chain_length
        for state in chain:
            v = self.markov.get(tuple(buf), [])
            v.append(state)
            self.markov[tuple(buf)] = v
            buf = buf[1:] + [state]
            self.states.add(state)
        v = self.markov.get(tuple(buf), [])
        v.append(Markov.STOP_TOKEN)
        self.markov[tuple(buf)] = v

    def generate(self, seed=[], max_len=float("inf")):
        '''
        generate a statechain from a (already trained) model
        seed is optional; if provided, will build statechain from seed
        len is optional; if provided, will only build statechain up to given len

        note: seed is untested and may very well not work...
        however, seed is core functionality that will help combine all_keys and segmentation (in the future)
        transitions from segments to segments are accomplished by using seed to look for a next valid segment
        that starts with the same seed as the previous segment ends with?
        '''
        buf = self.get_start_buffer(seed)
        state_chain = []
        count = 0

        # we might generate an empty statechain, count will stop us from infinite loop
        while not state_chain and count < 50:
            elem = self.generate_next_state(buf)
            while elem != Markov.STOP_TOKEN and len(state_chain) < max_len:
                state_chain.append(elem)
                buf = self.shift_buffer(buf, elem)
                elem = self.generate_next_state(buf) # generate another
            count += 1
        if not state_chain:
            print "Warning: state_chain empty; seed={}".format(seed)
        return state_chain

    def get_start_buffer(self, seed=[]):
        buf = [Markov.START_TOKEN] * self.chain_length
        if seed and len(seed) > self.chain_length:
            buf = seed[-self.chain_length:]
        elif seed:
            buf[-len(seed):] = seed
        return buf

    def shift_buffer(self, buf, elem):
        return buf[1:] + [elem] # shift buf, add elem to the end

    def generate_next_state(self, buf):
        elem = random.choice(self.markov[tuple(buf)]) # take a random next state using buf
        if elem != Markov.STOP_TOKEN:
            return elem.copy() # prevents change of the underlying states of the markov model
        else:
            return elem

    def copy(self):
        mm = Markov()
        # shallow copies (TODO: deep copy?)
        mm.chain_length = self.chain_length
        mm.markov = {k: v[:] for k, v in self.markov.iteritems()}
        mm.states = self.states.copy()
        mm.state_chains = [ chain[:] for chain in self.state_chains ]
        return mm

    def add_model(self, model):
        '''
        union of the states and state transitions of self and model
        returns a new markov model
        '''
        mm = self.copy()
        for chain in model.state_chains:
            mm.add(chain)
        return mm

class State(object):

    '''
    Basic interface of a state to be used in a Markov model
    Please override state_data() and copy()

    '''

    def state_data(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def __hash__(self):
        tup = self.state_data()
        return hash(tup)

    def __eq__(self, other):
        return self.state_data() == other.state_data()

    def __repr__(self):
        tup = self.state_data()
        return str(tup)

    def copy(self):
        raise NotImplementedError("Subclass must implement abstract method")


class SegmentState(State):
    '''
    SegmentState: a Markov state representing a segment of music (from segmentation)

    Instance attributes:
    - string label: name of the SegmentState, possibly arbitrary, for bookkeeping
    - Markov mm: a Markov model consisting of NoteStates. This will be used for generating the NoteStates
        within the segment

    '''

    def __init__(self, label, mm):
        self.label = label
        self.mm = mm

    def state_data(self):
        relevant = [self.label]
        return tuple(relevant)

    def copy(self):
        s = SegmentState(self.label, self.mm)
        return s

    @staticmethod
    def state_chain_to_note_states(state_chain):
        note_states = []
        for s in state_chain:
            gen = s.mm.generate()
            note_states.extend(gen)
        return note_states


class NoteState(State):
    '''
    NoteState: a Markov state representing a group of notes (all starting from the same position)

    Instance attributes:
    - notes: a list of Notes, all with the same position, sorted by duration then pitch
    - bar: number of ticks in a bar (this is used for converting positions to fixed/decimal values
    - bar_pos: a fixed/decimal value denoting the position of these notes relative to a bar

    (docs: todo)
    - state_position:
    - state_duration:
    - chord: the chord that these notes most likely belong to
    - origin: midi filename

    '''

    def __init__(self, notes, bar, chord='', origin=''):
        # State holds multiple notes, all with the same pos
        self.notes = [ n.copy() for n in sorted(notes, key=lambda x: (x.dur, x.pitch)) ]
        self.bar = bar
        self.bar_pos = fixed(self.notes[0].pos % bar) / bar
        self.state_position = fixed(self.notes[0].pos) / bar
        self.state_duration = 0 # set later
        self.chord = chord
        self.origin = origin

        for n in self.notes:
            n.dur = fixed(n.dur) / bar

    def state_data(self):
        ''' make hashable version of state information intended to be hashed '''
        notes_info = [n.pitch for n in self.notes]
        quantized_pos = self.bar_pos.quantize(fixed('0.01'), rounding=ROUND_HALF_DOWN)
        quantized_dur = self.state_duration.quantize(fixed('0.0001'), rounding=ROUND_HALF_DOWN)
        relevant = [quantized_pos, quantized_dur, self.chord, frozenset(notes_info)]
        return tuple(relevant)

    def copy(self):
        s = NoteState(self.notes, 1, self.chord, self.origin)
        s.bar = self.bar
        s.bar_pos = self.bar_pos
        s.state_position = self.state_position
        s.state_duration = self.state_duration
        return s

    def transpose(self, offset):
        s = self.copy()
        ctemp = self.chord.split('m')[0]
        s.chord = chords.translate(chords.untranslate(ctemp)+offset) + ('m' if 'm' in self.chord else '')
        s.origin = 'T({})'.format(offset) + s.origin
        for n in s.notes:
            n.pitch += offset
        return s

    def to_notes(self, bar, last_pos):
        '''
        Convert this NoteState to a list of notes,
        pos of each note will be assigned to last_pos
        bar is the number of ticks to form a bar
        returns the list of notes and the position of the next state (which can be used for the next call)
        '''

        notes = []
        for n in self.notes:
            nc = n.copy()
            nc.pos = last_pos
            nc.dur = int(n.dur * bar)
            notes.append(nc)
        last_pos += int(self.state_duration * bar)
        return notes, last_pos

    @staticmethod
    def state_chain_to_notes(state_chain, bar):
        '''
        Convert a state chain (a list of NoteStates) to notes
        arg bar: number of ticks to define a bar for midi files

        '''

        last_pos = 0
        notes = []
        for s in state_chain: # update note positions for each s in state_chain
            for n in s.notes:
                nc = n.copy()
                nc.pos = int(last_pos * bar)
                nc.dur = int(n.dur * bar)
                notes.append(nc)
            last_pos += s.state_duration
        return notes

    @staticmethod
    def notes_to_state_chain(notes, bar):
        '''
        Convert a list of Notes to a state chain (list of NoteStates)
        arg bar: number of ticks to define a bar for midi files

        '''

        # group notes into bins by their starting positions
        bin_by_pos = {}
        for n in notes:
            v = bin_by_pos.get(n.pos, [])
            v.append(n)
            bin_by_pos[n.pos] = v

        positions = sorted(bin_by_pos.keys())

        # produce a state_chain by converting the notes at every position x into a NoteState
        state_chain = map(lambda x: NoteState(bin_by_pos[x], bar), positions)

        if not len(state_chain):
            return state_chain

        # calculate state_duration for each state
        for i in range(len(state_chain) - 1):
            state_chain[i].state_duration = state_chain[i+1].state_position - state_chain[i].state_position
        state_chain[-1].state_duration = max(n.dur for n in state_chain[-1].notes) # the last state needs special care

        return state_chain

    @staticmethod
    def piece_to_state_chain(piece, use_chords=True):
        '''
        Convert a data.piece into a state chain (list of NoteStates)
        arg use_chords: if True, NoteState holds chord label as state information

        '''
        # TODO: shouldn't have to hardcode this
        use_chords = True
        key_sig = ""

        # group notes into bins by their starting positions
        bin_by_pos = {}
        for n in piece.unified_track.notes:
            v = bin_by_pos.get(n.pos, [])
            v.append(n)
            bin_by_pos[n.pos] = v

        positions = sorted(bin_by_pos.keys())
        if use_chords:
            cc = chords.fetch_classifier()
            key_sig, allbars = cc.predict(piece) # assign chord label for each bar
            state_chain = map(lambda x: NoteState(bin_by_pos[x], piece.bar, chord=allbars[x/piece.bar], origin=piece.filename), positions)
        else:
            state_chain = map(lambda x: NoteState(bin_by_pos[x], piece.bar, chord='', origin=piece.filename), positions)

        if not len(state_chain):
            return state_chain

        # calculate state_duration for each state
        for i in range(len(state_chain) - 1):
            state_chain[i].state_duration = state_chain[i+1].state_position - state_chain[i].state_position
        state_chain[-1].state_duration = max(n.dur for n in state_chain[-1].notes) # the last state needs special care

        return key_sig, state_chain

    def __repr__(self):
        tup = self.state_data()
        return str(tup) + ' ' + str(self.notes)

def get_key_offset(key1, key2):
    '''
    Returns the relative pitch offset between two keys.
    :param key1: The original key
    :param key2: The new, desired key
    :return: int
    '''
    key1 = str(key1)
    key2 = str(key2)

    # map all minor keys to major keys, and also all redundant major keys to major keys
    key_dict = {'Am': 'C',          # begin minor keys
                'Em': 'G',
                'Bm': 'D',
                'F#/Gbm': 'A',
                'C#/Dbm': 'E',
                'G#/Abm': 'B',
                'D#/Ebm': 'F#',
                'A#/Bbm': 'C#',
                'Dm': 'F',
                'Gm': 'Bb',
                'Cm': 'Eb',
                'Fm': 'Ab',         # end minor keys
                'C#/Db': 'C#',      # begin redundant major keys
                'F#/Gb': 'F#',
                'G#/Ab': 'Ab',
                'D#/Eb': 'Eb',
                'A#/Bb': 'Bb'}      # end redundant major keys

    # keep a dict of the pitch deltas (a delta of 1 corresponds to moving up a half note on the keyboard)
    # transposing from the first key to the second, ie ('C','G') means transposing from C major to G major
    pitch_delta = {('C', 'G'): -5, ('C', 'D'): 2, ('C', 'A'): -3, ('C', 'E'): 4, ('C', 'B'): -1, ('C', 'F#'): 6,
                   ('C', 'C#'): 1, ('C', 'F'): 5, ('C', 'Bb'): -2, ('C', 'Eb'): 3, ('C', 'Ab'): -4,
                   ('G', 'D'): -5, ('G', 'A'): 2, ('G', 'E'): -3, ('G', 'B'): 4, ('G', 'F#'): -1, ('G', 'C#'): 6,
                   ('G', 'F'): -2, ('G', 'Bb'): 3, ('G', 'Eb'): -4, ('G', 'Ab'): 1,
                   ('D', 'A'): -5, ('D', 'E'): 2, ('D', 'B'): -3, ('D', 'F#'): 4, ('D', 'C#'): -1, ('D', 'F'): 3,
                   ('D', 'Bb'): -4, ('D', 'Eb'): 1, ('D', 'Ab'): 6,
                   ('A', 'E'): -5, ('A', 'B'): 2, ('A', 'F#'): -3, ('A', 'C#'): 4, ('A', 'F'): -4, ('A', 'Bb'): 1,
                   ('A', 'Eb'): 6, ('A', 'Ab'): -1,
                   ('E', 'B'): -5, ('E', 'F#'): 2, ('E', 'C#'): -3, ('E', 'F'): 1, ('E', 'Bb'): 6, ('E', 'Eb'): -1,
                   ('E', 'Ab'): 4,
                   ('B', 'F#'): -5, ('B', 'C#'): 2, ('B', 'F'): 6, ('B', 'Bb'): -1, ('B', 'Eb'): 4, ('B', 'Ab'): -3,
                   ('F#', 'C#'): -5, ('F#', 'F'): -1, ('F#', 'Bb'): 4, ('F#', 'Eb'): -3, ('F#', 'Ab'): 2,
                   ('C#', 'F'): 4, ('C#', 'Bb'): -3, ('C#', 'Eb'): 2, ('C#', 'Ab'): -5,
                   ('F', 'Bb'): 5, ('F', 'Eb'): -2, ('F', 'Ab'): 3,
                   ('Bb', 'Eb'): 5, ('Bb', 'Ab'): -2,
                   ('Eb', 'Ab'): 5}

    # remap the keys if needed
    if key1 in key_dict.keys():
        orig_key_major = key_dict[key1]
    else:
        orig_key_major = key1
    if key2 in key_dict.keys():
        new_key_major = key_dict[key2]
    else:
        new_key_major = key2

    if orig_key_major == new_key_major:
        return 0

    if (orig_key_major, new_key_major) in pitch_delta.keys():
        delta = pitch_delta[(orig_key_major, new_key_major)]
    else:   # reverse order is still the same offset, just in the opposite direction
        delta = -1 * pitch_delta[(new_key_major, orig_key_major)]

    return delta

def piece_to_markov_model(musicpiece, classifier=None, segmentation=False, all_keys=False):
    '''
    Train a markov model on a music piece

    Note: (important!)
    If segmentation is True, train a markov model of SegmentStates, each holding a Markov consisting of NoteStates
    Otherwise, the Markov model will consist of NoteStates

    '''

    mm = Markov()
    print "all_keys:" + str(all_keys)
    if not segmentation:
        key_sig, state_chain = NoteState.piece_to_state_chain(musicpiece, True) # always use chords
        offset = get_key_offset(key_sig[0], 'C')   # transpose everything to C major
        shifted_state_chain = [s.transpose(offset) for s in state_chain]
        mm.add(shifted_state_chain)

        if all_keys: # shift piece up some number of tones, and down some number of tones
            for i in range(1, 6):
                shifted_state_chain = [ s.transpose(i) for s in state_chain ]
                mm.add(shifted_state_chain)
            for i in range(1, 7):
                shifted_state_chain = [ s.transpose(-i) for s in state_chain ]
                mm.add(shifted_state_chain)
    else:
        if classifier == None:
            raise Exception("classifier cannot be None when calling piece_to_markov_model with segmentation=True")
        segmented = experiments.analysis(musicpiece, classifier)
        chosenscore, chosen, labelled_sections = segmented.chosenscore, segmented.chosen, segmented.labelled_sections

        # state_chain implementation #2: more correct than #1, at least
        state_chain = []
        labelled_states = {}
        for ch in chosen:
            i, k = ch[0], ch[1]
            label = labelled_sections[ch]
            ss = labelled_states.get(label, None)
            segment = musicpiece.segment_by_bars(i, i+k)
            if not ss:
                ss = SegmentState(label, piece_to_markov_model(segment, classifier, segmentation=False, all_keys=all_keys))
                labelled_states[label] = ss
            else:
                # ss.mm holds the mm that generates notes
                _state_chain = NoteState.notes_to_state_chain(segment.unified_track.notes, segment.bar)
                ss.mm.add(_state_chain)
            state_chain.append(ss)

        print 'Original Sections: ({})'.format(musicpiece.filename)
        print [ g.label for g in state_chain ]
        print chosenscore
        mm.add(state_chain)
    return mm

def test_variability(mm, meta, bar):
    '''
    Generate song 10 times from a trained markov model  and print out their lengths
    if they are all the same lengths, chances are the pieces are all the same

    '''
    lens = []
    for i in range(10):
        song, gen, a = generate_song(mm, meta, bar, True)
        lens.append(len(a))
    print lens

def generate_song(mm, meta, bar, segmentation=False):

    '''
    Generate music, i.e. a list of MIDI tracks, from the given Markov mm
    you would also need to provide a list of meta events (which you can pull from any MIDI file)

    mm: the Markov model
    meta: list of meta events, is an attribute of class piece
    bar: ticks per bar
    '''

    song = []
    song.append(meta)

    if not segmentation:
        gen = mm.generate()
        print [g.origin + ('-' if g.chord else '') + g.chord for g in gen]
    else:
        # if segmentation, mm is a markov model of SegmentStates
        # generate SegmentStates from mm and then generate NoteStates from each

        gen_seg = mm.generate()
        print 'Rearranged Sections:'
        print [ g.label for g in gen_seg ]
        gen = SegmentState.state_chain_to_note_states(gen_seg)

    a = NoteState.state_chain_to_notes(gen, bar)
    if not a: return generate_song(mm, meta, bar, segmentation)
    song.append([ n.note_event() for n in a ])

    return song, gen, a

def generate_output():
    c = patterns.fetch_classifier()
    segmentation = False
    all_keys = False

    if len(sys.argv) == 4: # <midi-file> <start-bar> <end-bar>
        musicpiece = data.piece(sys.argv[1])
        musicpiece = musicpiece.segment_by_bars(int(sys.argv[2]), int(sys.argv[3]))
        mm = piece_to_markov_model(musicpiece, c, segmentation)
        song, gen, a = generate_song(mm, musicpiece.meta, musicpiece.bar, segmentation)

    else:
        dir = "./mid/Bach/*"
        pieces = glob.glob(dir)
        #pieces = ["mid/Bach/bwv804.mid", "mid/Bach/bwv802.mid"]

        mm = Markov()

        # generate a model _mm for each piece then add them together
        p = pieces.pop(0)
        musicpiece = data.piece(p)
        _mm = piece_to_markov_model(musicpiece, c, segmentation, all_keys) # no transpose
        mm = mm.add_model(_mm)

        for p in pieces:
            musicpiece = data.piece(p)
            _mm = piece_to_markov_model(musicpiece, c, segmentation, all_keys)
            mm = mm.add_model(_mm)

        song, gen, a = generate_song(mm, musicpiece.meta, musicpiece.bar, segmentation)

    midi.write('output.mid', song)
    return

def generate_score(file):
    '''(str) -> NoneType
    Generate the score (.xml) for the auto-generated midi file.
    Please open using a MusicXMl reader such as Finale NotePad.
    '''
    import music21   # required to display score
    midi_file_output = music21.converter.parse(file)
    midi_file_output.show()
    return

if __name__ == '__main__':
    generate_output()
    #generate_score()
