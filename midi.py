from more_itertools import unique_everseen
help={}
help["song"]="List of tracks. Track 1 contains events that apply to all tracks, the other tracks contain individual events."
help["track"]="List of events."
help["event"]="An event has the form [description, position, values...]. The possible descriptions are \"ticks\", \"tempo\", \"time\", \"key\", \"note\", and \"bar\"."
help["ticks"]="This type of event has the form [\"ticks\", position, number of ticks per quarter note]."
help["tempo"]="This type of event has the form [\"tempo\", position, number of microseconds per quarter note]."
help["time"]="This type of event has the form [\"time\", position, top of time signature, bottom]."
help["key"]="This type of event has the form [\"key\", position, number of sharps in key signature, major(0) or minor(1)]."
help["note"]="This type of event has the form [\"note\", position, duration, channel, note number]."
help["bar"]="This type of event has the form [\"bar\", position]."
help["transpose"]="transpose(song, amount) transposes song in place up by number of semitones specified in amount."
help["bar"]="bar(song) adds bars to song in place. Note positions are recalculated relative to the last bar."
help["unbar"]="unbar(song) removes bars from song in place."
help["playing"]="playing(song, time) returns a list, each element is a list of notes playing, or [\"rest\"] if none are playing. The element number in the higher list corresponds to the track number (ignoring track 1)."
help["harmony"]="harmony(song) returns a list of harmonies. Each harmony is a list of what note is playing in each track (ignoring track 1) plus the frequency of the harmony."
help["write"]="write(filename, song)"

trackheaderlength=8

#Return [delta, length] where
#delta is the delta ticks encoded starting from bytes[0] and
#length is the length of the delta time in bytes.
#Return -1 in case of error.
def getdelta(bytes):
    result=0
    for i in range(4):
        result<<=7
        result+=ord(bytes[i])&0x7f
        if not ord(bytes[i])&0x80:#The most significant bit signals when to stop.
            break
        elif i==3:#Delta time should not be more than 4 bytes long.
            raise Exception("Delta time should not be more than 4 bytes long.")
            return -1
    return [result,i+1]

#Return the bytes that specify a delta time equal to ticks.
def delta(ticks):
    mask=0x7f
    result=[]
    while True:
        byte=ticks&mask
        ticks=ticks>>7
        result=[byte]+result
        if len(result)>4:
                raise Exception("Delta time should not be more than 4 bytes long.")
                return -1
        if ticks==0:
            for i in range(len(result)-1):
                result[i]|=0x80
            for i in range(len(result)):
                result[i]=chr(result[i])
            return result

#Return the midi commands in trackchunk in a list.
#trackchunk is assumed to have come from a track produced by chunkitize.
#Return -1 in case the midi file contains unexpected data
def getcommands(trackchunk):
    commands=[]
    global trackheaderlength
    i=trackheaderlength
    # handle running status data by storing the last status byte seen so far
    running_status = 0
    while i<len(trackchunk):
        commands+=[[]]
        result=getdelta(trackchunk[i:])
        if result==-1:
            return -1
        commands[-1]+=[result[0]]
        i+=result[1]
        if ord(trackchunk[i])&0xf0 in [0x80,0x90,0xa0,0xb0,0xe0]:
            running_status = trackchunk[i]
            commands[-1]+=[trackchunk[i:i+3]]
            i+=3
        elif ord(trackchunk[i])&0xf0 in [0xc0,0xd0]:
            running_status = trackchunk[i]
            commands[-1]+=[trackchunk[i:i+2]]
            i+=2
        elif ord(trackchunk[i])&0xf0==0xf0:
            # system messages 0xf0 to 0xf7 clear the running status
            if running_status and ord(trackchunk[i]) < 0xf8:
                running_status = 0
            # 0xff denotes a meta event:
            # 0xff + <meta_type> [1 byte] + <v_length> [variable len] + <event_data_bytes> [v_length]
            if ord(trackchunk[i])==0xff:
                result=getdelta(trackchunk[i+2:])
                if result==-1:
                    return -1
                # i + 1 byte for <meta_type> + len of <v_length> + v_length + 1 because of slice defn
                commands[-1]+=[trackchunk[i:i+2+result[1]+result[0]]]
                i+=2+result[1]+result[0]
            # all system-exclusive events begin with either F0 or F7, and end with F7
            # 0xf0 + <length> + <bytes to be transmitted after F0>
            # 0xf7 + <length> + <all bytes to be transmitted>
            # <length> is a variable-length quantity, up to 4 bytes
            elif ord(trackchunk[i]) in [0xf0, 0xf7]:
                # result is a 2 entry array where result[0] is <length> and
                # result[1] is the length of <length> in bytes (from 1 to 4)
                result=getdelta(trackchunk[i+1:])
                if result==-1:
                    return -1
                # do we even care about system-exclusive events?
                commands[-1]+=[trackchunk[i:i+1+result[0]+result[1]]]
                i+=1+result[0]+result[1]
            # song position pointer message has two following data bytes
            elif ord(trackchunk[i])==0xf2:
                commands[-1]+=[trackchunk[i:i+3]]
                i+=3
            # song select message has one following data byte
            elif ord(trackchunk[i])==0xf3:
                commands[-1]+=[trackchunk[i:i+2]]
                i+=2
            # all other messages have no associated data bytes
            else:
                commands[-1]+=[trackchunk[i]]
                i+=1
        # check if we are in running status mode. If so, use the last seen
        # status byte for the received data bytes
        elif running_status:
            if ord(running_status)&0xf0 in [0x80, 0x90, 0xa0, 0xb0, 0xe0]:
                commands[-1]+=[running_status+trackchunk[i:i+2]]
                i+=2
            elif ord(running_status)&0xf0 in [0xc0, 0xd0]:
                commands[-1]+=[running_status+trackchunk[i:i+1]]
                i+=1
            else: # this is technically impossible
                raise Exception("error")
        # received a data byte when not in running status mode; exit
        else:
            raise Exception("error")
    if len(commands)<1:
        raise Exception("error")
    if ord(commands[-1][1][1])!=0x2f:
        raise Exception("error")
    return commands

#return an unsigned integer from a big endian string of bytes
def uintb(bytes):
    result=0
    for byte in bytes:
        result<<=8
        result+=ord(byte)
    return result

#return a list of lists of of midi file bytes, separated into the header and track chunks
#return -1 if the midi file doesn't look as expected
def chunkitize(bytes):
    # the header chunk is always 14 bytes long
    headerlength=14
    headertitle="MThd"
    if len(bytes)<headerlength:
        raise Exception("error")
        return -1
    # the header chunk should always be first in the bytestring
    if bytes[0:len(headertitle)]!=headertitle:
        raise Exception("error")
        return -1
    # add the header chunk to our list
    chunks=[bytes[0:headerlength]]
    # keep track of where we are in the bytestring. Currently, we are just past the header chunk
    i=headerlength
    tracktitle="MTrk"
    global trackheaderlength
    while True:
        if len(bytes)<i+trackheaderlength:
            break
        if bytes[i:i+len(tracktitle)]!=tracktitle:
            raise Exception("error")
            return -1
        # a track chunk = 'MTrk' + <length> + <track_event> [+ <track_event> ...]
        # get <length>, a 4-byte value, which is the length of the track chunk
        tracklength=uintb(bytes[i+4:i+8])
        if len(bytes)<i+trackheaderlength+tracklength:
            raise Exception("error")
            return -1
        # add the entire track chunk to our list
        chunks+=[bytes[i:i+trackheaderlength+tracklength]]
        # move our tracking pointer down the bytestring, continue reading until there are no more track chunks
        i+=trackheaderlength+tracklength
    if i!=len(bytes):
        raise Exception("error")
        return -1
    if uintb(bytes[10:12])!=len(chunks)-1:
        raise Exception("error")
        return -1
    return chunks

#Take the number of sharps in the key signature (negative value for flats)
#and return the semitone number of the root, C being 0.
def sharpstosemitones(sharps, minor):
    return (sharps*7-minor*3)%12

# Uniquify a list of notes and preserve the order
def uniquify_notes(seq):
   seen = {}
   result = []
   for item in seq:
       tup = tuple(item) #lists are not hashable
       if tup in seen: continue
       seen[tup] = 1
       result.append(item)
   return result

#Parse a midi bytes and return a song
#Return -1 in case of error
def parse(bytes):
    chunks=chunkitize(bytes)
    if chunks==-1:
        raise Exception("error")
    ticksperquarter=uintb(chunks[0][12:14])
    if ticksperquarter==0:
        raise Exception("error")
    # support single and multi-track format
    if uintb(chunks[0][8:10]) in [0,1]:
        song=[]
        if len(chunks)<2:
            return song
        commands=getcommands(chunks[1])
        ticks=0
        track=[["ticks", ticks, ticksperquarter]]
        for command in commands:
            ticks+=command[0]
            if ord(command[1][0])&0xf0==0xf0:
                if ord(command[1][0])==0xff:
                    if ord(command[1][1])==0x51:#Tempo
                        track+=[["tempo", ticks, uintb(command[1][3:6])]]
                    elif ord(command[1][1])==0x58:#Time signature
                        track+=[["time", ticks, ord(command[1][3]), 1<<ord(command[1][4])]]
                    elif ord(command[1][1])==0x59:#Key signature
                        sharps=ord(command[1][3])
                        if sharps&0x80:
                            sharps=(sharps&0x7f)-0x80
                        track+=[["key", ticks, sharps, ord(command[1][4])]]
        song+=[track]
        # assume the format is multi-track
        val_start = 2
        val_stop = len(chunks)
        # in case of single track, readjust for-loop values
        if uintb(chunks[0][8:10])==0:
            val_start = 1
        # extract notes
        for i in range(val_start,val_stop):
            ticks=0
            if i > 1: # don't re-fetch single track commands
                commands=getcommands(chunks[i])
            track=[]
            for j in range(len(commands)):
                command=commands[j]
                ticks+=command[0]
                if ord(command[1][0])&0xf0==0x90 and ord(command[1][2])!=0:#Note on
                    duration=0
                    for offcommand in commands[j+1:]:
                        duration+=offcommand[0]
                        if ord(offcommand[1][0])&0xf0==0x90 and ord(offcommand[1][2])==0 or ord(offcommand[1][0])&0xf0==0x80:#Note off
                            if offcommand[1][1]==command[1][1]:
                                break
                    track+=[["note", ticks, duration, ord(command[1][0])&0x0f, ord(command[1][1])]]
            # uniquify track list
            track=uniquify_notes(track)
            song+=[track]
        return song
    # no support for multi-song format
    raise Exception("error")

#Read a midi file and return a nicely constructed list that represents the song in the midi.
#Return -1 in case of error
#The list is structured as follows:
#It is a list of tracks.
#The first track specifies things that correspond to all tracks.
#The other tracks (if they exist) specify things specific to themselves.
#Each track is a list of events.
#An event has the form [description, position, values...]
#description is a string describing the type of event.
#position is the position of the event in the song in ticks.
#values is a bunch of values based on which kind of event it is.
#Here's a list of description and their corresponding values
#"ticks": Value is number of ticks per quarter note.
#"tempo": Value is number of microseconds per quarter note.
#"time": Values are top of time signature, bottom of time signature.
#"key": Values are ticks, number of sharps (negative means flats), major (0) or minor (1)
#"note": Values are ticks, duration, channel, note number
def read(filename):
    file=open(filename,"rb")#Open in binary read mode.
    bytes=file.read()
    file.close()
    return parse(bytes)

#Write a midi track to a file based on a list of bytes.
#The track header and end message are appended automatically, so they should not be included in bytes.
def writetrack(file, bytes):
    bytes=["\0","\xff","\x01","\0"]+bytes+["\x01","\xff","\x2f","\0"]#Put a 1 here to match Sibelius 2.
    size0=len(bytes)%0x100
    size1=(len(bytes)>>8)%0x100
    size2=(len(bytes)>>16)%0x100
    size3=(len(bytes)>>24)
    trackheader=["M","T","r","k",chr(size3),chr(size2),chr(size1),chr(size0)]
    bytes=trackheader+bytes
    for byte in bytes:
        file.write(byte)
    return

def comparetime(message1, message2):
    if message1[1]<message2[1]:
        #raise Exception("error")
        return -1
    elif message1[1]==message2[1]:
        return 0
    else:
        return 1

def ilog2(x):
    result=-1
    while x!=0:
        x>>=1
        result+=1
    return result

#Write a midi file based on a nicely constructed list.
def write(filename, song):
    file=open(filename,"wb")
    ticks=0
    for event in song[0]:
        if event[0]=="ticks":
            ticks=event[2]
    if ticks==0:
        raise Exception("error")
        return -1
    tracks=len(song)
    header=["M","T","h","d","\0","\0","\0","\6","\0","\1","\0",chr(tracks),chr(ticks>>8),chr(ticks&0xff)]
    for byte in header:
        file.write(byte)
    bytes=[]
    lasttime=0
    for event in song[0]:
        if event[0]=="tempo":
            bytes+=delta(event[1]-lasttime)
            bytes+=["\xff","\x51","\x03"]
            bytes+=[chr(event[2]>>16),chr((event[2]>>8)&0xff),chr(event[2]&0xff)]
        elif event[0]=="time":
            bytes+=delta(event[1]-lasttime)
            bytes+=["\xff","\x58","\x04"]
            bytes+=[chr(event[2]),chr(ilog2(event[3])),chr(24),chr(8)]
        elif event[0]=="key":
            bytes+=delta(event[1]-lasttime)
            bytes+=["\xff","\x59","\x02"]
            sharps=event[2]
            if sharps<0:
                sharps=0x100+sharps
            bytes+=[chr(sharps),chr(event[3])]
        lasttime=event[1]
    writetrack(file, bytes)
    for track in song[1:]:
        messages=[]
        for event in track:
            if event[0]=="note":
                messages+=[["on",event[1],event[3],event[4]]]
                messages+=[["off",event[1]+event[2],event[3],event[4]]]
        messages.sort(comparetime)
        lasttime=0
        for i in range(len(messages)):
            temp=messages[i][1]
            messages[i][1]=messages[i][1]-lasttime
            lasttime=temp
        bytes=[]
        for message in messages:
            if message[0]=="on":
                bytes+=delta(message[1])
                bytes+=[chr(0x90|message[2])]
                bytes+=[chr(message[3])]
                bytes+=["\x79"]
            elif message[0]=="off":
                bytes+=delta(message[1])
                bytes+=[chr(0x80|message[2])]
                bytes+=[chr(message[3])]
                bytes+=["\0"]
        writetrack(file, bytes)
    file.close()
    return

def transpose(song, amount):
    for i in range(len(song)):
        for j in range(len(song[i])):
            if song[i][j][0]=="note":
                song[i][j][4]+=amount
            elif song[i][j][0]=="key":
                song[i][j][2]+=amount
    return

def bar(song):
    for event in song[0]:
        if event[0]=="ticks":
            ticks=event[2]
        elif event[0]=="time": #TODO: make this better -- time sig could change
            top=event[2]
            bottom=event[3]
    bar=4*ticks*top/bottom
    for i in range(len(song)):
        offset=bar
        j=0
        while j<len(song[i]):
            while song[i][j][1]>=offset:
                song[i].insert(j,["bar",offset])
                offset+=bar
                j+=1
            song[i][j][1]%=bar
            j+=1
    return

def unbar(song):
    for i in range(len(song)):
        bar=0
        j=0
        while j<len(song[i]):
            if song[i][j][0]=="bar":
                bar=song[i][j][1]
                del song[i][j]
            else:
                song[i][j][1]+=bar
                j+=1
    return

def playing(song, time):
    result=[]
    for track in song[1:]:
        result+=[["rest"]]
        for event in track:
            if event[0]=="note":
                if event[1]+event[2]>time:
                    if event[1]<=time:
                        if result[-1]==["rest"]:
                            result[-1]=[event[4]]
                        else:
                            result[-1]+=[event[4]]
            if event[1]>time:
                break
    return result

def harmony(song):
    tracks=song[1:]
    i=[0]*len(tracks)
    result=[]
    while True:
        #find minimum time, doesn't have to be unique
        ind=len(i)#poison value
        min=-1
        dur=0
        for j in range(len(i)):
            if i[j]<len(tracks[j]) and (tracks[j][i[j]][1]<min or min==-1):
                min=tracks[j][i[j]][1]
                dur=tracks[j][i[j]][2]
                ind=j
        if min==-1:
            print "no minimum time found."
        #find pairs of concurrent notes
        import rel
        temp=rel.cproduct(playing(song, tracks[ind][i[ind]][1]))
        #Add to result
        for x in temp:
            if x in [y[0:2] for y in result]:
                result[[y[0:2] for y in result].index(x)][2]+=1
            else:
                result+=[x+[1]]
        #increment the track we just looked at so we don't count it again
        i[ind]+=1
        #see if we need to quit
        quit=True
        for j in range(len(i)):
            if i[j]<len(tracks[j]):
                quit=False
        if quit:
            break
    return result
