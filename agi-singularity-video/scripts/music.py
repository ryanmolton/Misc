import mido
TPB=480  # ticks per beat; tempo 60bpm -> 1 beat = 1 s
mid=mido.MidiFile(ticks_per_beat=TPB)
ev=[]  # (time_s, track, msg)
def note(ch,t,n,d,v): ev.append((t,mido.Message('note_on',channel=ch,note=n,velocity=v))); ev.append((t+d,mido.Message('note_off',channel=ch,note=n,velocity=0)))
def cc(ch,t,c,val): ev.append((t,mido.Message('control_change',channel=ch,control=c,value=int(max(0,min(127,val))))))
def ramp(ch,t0,t1,v0,v1,c=11,steps=None):
    steps=steps or max(2,int((t1-t0)*10))
    for k in range(steps+1): cc(ch,t0+(t1-t0)*k/steps,c,v0+(v1-v0)*k/steps)
PIANO,STR,CELLO,PAD,CHOIR,CEL=0,1,2,3,4,5
for ch,prog in [(PIANO,0),(STR,49),(CELLO,42),(PAD,89),(CHOIR,52),(CEL,8)]:
    ev.append((0,mido.Message('program_change',channel=ch,program=prog)))
    cc(ch,0,7,100); cc(ch,0,91,70)
ev.append((0,mido.Message('control_change',channel=PIANO,control=64,value=0)))
N=lambda s: {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[s[0]]+(1 if '#' in s else -1 if 'b' in s[1:] else 0)+12*(int(s[-1])+1)
# chord: (start, dur, bass, voicing(list), arpeggio notes)
def pedal(t,d):
    cc(PIANO,t,64,0); cc(PIANO,t+0.02,64,127); cc(PIANO,t+d-0.03,64,0)
def piano_chord(t,d,bass,arp,vel=48,pattern=(0,1,2,3),step=1.0,melody=None):
    pedal(t,d)
    note(PIANO,t,N(bass),d,vel+6)
    for k,beat in enumerate(pattern):
        if beat*step < d-0.05:
            note(PIANO,t+beat*step,N(arp[k%len(arp)]),d-beat*step,vel)
    if melody:
        for (mt,mn,md,mv) in melody: note(PIANO,t+mt,N(mn),md,mv)
def pad(ch,t,d,notes,vel=60):
    for n in notes: note(ch,t,N(n),d+0.3,vel)

# ---- Section A 0-24: tender solo piano, pad enters at 8
A=[('C2',['G3','C4','E4','G4']),('B1',['G3','B3','D4','G4']),('A1',['E3','A3','C4','E4']),('F1',['A3','C4','F4','A4']),
   ('C2',['G3','C4','E4','G4']),('B1',['G3','B3','D4','G4'])]
melA=[[(0.0,'E5',2,52),(2,'D5',2,46)],[(0,'D5',1.5,48),(1.5,'C5',0.5,40),(2,'B4',2,44)],[(0,'C5',3,48)],[(1,'A4',1,40),(2,'C5',2,46)],
      [(0,'G5',2,50),(2,'E5',2,46)],[(0,'D5',4,46)]]
t=0.6
for i,(b,arp) in enumerate(A):
    piano_chord(t+i*4,4,b,arp,vel=38+i*2,melody=melA[i])
cc(PAD,0,11,0); pad(PAD,8.6,15.5,['C3','G3','E4']); ramp(PAD,8.6,14,0,70)
# ---- Section B 24.6-56.6: strings + cello, arps in 8ths
B=[('A1',['E3','A3','C4','E4'],['A2','E3','C4']),('F1',['C4','F4','A4','C5'],['F2','C3','A3']),('C2',['G3','C4','E4','G4'],['C3','G3','E4']),
   ('G1',['D4','G4','B4','D5'],['G2','D3','B3']),('A1',['E4','A4','C5','E5'],['A2','E3','C4']),('F1',['C4','F4','A4','C5'],['F2','C3','A3']),
   ('D2',['F4','A4','C5','D5'],['D3','A3','F4']),('G1',['D4','G4','C5','D5'],['G2','D3','C4'])]
melB=[[(0,'E5',2,54),(2,'C5',2,48)],[(0,'F5',1,50),(1,'E5',1,46),(2,'C5',2,48)],[(0,'G5',3,56)],[(0,'G5',1,52),(1,'A5',1,52),(2,'B5',2,56)],
      [(0,'C6',2,58),(2,'A5',2,52)],[(0,'A5',2,54),(2,'C6',2,56)],[(0,'D6',3,60)],[(0,'D6',2,54),(2,'C6',2,48)]]
t0=24.6
cc(STR,0,11,0); cc(CELLO,0,11,0)
for i,(b,arp,sv) in enumerate(B):
    tt=t0+i*4
    piano_chord(tt,4,b,arp,vel=46+i,pattern=(0,1,2,3,4,5,6,7),step=0.5,melody=melB[i])
    pad(STR,tt,4,sv,vel=64); note(CELLO,tt,N(b)+12,4.2,70)
    pad(PAD,tt,4,sv[1:],vel=50)
ramp(STR,24.6,40,20,85); ramp(STR,40,54,85,105); ramp(CELLO,24.6,34,0,80)
ramp(PAD,24.6,56,70,55)
# ---- Section C 56.6-61.0: the held breath (singularity/ending)
pad(STR,56.6,3.2,['F3','A3','E4'],vel=50); ramp(STR,56.6,59.6,90,0); ramp(CELLO,56.6,58.5,80,0)
pad(PAD,56.6,4.2,['F3','C4','E4'],vel=45); ramp(PAD,56.6,60.8,55,20)
note(PIANO,56.6,N('F2'),3,40); pedal(56.6,4.3); note(PIANO,56.6,N('E5'),3,36)
note(PIANO,59.3,N('E5'),1.6,28)  # "It wasn't." single note
# ---- Section D 61.0-: the door
cc(STR,60.9,11,0); cc(CHOIR,0,11,0); cc(CELLO,60.9,11,0)
D=[(61.0,2.2,'Ab1',['Eb3','Ab3','C4','Eb4','Bb4'],['Ab2','Eb3','C4','Eb4']),
   (63.2,2.2,'Bb1',['F3','Bb3','D4','F4','C5'],['Bb2','F3','D4','F4']),
   (65.4,4.0,'C2',['G3','C4','E4','G4','D5'],['C3','G3','E4','G4']),
   (69.4,2.0,'A1',['E3','A3','C4','E4','B4'],['A2','E3','C4','E4']),
   (71.4,2.0,'F1',['C4','F4','A4','C5','G5'],['F2','C3','A3','C4']),
   (73.4,8.5,'C2',['G3','C4','E4','G4','D5'],['C3','G3','E4','G4'])]
for (tt,d,b,arp,sv) in D:
    pedal(tt,d)
    note(PIANO,tt,N(b),d,70); note(PIANO,tt,N(b)+12,d,62)
    for k,n in enumerate(arp): note(PIANO,tt+0.06*k,N(n),d,58)
    pad(STR,tt,d,sv,vel=80); pad(CHOIR,tt,d,sv[1:],vel=60); note(CELLO,tt,N(b)+12,d+0.2,80); pad(PAD,tt,d,sv[1:],vel=55)
ramp(STR,61.0,62.4,10,112); ramp(CELLO,61.0,62.4,10,100); ramp(CHOIR,61.0,66,0,70)
ramp(STR,74,81.5,112,0); ramp(CHOIR,74,81,70,0); ramp(CELLO,74,80,100,0); ramp(PAD,74,81.5,60,0)
# melody on top in D
for (mt,mn,md,mv) in [(61.0,'C6',2.2,70),(63.2,'D6',2.2,70),(65.4,'E6',2.0,74),(67.4,'G6',2.0,66),(69.4,'E6',2,62),(71.4,'C6',2,60),(73.4,'D6',4,56)]:
    note(PIANO,mt,N(mn),md,mv)
# celesta twinkles after welcome
import random; random.seed(3)
for k in range(14):
    tt=66.0+k*1.0+random.choice([0,0.5]); n=random.choice(['G5','C6','D6','E6','G6'])
    note(CEL,tt,N(n),1.5,int(40-k*2))
# write
ev.sort(key=lambda e:(e[0], 0 if e[1].type!='note_on' else 1))
tr=mido.MidiTrack(); mid.tracks.append(tr)
tr.append(mido.MetaMessage('set_tempo',tempo=1000000))
last=0
for tt,m in ev:
    tick=int(round(tt*TPB)); tr.append(m.copy(time=tick-last)); last=tick
mid.save('score.mid'); print('ok',len(ev))
