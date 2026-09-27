import json, subprocess, os
meta=json.load(open('vo/meta.json'))
starts=[2.5,5.3,7.0,9.3,14.2,17.7,20.4,23.2,26.0,28.6,32.3,34.4,38.6,40.8,43.6,46.6,49.6,51.1,52.6,55.4,59.3,61.2,63.4,66.6,70.0]
assert len(starts)==len(meta)
os.makedirs('txt',exist_ok=True)
# ---- audio
inputs=['-i','music_rev.wav']; parts=[]
for i,(m,s) in enumerate(zip(meta,starts)):
    inputs+=['-i',f'vo/{i:02d}.wav']; ms=int(s*1000)
    parts.append(f'[{i+1}:a]aresample=48000,adelay={ms}|{ms},apad[v{i}]')
n=len(meta)
fc=';'.join(parts)+';'+''.join(f'[v{i}]' for i in range(n))+f'amix=inputs={n}:normalize=0:duration=longest,atrim=0:82,'\
   'highpass=f=70,equalizer=f=3000:t=q:w=1:g=2,acompressor=threshold=0.1:ratio=3:attack=5:release=120,aecho=0.8:0.6:45|80:0.10|0.06,pan=stereo|c0=c0|c1=c0,asplit[vo][key];'\
   '[0:a]aresample=48000,atrim=0:82,volume=0.9[mus];[mus][key]sidechaincompress=threshold=0.04:ratio=3:attack=30:release=500:makeup=1[duck];'\
   '[duck][vo]amix=inputs=2:normalize=0:weights=1 1.25,afade=t=out:st=79:d=3,loudnorm=I=-14:TP=-1.5:LRA=11[out]'
subprocess.run(['ffmpeg','-v','error','-y',*inputs,'-filter_complex',fc,'-map','[out]','-ar','48000','-c:a','pcm_s16le','mix.wav'],check=True)
# ---- text
GAR='/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf'
INT='/usr/share/fonts/opentype/inter/Inter-Medium.otf'
INTL='/usr/share/fonts/opentype/inter/Inter-Light.otf'
def tf(name,text):
    p=f'txt/{name}.txt'; open(p,'w').write(text); return p
def fade_alpha(a,b,f=0.6): return f"if(lt(t,{a}),0,if(lt(t,{a+f}),(t-{a})/{f},if(lt(t,{b-f}),1,if(lt(t,{b}),({b}-t)/{f},0))))"
end=[f"drawtext=fontfile={GAR}:textfile={tf('end1','Welcome to the other side.')}:fontsize=78:fontcolor=white:x=(w-tw)/2:y=(h-th)/2-10:alpha='{fade_alpha(74.4,79.6,1.0)}'",
     f"drawtext=fontfile={INTL}:textfile={tf('end2','Archival footage: U.S. public domain, found via destockd.com')}:fontsize=22:fontcolor=0x9a9a9a:x=(w-tw)/2:y=h-90:alpha='{fade_alpha(76.0,81.6,0.8)}'"]
caps=[]
for i,(m,s) in enumerate(zip(meta,starts)):
    e=s+m['dur']+0.35
    if i+1<len(starts): e=min(e,starts[i+1]-0.05)
    caps.append(f"drawtext=fontfile={INT}:textfile={tf(f'c{i:02d}',m['text'])}:fontsize=40:fontcolor=white:shadowcolor=black@0.7:shadowx=0:shadowy=2:x=(w-tw)/2:y=h-150:alpha='{fade_alpha(s-0.1,e,0.15)}'")
base="noise=alls=5:allf=t,vignette=PI/5"
for name,chain in [('clean',[base]+end),('captions',[base]+caps+end)]:
    subprocess.run(['ffmpeg','-v','error','-y','-i','picture.mp4','-i','mix.wav','-vf',','.join(chain),'-t','82','-map','0:v','-map','1:a',
        '-c:v','libx264','-preset','medium','-b:v','14M','-maxrate','18M','-bufsize','28M','-pix_fmt','yuv420p','-profile:v','high','-movflags','+faststart','-c:a','aac','-b:a','256k',f'out_{name}.mp4'],check=True)
    print(name,'done')
