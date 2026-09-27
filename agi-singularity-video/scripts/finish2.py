import json, subprocess
exec(open('finish.py').read().split('# ---- audio')[0])  # meta, starts, tf
GAR='/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf'
INT='/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
INTL='/usr/share/fonts/opentype/inter/Inter-Light.otf'
def tf(name,text):
    p=f'txt/{name}.txt'; open(p,'w').write(text); return p
def fa(a,b,f=0.6): return f"if(lt(t,{a}),0,if(lt(t,{a+f}),(t-{a})/{f},if(lt(t,{b-f}),1,if(lt(t,{b}),({b}-t)/{f},0))))"
import textwrap
NL=chr(10)
def caps(size,y,wrap=0):
    out=[]
    for i,(m,s) in enumerate(zip(meta,starts)):
        e=s+m['dur']+0.35
        if i+1<len(starts): e=min(e,starts[i+1]-0.05)
        out.append(f"drawtext=fontfile={INT}:textfile={tf(f'c{i:02d}' + ('v' if wrap else ''),NL.join(textwrap.wrap(m['text'],wrap)) if wrap else m['text'])}:text_align=C:line_spacing=10:fontsize={size}:fontcolor=white:borderw=3:bordercolor=black@0.55:shadowcolor=black@0.6:shadowy=3:x=(w-tw)/2:y={y}:alpha='{fa(s-0.1,e,0.15)}'")
    return out
def end(title,credit,cy):
    return [f"drawtext=fontfile={GAR}:textfile={tf('end1','Welcome to the other side.')}:fontsize={title}:fontcolor=white:x=(w-tw)/2:y=(h-th)/2-10:alpha='{fa(74.4,79.6,1.0)}'",
            f"drawtext=fontfile={INTL}:textfile={tf('end2','Archival footage: U.S. public domain, found via destockd.com')}:fontsize={credit}:fontcolor=0xa0a0a0:x=(w-tw)/2:y={cy}:alpha='{fa(76.0,81.6,0.8)}'"]
enc=['-c:v','libx264','-preset','medium','-b:v','14M','-maxrate','18M','-bufsize','28M','-pix_fmt','yuv420p','-profile:v','high','-movflags','+faststart','-c:a','aac','-b:a','256k','-t','82']
base="noise=alls=5:allf=t,vignette=PI/5"
if 0: subprocess.run(['ffmpeg','-v','error','-y','-i','picture.mp4','-i','mix.wav','-vf',','.join([base]+caps(50,'h-165')+end(78,24,'h-90')),'-map','0:v','-map','1:a',*enc,'out_captions.mp4'],check=True); print('captions')
v=f"{base},crop=864:1080:528:0,scale=1080:1350:flags=lanczos,pad=1080:1920:0:285:black,"+','.join(caps(46,'1690',30)+end(64,22,'h-140'))
subprocess.run(['ffmpeg','-v','error','-y','-i','picture.mp4','-i','mix.wav','-vf',v,'-map','0:v','-map','1:a',*enc,'out_vertical.mp4'],check=True); print('vertical')
