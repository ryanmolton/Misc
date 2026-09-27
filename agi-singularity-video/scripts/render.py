import subprocess, json, os, sys
FPS=24; W,H=1920,1080
CLIP='clips/%s.mp4'
def dur(f): return float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',f],capture_output=True,text=True).stdout)
# (id, src_start, out_start, out_end, opts)
EDL=[
 ('face_34',0.5,1.0,5.0,dict(fi=0.9,yf=0.3)),
 ('tinker_35',0.3,5.0,7.0,{}),
 ('lookup_21',0.1,7.0,9.2,dict(yf=0.35)),
 ('tinker_4',0.0,9.2,11.4,{}),
 ('hands_4',4.0,11.4,13.6,{}),
 ('ship_43',0.0,13.6,15.8,{}),
 ('ship_20',0.0,15.8,17.6,{}),
 ('telephone_2',0.3,17.6,19.4,dict(yf=0.4)),
 ('radio_4',0.2,19.4,20.6,dict(yf=0.3)),
 ('computer_5',0.3,20.6,23.2,{}),
 ('explode_9',1.6,23.2,26.0,dict(yf=0.7)),
 ('rocket_5',37.0,26.0,28.6,dict(yf=0.6)),
 ('sunrise_3',1.4,28.6,30.6,dict(yf=0.7)),
 ('atomic_13',7.4,30.6,32.4,dict(yf=0.55)),
 ('scientist_47',0.0,32.4,34.6,dict(yf=0.35)),
 ('hands_36',0.3,34.6,38.4,{}),
 ('plane_6',1.6,38.4,42.2,dict(yf=0.45)),
 ('mother_35',2.0,42.2,46.2,dict(yf=0.35)),
 ('library_11',0.0,46.2,49.4,{}),
 ('hands_11',0.0,49.4,50.9,{}),
 ('chalk_2',2.0,50.9,52.4,dict(yf=0.3)),
 ('mother_36',0.0,52.4,55.0,dict(yf=0.4)),
 ('sunrise_1',0.0,55.0,58.9,dict(fo=1.4,yf=0.55)),
 ('door_2',0.0,61.0,64.2,dict(fi=0.3,zoom=0.16,yf=0.5)),
 ('sunrise_10',0.0,64.2,66.4,dict(yf=0.5,fo_white=0.5)),
 ('apollo_32',0.5,66.4,68.2,dict(color=True,fi_white=0.6)),
 ('colorface_1',3.6,68.2,69.4,dict(color=True,yf=0.3)),
 ('dancing_4',0.0,69.4,70.6,dict(color=True,yf=0.4)),
 ('child_26',14.0,70.6,73.6,dict(color=True,yf=0.35,fo=1.2)),
]
END=82.0
os.makedirs('seg',exist_ok=True)
def frames(t): return int(round(t*FPS))
segs=[]; cur=0.0
def black(a,b,name):
    n=frames(b)-frames(a)
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'color=black:s={W}x{H}:r={FPS}','-frames:v',str(n),'-pix_fmt','yuv420p','-c:v','libx264','-crf','12','-preset','fast',name],check=True)
    segs.append(name)
for k,(cid,ss,a,b,o) in enumerate(EDL):
    if frames(a)>frames(cur): black(cur,a,f'seg/{k:02d}_black.mp4')
    n=frames(b)-frames(a); outdur=n/FPS
    f=CLIP%cid; avail=dur(f)-ss-0.08
    speed=min(1.0,avail/outdur)
    yf=o.get('yf',0.5); z=o.get('zoom',0.06)
    src_w=subprocess.run(['ffprobe','-v','error','-select_streams','v','-show_entries','stream=width','-of','csv=p=0',f],capture_output=True,text=True).stdout.strip()
    pre='' if src_w=='1920' else 'scale=1920:1080,'
    y0=int((1050-788)*yf)
    grade = 'eq=saturation=1.12:contrast=1.04,colorbalance=rm=0.03:bm=-0.03' if o.get('color') else 'hue=s=0,eq=contrast=1.12:gamma=0.97'
    T=outdur
    vf=(f"{pre}crop=1400:1050:260:15,crop=1400:788:0:{y0},setpts=(PTS-STARTPTS)/{speed:.4f},fps={FPS},"
        f"scale=w='trunc({W}*(1+{z}*t/{T:.3f})/2)*2':h='trunc({H}*(1+{z}*t/{T:.3f})/2)*2':eval=frame:flags=bicubic,"
        f"crop={W}:{H}:(iw-{W})/2:(ih-{H})/2,setsar=1,{grade}")
    if 'fi' in o: vf+=f",fade=in:st=0:d={o['fi']}"
    if 'fo' in o: vf+=f",fade=out:st={T-o['fo']:.3f}:d={o['fo']}"
    if 'fi_white' in o: vf+=f",fade=in:st=0:d={o['fi_white']}:color=white"
    if 'fo_white' in o: vf+=f",fade=out:st={T-o['fo_white']:.3f}:d={o['fo_white']}:color=white"
    name=f'seg/{k:02d}_{cid}.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(ss),'-i',f,'-an','-vf',vf,'-frames:v',str(n),'-pix_fmt','yuv420p','-c:v','libx264','-crf','12','-preset','fast',name],check=True)
    segs.append(name); cur=b
    print(cid, f'speed={speed:.2f}', n)
black(cur,END,'seg/99_black.mp4')
open('seg/list.txt','w').write(''.join(f"file '{os.path.basename(s)}'\n" for s in segs))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i','seg/list.txt','-c','copy','picture.mp4'],check=True)
print('total',dur('picture.mp4'))
