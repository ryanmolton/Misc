import time, sys, json, urllib.request, urllib.parse, os, subprocess
from PIL import Image, ImageDraw
H={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) Chrome/126 Safari/537.36"}
q = sys.argv[1]; tag = sys.argv[2]; pages = int(sys.argv[3]) if len(sys.argv)>3 else 1
res=[]
for p in range(1,pages+1):
    url = "https://destockd.com/api/search?"+urllib.parse.urlencode({"q":q,"page":p})
    res += json.load(urllib.request.urlopen(urllib.request.Request(url,headers=H)))["results"]; time.sleep(0.5)
os.makedirs("kf",exist_ok=True)
tiles=[]
for i,r in enumerate(res):
    fn=f"kf/{tag}_{i}.jpg"
    if not os.path.exists(fn):
        try: open(fn,"wb").write(urllib.request.urlopen(urllib.request.Request("https://destockd.com"+r["keyframe"],headers=H)).read())
        except Exception as e: continue
    tiles.append((i,fn,r))
json.dump([r for _,_,r in tiles], open(f"kf/{tag}.json","w"))
W,H=240,180; cols=8; rows=(len(tiles)+cols-1)//cols
sheet=Image.new("RGB",(W*cols,H*rows))
d=ImageDraw.Draw(sheet)
for k,(i,fn,r) in enumerate(tiles):
    im=Image.open(fn).convert("RGB"); im.thumbnail((W,H))
    x,y=(k%cols)*W,(k//cols)*H; sheet.paste(im,(x,y))
    d.rectangle([x,y,x+40,y+22],fill="red"); d.text((x+4,y+4),str(i),fill="white")
sheet.save(f"sheet_{tag}.jpg",quality=80)
print(tag,len(tiles))
