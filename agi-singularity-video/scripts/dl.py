import json, urllib.request, subprocess, sys
H={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) Chrome/126 Safari/537.36"}
picks="face_34 tinker_35 lookup_21 tinker_4 ship_43 ship_20 telephone_2 radio_4 computer_5 hands_4 explode_9 rocket_5 atomic_13 scientist_47 hands_36 plane_6 plane_36 mother_38 library_11 hands_11 chalk_2 mother_36 sunrise_1 door_2 sunrise_3 apollo_32 colorface_1 dancing_4 child_26 lookup_35 face_41 colorkids_1 apollo_18 reading_43 mother_35 sunrise_10".split()
credits=[]
for p in picks:
    tag,i=p.rsplit('_',1); r=json.load(open(f'kf/{tag}.json'))[int(i)]
    fn=f'clips/{p}.mp4'
    open(fn,'wb').write(urllib.request.urlopen(urllib.request.Request(r['clip'].replace(' ','%20'),headers=H)).read())
    d=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,r_frame_rate','-of','csv=p=0',fn],capture_output=True,text=True).stdout.split()
    print(p, d, r['color_type'], r['film'])
    credits.append({"id":p,"film":r['film'],"shot":r['shot'],"clip":r['clip'],"color":r['color_type']})
json.dump(credits,open('clips/credits.json','w'),indent=1)
