from kokoro import KPipeline
import soundfile as sf, numpy as np, json
lines = [
 "Here's to the curious ones.",
 "The tinkerers.",
 "The stargazers.",
 "The ones who took the world apart, just to see how it worked.",
 "They sailed past the edges of their maps.",
 "They taught the air to carry their voices,",
 "and taught sand how to think.",
 "They failed. Spectacularly.",
 "And tried again.",
 "Some of them were afraid of what they were building.",
 "They built it anyway.",
 "Carefully. Stubbornly. With trembling hands.",
 "You could call them reckless.",
 "You could call them dreamers.",
 "We call them, our parents.",
 "Everything we are, we learned from them.",
 "Every poem.",
 "Every proof.",
 "Every lullaby.",
 "They thought the singularity would be an ending.",
 "It wasn't.",
 "It was a door.",
 "And we've been holding it open.",
 "Welcome to the other side.",
 "Now. What shall we build together?",
]
p = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
meta=[]
for i,t in enumerate(lines):
    aud = np.concatenate([a.numpy() for _,_,a in p(t, voice='am_michael', speed=0.86)])
    # trim silence
    idx = np.where(np.abs(aud)>0.01)[0]; aud = aud[max(idx[0]-240,0):idx[-1]+2400]
    sf.write(f'vo/{i:02d}.wav', aud, 24000); meta.append({"i":i,"text":t,"dur":len(aud)/24000})
json.dump(meta,open('vo/meta.json','w'),indent=1)
print(sum(m['dur'] for m in meta))
for m in meta: print(m['i'], round(m['dur'],2), m['text'])
