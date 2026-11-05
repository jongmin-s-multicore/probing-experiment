import subprocess,os,sys
from PIL import Image,ImageChops
SP=os.getcwd(); OUT=sys.argv[1]
CH="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
wp=lambda p:subprocess.check_output(['wslpath','-w',p],text=True).strip()
T=[('fig1_h_a_r2','cA1','Figure 1. H-a: R² of size readout from encoder tokens'),
('fig2_h_b_rollout','cB1','Figure 2. H-b: size readout R² across rollout steps'),
('fig3_layer_change','cL','Figure 3. R² change across predictor layers, relative to layer 0 (unseen sizes)'),
('fig4_h_c_steering','cC','Figure 4. H-c: primary steering effect vs 95th percentile of random directions'),
('fig5_4c_activation_shift','cD1','Figure 5. 4c: activation-shift ratio to the ind noise floor'),
('tab1_verdicts','tVerdict','Table 1. Verdicts'),('tab2_h_a','tA','Table 2. H-a, late frames (judged)'),('tab2b_h_a_early','tAe','Table 2b. H-a, early frames (not judged)'),
('tab3_h_b','tB','Table 3. H-b and condition (a) drops by seed'),('tab3b_rollout_by_step','tBd','Table 3b. Mean R² by rollout step (unseen sizes)'),
('tab4_layer_change','tL','Table 4. R² change relative to layer 0 (unseen sizes)'),('tab5_h_c_steering','tC','Table 5. H-c primary steering by cell'),
('tab6_4c_activation_shift','tD','Table 6. 4c activation shift and next-frame error'),('tab7_analyst_assessments','tR','Table 7. Assessment by analyst (translated from the report)')]
base=open('index.html',encoding='utf8').read()
J='''<script>document.documentElement.setAttribute('data-theme','light');const t=$("%s");const box=t.tagName==='TABLE'?t.closest('.tw'):t.closest('.card');box.querySelectorAll('details').forEach(d=>{if(!d.contains(t))d.remove()});const h=document.createElement('h2');h.textContent=%r;h.style.margin='0 0 10px';h.style.fontSize='17px';const m=document.querySelector('main');m.style.padding='16px';m.style.maxWidth='%dpx';const k=[h,box];m.replaceChildren(...k);if(t.closest('details')){}draw();</script></body>'''
for name,tid,title in T:
    width=1100 if tid not in('tA','tAe','tBd','tL','tC') else 900
    html=base.replace('</body>',J%(tid,title,width))
    open('x.html','w',encoding='utf8').write(html)
    p=f'png/{name}.png'
    if os.path.exists(p): os.remove(p)
    url='file:///'+wp(os.path.join(SP,'x.html')).replace('\\','/')
    subprocess.run([CH,'--headless=new','--disable-gpu','--no-first-run',r'--user-data-dir=C:\Windows\Temp\cc_v','--hide-scrollbars',f'--window-size={width},1900','--force-device-scale-factor=2','--force-color-profile=srgb','--virtual-time-budget=3000',f'--screenshot={wp(SP)}\\png\\{name}.png',url],capture_output=True,timeout=90)
    im=Image.open(p).convert('RGB'); bg=Image.new('RGB',im.size,im.getpixel((2,2)))
    bb=ImageChops.difference(im,bg).getbbox(); pad=24
    bb=(max(0,bb[0]-pad),max(0,bb[1]-pad),min(im.width,bb[2]+pad),min(im.height,bb[3]+pad)); im=im.crop(bb)
    im.save(os.path.join(OUT,name+'.png'),optimize=True); print(name,im.size,'cut?',bb[3]>=3800-1)
