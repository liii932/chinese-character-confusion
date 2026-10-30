"""Development-only low-resolution samples, blinded forms, and offline scoring."""
import argparse, base64, csv, hashlib, io, json, math, random, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/data/round2-1140-v1'
OUT=ROOT/'training/data/lowres1140-human-v1'
RS=[64,48,32,24,16]
SEED=20261029

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>单字盲认</title>
<style>body{font:18px sans-serif;max-width:760px;margin:32px auto;padding:16px;color:#183047}button,input{font:inherit;padding:10px;margin:5px}img{width:128px;height:128px;image-rendering:pixelated;display:block;margin:25px auto;border:1px solid #ddd}#panel{text-align:center}p{line-height:1.6}</style>
<h1>单字盲认</h1><p>每人只做分配的一套，使用电脑、浏览器100%缩放，保持默认图像大小。每30题可休息。不要先看答案或带标签预览，不使用OCR、搜索或字表。输入看到的一个汉字，看不清选“不认识”。测试中不显示答案。</p>
代号<input id="pid" placeholder="例如P01"><button onclick="start()">开始/继续</button><button onclick="reset()">新受试者/清空</button>
<div id="panel" hidden><p id="status"></p><img id="pic" alt="待识别单字"><input id="answer" autocomplete="off" placeholder="一个汉字"><br><button onclick="submit(false)">提交并下一题</button><button onclick="submit(true)">不认识</button><button onclick="back()">上一题</button><button onclick="download()">导出作答JSON</button></div>
<script>
const form=FORM_TOKEN, trials=TRIALS_TOKEN, key='hanzi-lowres-v1-'+form;
let s=JSON.parse(localStorage.getItem(key)||'null')||{participant:'',index:0,answers:{},started:null};
const el=id=>document.getElementById(id);el('pid').value=s.participant;
function persist(){localStorage.setItem(key,JSON.stringify(s));}
function start(){let p=el('pid').value.trim();if(!p){alert('请输入代号');return;}if(s.participant&&s.participant!==p&&Object.keys(s.answers).length){alert('请先导出旧进度，再清空。');return;}s.participant=p;s.started=s.started||new Date().toISOString();el('panel').hidden=false;show();persist();}
function show(){let t=trials[s.index];el('pic').src=t.image;el('answer').value=(s.answers[t.id]||{}).response||'';el('status').textContent=`题目 ${s.index+1}/600 · 组 ${Math.floor(s.index/30)+1}/20 · 已答 ${Object.keys(s.answers).length}`;el('answer').focus();}
function submit(unknown){let a=unknown?'':el('answer').value.trim();if(!unknown&&Array.from(a).length!==1){alert('请输入一个字，或选择不认识');return;}s.answers[trials[s.index].id]={response:a,answered_at:new Date().toISOString()};if(s.index<599){s.index++;show();if(s.index%30===0)alert('已完成一组，可以休息。');}else alert('请导出作答JSON。');persist();}
function back(){s.index=Math.max(0,s.index-1);show();persist();}
function download(){let completed=trials.every(t=>s.answers[t.id]);let r={schema:1,form,participant:s.participant,started:s.started,exported:new Date().toISOString(),completed,answers:trials.map(t=>({trial_id:t.id,...(s.answers[t.id]||{response:null})})),display:{css_pixels:128,dpr:devicePixelRatio,user_agent:navigator.userAgent}};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(r,null,2)],{type:'application/json'}));a.download=form+'_'+s.participant+(completed?'_complete':'_partial')+'.json';a.click();URL.revokeObjectURL(a.href);}
function reset(){if(confirm('请先导出作答。确定清空本套进度吗？')){localStorage.removeItem(key);location.reload();}}
el('answer').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.isComposing)submit(false);});
</script></html>'''

def generate():
    if OUT.exists():raise FileExistsError(str(OUT))
    m=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    for name in ['val_graph.npz','conditions.json','labels.json']:
        if digest(DATA/name)!=m['files'][name]:raise ValueError('Source changed: '+name)
    labels=json.loads((DATA/'labels.json').read_text(encoding='utf-8'))['characters']
    cs=json.loads((DATA/'conditions.json').read_text(encoding='utf-8'))['val_graph']
    cid=min((i for i,c in enumerate(cs) if c['sigma']==0),key=lambda i:cs[i]['dx']**2+cs[i]['dy']**2)
    with np.load(DATA/'val_graph.npz',allow_pickle=False) as d:
        ids=np.flatnonzero(d['condition_ids']==cid);images=d['images'][ids]
        np.testing.assert_array_equal(d['labels'][ids],np.arange(1140))
    public=OUT/'participants';private=OUT/'organizer';preview=private/'preview'
    public.mkdir(parents=True);preview.mkdir(parents=True)
    font=ImageFont.truetype(str(ROOT/'pilot/fonts/SourceHanSansSC-Regular.otf'),18)
    encoded={};seen=set();pixels=[]
    for r in RS:
        tiles=[]
        for i,row in enumerate(labels):
            im=Image.fromarray(images[i])
            if r!=64:im=im.resize((r,r),Image.Resampling.LANCZOS).resize((64,64),Image.Resampling.BICUBIC)
            a=np.asarray(im)
            if a.min()==255:raise ValueError('Blank image')
            if r==64:np.testing.assert_array_equal(a,images[i])
            h=hashlib.sha256(a.tobytes()).hexdigest()
            if h in seen:raise ValueError('Repeated pixels')
            seen.add(h);buffer=io.BytesIO();im.save(buffer,format='PNG')
            raw=buffer.getvalue();encoded[i,r]='data:image/png;base64,'+base64.b64encode(raw).decode()
            (preview/f'{i:04d}_R{r}.png').write_bytes(raw);tiles.append(im)
            pixels.append({'class_id':i,'resolution':r,'pixel_sha256':h})
        for start in range(0,1140,120):
            subset=tiles[start:start+120];sheet=Image.new('RGB',(1000,70+100*math.ceil(len(subset)/10)),'white');draw=ImageDraw.Draw(sheet)
            draw.text((12,12),f'Organizer only R={r} sigma=0 IDs {start}..{start+len(subset)-1}',font=font,fill='black')
            for j,im in enumerate(subset):
                x,y=10+j%10*98,60+j//10*100;sheet.paste(im,(x,y));draw.text((x,y+65),f'{start+j} {labels[start+j]["char"]}',font=font,fill='black')
            sheet.save(preview/f'sheet_R{r}_{start//120+1:02d}.png')
    rng=random.Random(SEED);strata={}
    for row in labels:strata.setdefault(row['category'],[]).append(row['class_id'])
    quotas={c:int(len(v)*600/1140) for c,v in strata.items()}
    for c in sorted(strata,key=lambda c:len(strata[c])*600/1140-quotas[c],reverse=True)[:600-sum(quotas.values())]:quotas[c]+=1
    selected=[]
    for c,pool in strata.items():selected+=rng.sample(pool,quotas[c])
    rng.shuffle(selected);answers=[]
    for f in range(5):
        form=f'F{f+1}';trials=[];rows=[]
        for j,i in enumerate(selected):
            r=RS[(j+f)%5];tid=form+'-'+hashlib.sha256(f'blind-v1-{form}-{j+1:04d}'.encode()).hexdigest()[:16];trials.append({'id':tid,'image':encoded[i,r]})
            rows.append({'form':form,'trial_id':tid,'char':labels[i]['char'],'class_id':i,'resolution':r,'sigma':0,'source_condition_id':cid,'source_index':int(ids[i])})
        assert Counter(x['resolution'] for x in rows)=={r:120 for r in RS}
        assert len({x['class_id'] for x in rows})==600
        rng.shuffle(trials);page=PAGE.replace('FORM_TOKEN',json.dumps(form)).replace('TRIALS_TOKEN',json.dumps(trials))
        if '"char":' in page or '"resolution":' in page:raise ValueError('Answer leakage')
        (public/f'{form}.html').write_text(page,encoding='utf-8');answers+=rows
    with (private/'answer_key.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(answers[0]));w.writeheader();w.writerows(answers)
    save(private/'pixels.json',pixels)
    protocol={'seed':SEED,'source_split':'val_graph','source_manifest_sha256':digest(DATA/'manifest.json'),'source_condition_id':cid,'source_condition':cs[cid],'resolutions':RS,'sigma':0,'pipeline':'clear64 -> Lanczos R -> bicubic64; R64 identity','preview_images':len(pixels),'unique_pixels':len(seen),'forms':5,'trials_per_form':600,'per_resolution_per_form':120,'unique_characters_per_form':600,'stratification_quotas':quotas,'design':'One form per participant. Five forms rotate same 600 characters across R. No feedback. No train/test arrays read.','human_accuracy':'not_measured','display':'128 CSS pixels, nearest-neighbor, browser100%, computer','runtime':{'numpy':np.__version__,'Pillow':Image.__version__}}
    save(private/'protocol.json',protocol)
    with zipfile.ZipFile(OUT/'participant_forms.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(public.iterdir()):z.write(p,p.name)
    (OUT/'README.md').write_text('''# 低分辨率人工盲认预试验

participants 的HTML可直接用浏览器打开。仅将分配的一套交给受试者；participant_forms.zip只含五份盲认HTML。organizer含答案及带标签预览，不给受试者看。

每套600题，R=64/48/32/24/16各120题，sigma固定0。同一套每字只出现一次，五套轮换相同600字的分辨率。每人只做一套，每30题休息。使用电脑、浏览器100%缩放、默认128 CSS像素显示，不用OCR、搜索或字表，按看到的单字手工输入；不认识计错。页面本地保存进度，做完导出JSON交给组织者。

先分配一名受试者做F1检查流程；后续推荐五名不同受试者分别做F1..F5，不让同一人重复五套。记录是否看过本项目带标签样张；已看过者结果单列。组织者如果看过答案或标签预览，不作为同组盲认受试者。预试验覆盖600个分层抽样字，不能直接代表全部1140字可读率。

评分：在项目根目录运行 python -m training.prepare_human_lowres --score 作答JSON路径。生成scored.json和scored.csv。分母包含不认识和未答；未答完只作中间记录。报告各R准确率与Wilson95区间。拟定筛选线为准确率>=95%，同时查看区间下界；若要形成更强可读证据，完整测试区间下界也应>=95%。区间是描述性统计，相关字形、受试者差异仍需考虑。R64对照失败时先检查设置与输入问题，保留所有原始记录，不筛除不利结果。最差字组还需单独核验，不能只靠平均值定训练条件。

本轮只使用开发集val_graph清晰图，未运行模型、未重训、未读取最终test。人工结果尚未采集。
''',encoding='utf-8')
    print(json.dumps(protocol,ensure_ascii=False,indent=2))

def score(path):
    response=json.loads(Path(path).read_text(encoding='utf-8'))
    with (OUT/'organizer/answer_key.csv').open(encoding='utf-8-sig',newline='') as f:key={r['trial_id']:r for r in csv.DictReader(f) if r['form']==response['form']}
    answers={r['trial_id']:r for r in response['answers']}
    if not key or len(answers)!=len(response['answers']) or set(answers)!=set(key):raise ValueError('Trial ID mismatch')
    rows=[]
    for tid,k in key.items():
        value=answers[tid].get('response');answered=value is not None
        rows.append({**k,'response':value,'answered':answered,'correct':answered and value.strip()==k['char']})
    by={}
    for r in RS:
        sub=[x for x in rows if int(x['resolution'])==r];n=len(sub);correct=sum(x['correct'] for x in sub);p=correct/n;z=1.95996398454
        center=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        by[str(r)]={'correct':correct,'total':n,'answered':sum(x['answered'] for x in sub),'accuracy':p,'wilson95':[center-half,center+half],'proposed_threshold_met':p>=.95 and all(x['answered'] for x in sub),'lower_bound_ge95':center-half>=.95 and all(x['answered'] for x in sub)}
    report={'participant':response['participant'],'form':response['form'],'complete':all(x['answered'] for x in rows),'by_resolution':by,'note':'Per-participant development pilot; not a population guarantee.'}
    save(Path(path).with_suffix('.scored.json'),report)
    with Path(path).with_suffix('.scored.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--score',type=Path);a=p.parse_args()
    score(a.score) if a.score else generate()
