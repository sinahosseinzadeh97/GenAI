import React,{useState,useEffect,useRef} from 'react';
import './index.css';
import{checkHealth,queryLex,checkVigenza,risk231,analyzeContratto}from'./api.js';

const NAV=['Ricerca','Vigenza','Contratti','231','Massimario'];
const SETTORI=['Bancario','Assicurativo','Sanitario','PA','Manifatturiero','Tech','Altro'];
const COT_LABELS=['Analisi','Ricerca','Ragionamento','Conclusione'];
const cc=p=>p>=70?'#2A6B45':p>=50?'#8B6914':'#6B2A2A';

function Sk(){return<><div className="skel" style={{width:'90%'}}/><div className="skel" style={{width:'70%'}}/><div className="skel" style={{width:'80%'}}/></>}

function Empty({t}){
  return(
    <div className="empty">
      <div className="empty-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div className="empty-txt">{t}</div>
    </div>
  );
}

function Err({m}){
  return m?(
    <div className="banner-danger mt">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{verticalAlign:'middle',marginRight:6}}>
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      {m}
    </div>
  ):null;
}

function Ricerca({onReqId}){
  const[q,setQ]=useState('');const[col,setCol]=useState('lexreview_docs');
  const[load,setLoad]=useState(false);const[res,setRes]=useState(null);
  const[err,setErr]=useState(null);const[open,setOpen]=useState({});const ref=useRef();
  useEffect(()=>{const h=e=>{if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();ref.current?.focus();}};window.addEventListener('keydown',h);return()=>window.removeEventListener('keydown',h);},[]);
  const go=async e=>{e?.preventDefault();if(!q.trim())return;setLoad(true);setRes(null);setErr(null);
    try{const d=await queryLex(q,col);setRes(d);onReqId?.(d.request_id);}catch(e){setErr(e.message);}finally{setLoad(false);}};
  const pct=res?Math.round((res.confidence_score||0)*100):0;
  const showConf=res&&pct>0;
  return(
    <div className="view">
      <h1 className="hero-title">Ricerca Giuridica</h1>
      <p className="hero-sub">Normattiva · Cassazione · TAR · EUR-Lex</p>
      <form className="search-wrap" onSubmit={go}>
        <input ref={ref} className="search-input" value={q} onChange={e=>setQ(e.target.value)} placeholder="Poni una questione giuridica… (⌘K)"/>
        <button className="search-btn" type="submit" disabled={load||!q.trim()}>{load?'…':'Cerca'}</button>
      </form>
      <div style={{marginTop:14,display:'flex',alignItems:'center',gap:8}}>
        <span className="section-label" style={{marginBottom:0}}>Raccolta:</span>
        <input className="field" style={{width:220}} value={col} onChange={e=>setCol(e.target.value)}/>
      </div>
      <hr className="divider"/>
      <Err m={err}/>
      {load&&<div className="card"><Sk/></div>}
      {!load&&!res&&!err&&<Empty t='"Ubi societas, ibi ius"'/>}
      {!load&&res&&<div className="fu">
        <div className="banner-warn">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{verticalAlign:'middle',marginRight:6}}>
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          Supervisione umana richiesta — verificare con un professionista abilitato ai sensi dell'art. 2 L.247/2012
        </div>
        {showConf&&<div className="card mt">
          <div className="section-label">Affidabilità euristica</div>
          <div style={{display:'flex',justifyContent:'space-between',marginTop:4}}>
            <span style={{fontSize:12,color:'var(--muted)'}}>Score</span>
            <span className="mono" style={{fontSize:12,color:cc(pct)}}>{pct}%</span>
          </div>
          <div className="conf-track"><div className="conf-fill" style={{width:`${pct}%`,background:cc(pct)}}/></div>
          <div style={{marginTop:8,fontSize:11,color:'var(--muted)'}}>
            §&nbsp;<span className="mono" style={{color:'var(--gold)'}}>{res.request_id}</span>
          </div>
        </div>}
        {res.answer&&<div className="card mt" style={{borderLeft:'2px solid var(--gold)'}}>
          <div className="section-label" style={{marginBottom:8}}>Risposta</div>
          <p style={{fontSize:14,lineHeight:1.8,color:'var(--cream)'}}>{res.answer}</p>
        </div>}
        {res.reasoning_steps?.length>0&&<div className="mt">
          <div className="section-smallcaps">Percorso Argomentativo</div>
          {res.reasoning_steps.map((s,i)=><div key={i} className="cot-step fu" style={{animationDelay:`${i*150}ms`}}>
            <div className="cot-hd" onClick={()=>setOpen(o=>({...o,[i]:!o[i]}))}>
              <span style={{color:'var(--gold)',fontSize:12,fontVariant:'small-caps',letterSpacing:'.04em'}}>{COT_LABELS[i]||`Fase ${i+1}`}</span>
              <span style={{flex:1,fontSize:13,color:'var(--text)'}}>{(typeof s==='string'?s:JSON.stringify(s)).slice(0,90)}…</span>
              <span style={{color:'var(--muted)',fontSize:11}}>{open[i]?'›':''}{!open[i]?'›':''}{open[i]?'▲':'▼'}</span>
            </div>
            {open[i]&&<div className="cot-body">{typeof s==='string'?s:JSON.stringify(s,null,2)}</div>}
          </div>)}
        </div>}
        {res.citations?.length>0&&<div className="mt">
          <div className="section-label">Citazioni ({res.citations.length})</div>
          {res.citations.map((c,i)=>{
            const score=c.score!=null?Math.round(c.score*100):null;
            return(
              <div key={i} className="cite">
                <div className="cite-fonte">{c.fonte||c.source||`Fonte ${i+1}`}</div>
                <div className="cite-title">{c.title||c.snippet?.slice(0,120)||'—'}</div>
                {score!=null&&<div className="cite-score-track"><div className="cite-score-fill" style={{width:`${score}%`}}/></div>}
                <div className="cite-meta">
                  {score!=null&&<span>{score}% rilevanza</span>}
                  {c.date&&<span style={{marginLeft:score!=null?12:0}}>{c.date}</span>}
                </div>
              </div>
            );
          })}
        </div>}
        {res.citations?.length===0&&<div style={{marginTop:12,color:'var(--muted)',fontSize:13,fontStyle:'italic'}}>Nessuna citazione trovata.</div>}
      </div>}
    </div>
  );
}

function Vigenza(){
  const[norma,setNorma]=useState('');const[dt,setDt]=useState(new Date().toISOString().split('T')[0]);
  const[load,setLoad]=useState(false);const[res,setRes]=useState(null);const[err,setErr]=useState(null);
  const go=async e=>{e.preventDefault();if(!norma.trim())return;setLoad(true);setRes(null);setErr(null);
    try{setRes(await checkVigenza(norma,dt));}catch(e){setErr(e.message);}finally{setLoad(false);}};
  const items=res?[{date:res.data_entrata_vigore,label:'Entrata in vigore',color:'#2A6B45'},...(res.modifiche||[]).map(m=>({date:m.data,label:m.descrizione,color:'var(--gold)'})),...(res.abrogazione?[{date:res.abrogazione,label:'Abrogazione',color:'#6B2A2A'}]:[])]:[];
  return(
    <div className="view">
      <h1 className="hero-title">Verifica Vigenza</h1>
      <p className="hero-sub">Controllo validità normativa alla data indicata</p>
      <hr className="divider"/>
      <div className="card">
        <form onSubmit={go} style={{display:'flex',flexDirection:'column',gap:14}}>
          <div><div className="field-label">Norma da verificare</div><input className="field" value={norma} onChange={e=>setNorma(e.target.value)} placeholder="Es. Art. 1218 Codice Civile, D.Lgs. 231/2001…"/></div>
          <div><div className="field-label">Data di riferimento</div><input type="date" className="field" style={{width:'auto'}} value={dt} onChange={e=>setDt(e.target.value)}/></div>
          <button className="btn" type="submit" disabled={load||!norma.trim()}>{load?'Verifica in corso…':'Verifica Vigenza'}</button>
        </form>
      </div>
      <Err m={err}/>
      {load&&<div className="card mt"><Sk/></div>}
      {!load&&!res&&!err&&<Empty t='"Dura lex, sed lex"'/>}
      {!load&&res&&<div className="fu mt">
        <div className="card" style={{display:'flex',alignItems:'center',gap:16}}>
          {/* Colored circle — no emoji */}
          <div style={{
            width:48,height:48,borderRadius:'50%',flexShrink:0,
            background:res.vigente?'rgba(42,107,69,.15)':'rgba(107,42,42,.15)',
            border:`2px solid ${res.vigente?'#2A6B45':'#6B2A2A'}`,
            display:'flex',alignItems:'center',justifyContent:'center'
          }}>
            <div style={{width:14,height:14,borderRadius:'50%',background:res.vigente?'#60D090':'#D47070'}}/>
          </div>
          <div>
            <div style={{fontFamily:'Cormorant Garamond,serif',fontSize:22,color:res.vigente?'#60D090':'#D47070',fontWeight:600}}>
              {res.vigente?'Vigente':'Abrogata'}
            </div>
            <div style={{fontSize:12,color:'var(--muted)',marginTop:3}}>Fonte: {res.fonte} — verificata al {dt}</div>
          </div>
        </div>
        {items.length>0&&<div className="card mt">
          <div className="section-label" style={{marginBottom:14}}>Timeline normativa</div>
          <div className="tl">{items.map((it,i)=><div key={i} className="tl-item fu" style={{animationDelay:`${i*100}ms`}}>
            <div className="tl-dot" style={{borderColor:it.color}}/>
            <span className="mono" style={{fontSize:12,color:'var(--gold)',flexShrink:0}}>{it.date}</span>
            <span style={{fontSize:13}}>{it.label}</span>
          </div>)}</div>
        </div>}
      </div>}
    </div>
  );
}

function Contratti(){
  const[file,setFile]=useState(null);const[over,setOver]=useState(false);
  const[load,setLoad]=useState(false);const[res,setRes]=useState(null);const[err,setErr]=useState(null);
  const pick=f=>{if(f?.type==='application/pdf')setFile(f);};
  const go=async()=>{if(!file)return;setLoad(true);setRes(null);setErr(null);
    try{setRes(await analyzeContratto(file));}catch(e){setErr(e.message);}finally{setLoad(false);}};
  const emojiToNum=s=>({'🟢':1,'🟡':2,'🟠':3,'🔴':4}[s]||3);
  const rCol=s=>{const n=typeof s==='number'?s:emojiToNum(s);return n<=1?'#2A6B45':n<=2?'#8B6914':n<=3?'#C47A1E':'#6B2A2A'};
  const rLbl=s=>{const n=typeof s==='number'?s:emojiToNum(s);return['','Basso','Medio-Basso','Medio-Alto','Alto'][Math.min(n,4)]};
  return(
    <div className="view">
      <h1 className="hero-title">Analisi Contratto</h1>
      <p className="hero-sub">Identificazione clausole a rischio e riferimenti normativi</p>
      <hr className="divider"/>
      <div className={`drop${over?' over':''}`}
        onDragOver={e=>{e.preventDefault();setOver(true);}} onDragLeave={()=>setOver(false)}
        onDrop={e=>{e.preventDefault();setOver(false);pick(e.dataTransfer.files[0]);}}
        onClick={()=>document.getElementById('pf').click()}>
        <input id="pf" type="file" accept=".pdf" style={{display:'none'}} onChange={e=>pick(e.target.files[0])}/>
        <div className="drop-icon">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/>
          </svg>
        </div>
        {file?<><div style={{color:'var(--cream)',fontWeight:500}}>{file.name}</div><div style={{fontSize:12,color:'var(--muted)',marginTop:4}}>{(file.size/1024).toFixed(0)} KB — PDF</div></>
             :<><div style={{color:'var(--cream)'}}>Trascina il contratto PDF qui</div><div style={{fontSize:12,color:'var(--muted)',marginTop:4}}>oppure fai clic — max 50 MB</div></>}
      </div>
      {file&&<button className="btn mt" onClick={go} disabled={load}>{load?'Analisi in corso…':'Avvia Analisi Clausole'}</button>}
      <Err m={err}/>
      {load&&<div className="card mt"><div style={{color:'var(--muted)',fontSize:13,marginBottom:8}}>Analisi AI delle clausole contrattuali…</div><Sk/></div>}
      {!load&&!res&&!err&&!file&&<Empty t='"Pacta sunt servanda"'/>}
      {!load&&res&&<div className="fu mt">
        <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:10}}>
          <div className="section-label">{([...(res.clausole_vessatorie||[]),...(res.clausole_nulle||[])]).length} clausole analizzate</div>
        </div>
        <div className="tbl-wrap"><table>
          <thead><tr><th>Clausola</th><th>Tipo</th><th>Rischio</th><th>Rif. Normativo</th><th>Suggerimento</th></tr></thead>
          <tbody>{([...(res.clausole_vessatorie||[]),...(res.clausole_nulle||[])]).map((cl,i)=><tr key={i}>
            <td style={{maxWidth:200,color:'var(--text)'}}>{cl.testo}</td>
            <td style={{color:'var(--muted)',whiteSpace:'nowrap'}}>{cl.tipo}</td>
            <td><span style={{color:rCol(cl.risk_score),fontWeight:600}}>{rLbl(cl.risk_score)}</span></td>
            <td className="mono" style={{color:'var(--gold)',whiteSpace:'nowrap'}}>{cl.riferimento}</td>
            <td style={{color:'var(--muted)',maxWidth:220}}>{cl.suggerimento}</td>
          </tr>)}</tbody>
        </table></div>
      </div>}
    </div>
  );
}

function View231(){
  const[settore,setSettore]=useState('');const[desc,setDesc]=useState('');
  const[load,setLoad]=useState(false);const[res,setRes]=useState(null);const[err,setErr]=useState(null);
  const go=async e=>{e.preventDefault();if(!settore)return;setLoad(true);setRes(null);setErr(null);
    try{setRes(await risk231(settore,desc));}catch(e){setErr(e.message);}finally{setLoad(false);}};
  const score=res?.risk_score||0;const C=2*Math.PI*45;const off=C*(1-score/100);
  const gc=score>=70?'#6B2A2A':score>=40?'#8B6914':'#2A6B45';
  const pCol=p=>({Alta:'#D47070',Media:'#C9A84C',Bassa:'#60D090'}[p]||'var(--muted)');
  return(
    <div className="view">
      <h1 className="hero-title">Compliance D.Lgs. 231/2001</h1>
      <p className="hero-sub">Risk assessment enti e reati presupposto</p>
      <hr className="divider"/>
      <div className="card">
        <form onSubmit={go} style={{display:'flex',flexDirection:'column',gap:14}}>
          <div><div className="field-label">Settore di attività</div>
            <select className="field" value={settore} onChange={e=>setSettore(e.target.value)}>
              <option value="">Seleziona il settore…</option>
              {SETTORI.map(s=><option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div><div className="field-label">Descrizione attività (opzionale)</div>
            <textarea className="field" rows={3} value={desc} onChange={e=>setDesc(e.target.value)} style={{resize:'vertical'}} placeholder="Descrivere brevemente l'attività principale dell'ente…"/>
          </div>
          <button className="btn" type="submit" disabled={load||!settore}>{load?'Analisi in corso…':'Avvia Risk Assessment 231'}</button>
        </form>
      </div>
      <Err m={err}/>
      {load&&<div className="card mt"><Sk/></div>}
      {!load&&!res&&!err&&<Empty t='"Nemo iudex in causa sua"'/>}
      {!load&&res&&<div className="fu mt">
        <div className="card" style={{display:'flex',alignItems:'center',gap:24}}>
          <svg width={110} height={110} viewBox="0 0 110 110">
            <circle cx={55} cy={55} r={45} fill="none" stroke="var(--bg4)" strokeWidth={10}/>
            <circle cx={55} cy={55} r={45} fill="none" stroke={gc} strokeWidth={10}
              strokeDasharray={C} strokeDashoffset={off} strokeLinecap="round"
              transform="rotate(-90 55 55)" style={{transition:'stroke-dashoffset 1s ease'}}/>
            <text x={55} y={50} textAnchor="middle" fill="var(--cream)" fontSize={22} fontWeight={700} fontFamily="DM Sans">{score}</text>
            <text x={55} y={66} textAnchor="middle" fill="var(--muted)" fontSize={9} fontFamily="DM Sans">RISK SCORE</text>
          </svg>
          <div>
            <div style={{fontFamily:'Cormorant Garamond,serif',fontSize:22,color:gc}}>{score>=70?'Rischio Elevato':score>=40?'Rischio Moderato':'Rischio Contenuto'}</div>
            <div style={{fontSize:12,color:'var(--muted)',marginTop:4}}>Settore: <strong style={{color:'var(--gold)'}}>{settore}</strong></div>
            <div style={{fontSize:12,color:'var(--muted)',marginTop:2}}>{(res.reati_presupposto||[]).length} reati presupposto identificati</div>
          </div>
        </div>
        {(res.reati_presupposto||[]).length>0&&<div className="card mt">
          <div className="section-label" style={{marginBottom:12}}>Reati Presupposto</div>
          {(res.reati_presupposto||[]).map((r,i)=><div key={i} style={{display:'flex',alignItems:'center',gap:12,padding:'10px 12px',background:'var(--bg3)',marginBottom:8,border:'1px solid var(--border)',color:'var(--text)'}}>
            <span className="mono" style={{color:'var(--gold)',flexShrink:0,fontSize:12}}>{r.codice}</span>
            <span style={{flex:1,fontSize:13}}>{r.descrizione}</span>
            <span style={{color:pCol(r.probabilita),fontSize:12,flexShrink:0}}>{r.probabilita}</span>
            <span style={{color:'var(--muted)',fontSize:11,flexShrink:0}}>{r.sanzione}</span>
          </div>)}
        </div>}
        {(res.odv_raccomandazioni||[]).length>0&&<div className="card mt">
          <div className="section-label" style={{marginBottom:12}}>Raccomandazioni OdV</div>
          <ul style={{listStyle:'none',display:'flex',flexDirection:'column',gap:8}}>
            {(res.odv_raccomandazioni||[]).map((r,i)=><li key={i} style={{display:'flex',gap:10,fontSize:13}}>
              <span style={{flexShrink:0,width:20,height:20,borderRadius:'50%',background:'rgba(201,168,76,.08)',border:'1px solid rgba(201,168,76,.3)',color:'var(--gold)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:11}}>{i+1}</span>
              <span>{r}</span>
            </li>)}
          </ul>
        </div>}
      </div>}
    </div>
  );
}

function Massimario(){
  return(
    <div className="view">
      <div className="empty" style={{paddingTop:100}}>
        <div className="empty-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round">
            <path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/>
          </svg>
        </div>
        <div className="empty-txt">Massimario Automatico</div>
        <p style={{fontSize:13,color:'var(--muted)',marginTop:12}}>Generazione massime giurisprudenziali — <em style={{color:'var(--gold)'}}>in sviluppo</em></p>
      </div>
    </div>
  );
}

export default function App(){
  const[view,setView]=useState('Ricerca');
  const[online,setOnline]=useState(null);
  const[reqId,setReqId]=useState(null);
  const[noBackend,setNoBackend]=useState(false);

  useEffect(()=>{
    checkHealth().then(()=>{setOnline(true);setNoBackend(false);}).catch(()=>{setOnline(false);setNoBackend(true);});
  },[]);

  const views={Ricerca:<Ricerca onReqId={setReqId}/>,Vigenza:<Vigenza/>,Contratti:<Contratti/>,'231':<View231/>,Massimario:<Massimario/>};

  return(<>
    <div className="shell">
      {noBackend&&<div className="banner-amber" style={{borderRadius:0,borderLeft:'none',borderRight:'none',borderTop:'none',position:'fixed',top:64,left:0,right:0,zIndex:999}}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{verticalAlign:'middle',marginRight:6}}>
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        Backend non raggiungibile — avviare il server su porta 8000
      </div>}
      <nav className="top">
        <div className="logo"><div className="monogram">RF</div><span className="brand-name">RAGForge Italia</span></div>
        <div className="nav-links">
          {NAV.map((n,i)=><button key={n} className={`nl${view===n?' active':''}`} onClick={()=>setView(n)} style={{animationDelay:`${i*50}ms`}}>{n}</button>)}
        </div>
        <div className="nav-end">
          <div className={`dot${online===true?' on':online===false?' off':''}`}/>
          <span className="sys-label">{online===true?'Sistema Attivo':online===false?'Offline':'Controllo…'}</span>
          {reqId&&<span className="req-badge">{reqId}</span>}
        </div>
      </nav>
      <main style={{flex:1,overflowY:'auto'}}>
        {views[view]||views.Ricerca}
      </main>
      <footer className="bar">
        <div className="footer-left">
          <span>Studio Legale</span>
          <span className="footer-sep">·</span>
          <span>Piattaforma Riservata</span>
          <span className="footer-sep">·</span>
          <span>© 2026 RAGForge Italia</span>
        </div>
        <span style={{fontSize:11,color:'var(--muted)',fontStyle:'italic',maxWidth:420,textAlign:'right'}}>
          Le risposte hanno carattere informativo e non costituiscono parere legale ai sensi dell'art. 2 L.247/2012
        </span>
      </footer>
    </div>
  </>);
}
