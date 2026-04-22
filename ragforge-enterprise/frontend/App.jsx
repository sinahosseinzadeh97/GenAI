
const { useState, useEffect, useRef, useCallback, Fragment } = React;

// ── Design tokens ──────────────────────────────────────────────────────────
const T = {
  navy900:'#0A1628', navy800:'#111D35', navy700:'#162240',
  navy600:'#1E2D4A', navy500:'#263655', gold:'#C9A84C', goldLight:'#E2C97A',
  textPrimary:'#E8EAF0', textMuted:'#7A8BA8', textSubtle:'#4A5A72',
  success:'#2A7A4B', danger:'#8B2020', warning:'#8B6914',
};

// ── Collections ────────────────────────────────────────────────────────────
const COLLECTIONS = [
  { id:'normattiva', label:'Normattiva — Diritto Vigente' },
  { id:'cassazione', label:'Cassazione Civile e Penale' },
  { id:'tar', label:'TAR e Consiglio di Stato' },
  { id:'corte_cost', label:'Corte Costituzionale' },
  { id:'eur_lex', label:'EUR-Lex — Diritto UE' },
  { id:'all', label:'Tutte le Fonti' },
];

const MATERIE = [
  'Diritto Civile','Diritto Penale','Diritto del Lavoro',
  'Diritto Tributario','GDPR/Privacy','Appalti Pubblici',
  'Diritto Societario','231',
];

const SETTORI_231 = [
  'Bancario','Assicurativo','Sanitario','PA','Manifatturiero','Tech','Altro',
];

const COT_PHASES = [
  { key:'COMPRENSIONE',   label:'Comprensione della Questione', icon:'🔍' },
  { key:'NORME',         label:'Norme Applicabili',            icon:'📜' },
  { key:'GIURISPRUDENZA',label:'Giurisprudenza Rilevante',     icon:'⚖️'  },
  { key:'RAGIONAMENTO',  label:'Ragionamento Giuridico',       icon:'🧠' },
  { key:'RISPOSTA',      label:'Risposta',                     icon:'✅' },
  { key:'FONTI',         label:'Fonti Citate',                 icon:'📚' },
  { key:'AVVERTENZE',    label:'Avvertenze',                   icon:'⚠️'  },
];

const LEGAL_QUOTES = [
  '"Ubi societas, ibi ius"',''"Dura lex, sed lex"',
  '"Nemo iudex in causa sua"','"Ignorantia iuris non excusat"',
  '"In dubio pro reo"',
];

// ── Mock API ────────────────────────────────────────────────────────────────
const delay = ms => new Promise(r => setTimeout(r, ms));

const MOCK = {
  async query(payload) {
    await delay(2200);
    return {
      request_id: 'REQ-' + Math.random().toString(36).slice(2,10).toUpperCase(),
      requires_human_review: true,
      confidence_score: 0.74,
      cot: {
        COMPRENSIONE: 'La questione verte sulla responsabilità contrattuale ex art. 1218 c.c. e sulla ripartizione dell\'onere probatorio tra le parti.',
        NORME: 'Art. 1218 c.c. (responsabilità del debitore), art. 2697 c.c. (onere della prova), art. 1175 c.c. (buona fede oggettiva). Norma vigente al ' + new Date().toLocaleDateString('it-IT') + '.',
        GIURISPRUDENZA: 'Cass. Civ. Sez. III, 30 ottobre 2001 n. 13533 (sentenza cardine sull\'inversione dell\'onere). Cass. Civ. Sez. Unite 2022/26279.',
        RAGIONAMENTO: 'Applicando il principio enunciato da Cass. 13533/2001, il creditore deve provare solo il titolo contrattuale. L\'inadempimento si presume e spetta al debitore provare il fatto estintivo.',
        RISPOSTA: 'Il debitore è responsabile ex art. 1218 c.c. salvo provi che l\'inadempimento è dovuto a causa a lui non imputabile. Il creditore deve allegare il contratto e l\'inadempimento; il debitore deve provare la causa esimente.',
        FONTI: ['Art. 1218 Codice Civile','Cass. Sez. Unite 13533/2001','Art. 2697 c.c.'],
        AVVERTENZE: 'La presente analisi ha carattere puramente informativo e non costituisce parere legale ai sensi dell\'art. 2 L. 247/2012. Consultare un professionista abilitato per casi specifici.',
      },
      sources: [
        { fonte:'Normattiva', numero_atto:'R.D. 262/1942', articolo:'Art. 1218', data_vigenza:'1942-03-16', snippet:'Il debitore che non esegue esattamente la prestazione dovuta è tenuto al risarcimento del danno...' },
        { fonte:'Cassazione Civile', numero_atto:'Sent. 13533/2001', articolo:'Motivazione §3', data_vigenza:'2001-10-30', snippet:'Il creditore che agisce per la risoluzione contrattuale deve soltanto provare la fonte negoziale...' },
        { fonte:'Normattiva', numero_atto:'R.D. 262/1942', articolo:'Art. 2697', data_vigenza:'1942-03-16', snippet:'Chi vuol far valere un diritto in giudizio deve provare i fatti che ne costituiscono il fondamento...' },
      ],
    };
  },
  async vigenza(payload) {
    await delay(1400);
    return {
      norma: payload.norma,
      vigente: true,
      data_entrata_vigore: '1942-03-16',
      modifiche: [
        { data:'1994-01-01', descrizione:'Modificato da D.Lgs. 385/1993 art. 161' },
        { data:'2005-06-09', descrizione:'Integrato da D.Lgs. 206/2005 art. 142' },
        { data:'2021-11-26', descrizione:'Aggiornato da D.L. 152/2021' },
      ],
      abrogazione: null,
      fonte: 'Normattiva',
    };
  },
  async contratto(file) {
    await delay(2800);
    return {
      clauses: [
        { testo:'Il presente contratto si rinnova automaticamente salvo disdetta con preavviso di 90 giorni.', tipo:'Rinnovo automatico', risk_score: 3, riferimento:'Art. 1469-bis c.c.', suggerimento:'Ridurre il preavviso a 30 giorni o prevedere rinnovo su consenso esplicito.' },
        { testo:'Il fornitore è esonerato da ogni responsabilità per danni indiretti o consequenziali.', tipo:'Limitazione responsabilità', risk_score: 4, riferimento:'Art. 1229 c.c.', suggerimento:'Clausola potenzialmente nulla per dolo/colpa grave. Specificare limiti entro quelli di legge.' },
        { testo:'Le controversie sono devolute alla giurisdizione esclusiva del Foro di Milano.', tipo:'Foro esclusivo', risk_score: 1, riferimento:'Art. 28 c.p.c.', suggerimento:'Conforme alla normativa vigente. Nessuna modifica necessaria.' },
        { testo:'Il cliente acconsente al trattamento dei dati per finalità di marketing.', tipo:'Consenso GDPR', risk_score: 3, riferimento:'Art. 7 GDPR', suggerimento:'Il consenso deve essere granulare e revocabile. Prevedere opt-in distinto per ogni finalità.' },
        { testo:'Il pagamento deve avvenire entro 90 giorni dalla fattura.', tipo:'Termini di pagamento', risk_score: 4, riferimento:'D.Lgs. 231/2002 art. 4', suggerimento:'Termine superiore ai 60 gg legali nelle transazioni B2B. Rischio di nullità.' },
      ],
    };
  },
  async risk231(payload) {
    await delay(1800);
    return {
      risk_score: 67,
      reati_presupposto: [
        { codice:'Art. 24', descrizione:'Frode ai danni dello Stato', probabilita:'Media', sanzione:'Fino a 500 quote' },
        { codice:'Art. 25', descrizione:'Concussione e corruzione', probabilita:'Alta', sanzione:'Da 200 a 800 quote' },
        { codice:'Art. 25-octies', descrizione:'Ricettazione e riciclaggio', probabilita:'Bassa', sanzione:'Fino a 200 quote' },
        { codice:'Art. 25-ter', descrizione:'Reati societari', probabilita:'Media', sanzione:'Da 100 a 400 quote' },
      ],
      odv_raccomandazioni: [
        'Istituire o rafforzare l\'Organismo di Vigilanza con adeguati poteri autonomi di spesa.',
        'Implementare procedure di whistleblowing conformi a D.Lgs. 24/2023.',
        'Aggiornare il Modello 231 con focus sui processi a rischio specifico del settore ' + payload.settore + '.',
        'Prevedere audit periodici delle aree sensibili con frequenza almeno semestrale.',
        'Formare il personale sui reati presupposto pertinenti con registro delle presenze.',
      ],
    };
  },
};

// ── Utility ────────────────────────────────────────────────────────────────
function riskColor(score) {
  if (score <= 1) return { text:'text-emerald-400', bg:'bg-emerald-900/30', label:'Basso', emoji:'🟢' };
  if (score <= 2) return { text:'text-yellow-400', bg:'bg-yellow-900/30', label:'Medio-Basso', emoji:'🟡' };
  if (score <= 3) return { text:'text-orange-400', bg:'bg-orange-900/30', label:'Medio-Alto', emoji:'🟠' };
  return { text:'text-red-400', bg:'bg-red-900/30', label:'Alto', emoji:'🔴' };
}

function probColor(prob) {
  return { Alta:'text-red-400', Media:'text-orange-400', Bassa:'text-emerald-400' }[prob] || 'text-gray-400';
}

let reqIdCounter = 0;
function newReqId() { return 'REQ-' + (++reqIdCounter).toString().padStart(6,'0'); }

// ── Atom components ────────────────────────────────────────────────────────
function Skeleton({ w='100%', h='1rem', className='' }) {
  return <div className={`skeleton rounded ${className}`} style={{ width:w, height:h }} />;
}

function GoldBadge({ children, className='' }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium border border-yellow-600/40 bg-yellow-900/20 text-yellow-400 ${className}`}>
      {children}
    </span>
  );
}

function DangerBanner({ children }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-red-800/50 bg-red-950/30 text-red-300 text-sm">
      <span className="text-base">⚠️</span>
      <span>{children}</span>
    </div>
  );
}

function SuccessPill({ children }) {
  return <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs bg-emerald-900/40 border border-emerald-700/40 text-emerald-400">{children}</span>;
}

function Card({ children, className='' }) {
  return (
    <div className={`rounded-xl border border-navy-600 bg-navy-800 ${className}`}
         style={{ borderColor:'#1E2D4A', backgroundColor:'#111D35' }}>
      {children}
    </div>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="#C9A84C" strokeWidth="4"/>
      <path className="opacity-75" fill="#C9A84C" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
    </svg>
  );
}

function ConfidenceMeter({ score }) {
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? '#2A7A4B' : pct >= 50 ? '#8B6914' : '#8B2020';
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs" style={{ color:'#7A8BA8' }}>Affidabilità (euristica)</span>
      <div className="flex-1 h-2 rounded-full" style={{ background:'#1E2D4A' }}>
        <div className="h-2 rounded-full transition-all duration-700"
             style={{ width:`${pct}%`, background:color }} />
      </div>
      <span className="text-xs font-mono font-medium" style={{ color }}>{pct}%</span>
    </div>
  );
}

function AuditTrail({ reqId }) {
  return (
    <div className="flex items-center gap-1.5 text-xs" style={{ color:'#7A8BA8' }}>
      <span>🔒</span>
      <span>Query registrata — ID: <span className="font-mono" style={{ color:'#C9A84C' }}>{reqId}</span></span>
    </div>
  );
}

function EmptyState({ quote }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-4">
      <div className="text-5xl opacity-20">⚖️</div>
      <blockquote className="text-center italic" style={{ color:'#4A5A72', fontFamily:'"Playfair Display", serif', fontSize:'1.1rem' }}>
        {quote || LEGAL_QUOTES[Math.floor(Math.random() * LEGAL_QUOTES.length)]}
      </blockquote>
      <p className="text-xs" style={{ color:'#4A5A72' }}>Inserisci una ricerca per iniziare</p>
    </div>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id:'ricerca',    label:'Ricerca',    icon:'🔍' },
  { id:'vigenza',   label:'Vigenza',    icon:'📅' },
  { id:'contratti', label:'Contratti',  icon:'📄' },
  { id:'231',       label:'231',        icon:'🛡️' },
  { id:'massimario',label:'Massimario', icon:'📋' },
  { id:'settings',  label:'Impostazioni',icon:'⚙️' },
];

const SOURCE_BADGES = [
  { label:'Normattiva', color:'#1A5A8B' },
  { label:'Cassazione', color:'#5A1A1A' },
  { label:'TAR',        color:'#1A5A3A' },
  { label:'Corte Cost.',color:'#5A4A1A' },
];

function Sidebar({ view, setView }) {
  return (
    <aside className="flex flex-col flex-shrink-0 h-full" style={{ width:240, background:'#0A1628', borderRight:'1px solid #1E2D4A' }}>
      {/* Logo */}
      <div className="px-5 pt-6 pb-4" style={{ borderBottom:'1px solid #1E2D4A' }}>
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
               style={{ background:'linear-gradient(135deg,#C9A84C,#8B6914)', color:'#0A1628' }}>RF</div>
          <div>
            <div className="font-semibold text-sm leading-tight" style={{ fontFamily:'"Playfair Display",serif', color:'#E8EAF0' }}>RAGForge</div>
            <div className="text-xs leading-tight" style={{ color:'#C9A84C', fontFamily:'"Playfair Display",serif', fontStyle:'italic' }}>Italia</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-4 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(item => (
          <button key={item.id}
            onClick={() => setView(item.id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 text-left
              ${view === item.id ? 'nav-item-active' : 'hover:bg-navy-700/50'}`}
            style={{ color: view === item.id ? '#C9A84C' : '#7A8BA8', fontWeight: view === item.id ? 600 : 400,
                     borderLeft: view === item.id ? '2px solid #C9A84C' : '2px solid transparent' }}>
            <span className="text-base leading-none">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Source badges */}
      <div className="px-4 py-4" style={{ borderTop:'1px solid #1E2D4A' }}>
        <p className="text-xs mb-2" style={{ color:'#4A5A72' }}>Fonti integrate</p>
        <div className="flex flex-wrap gap-1.5">
          {SOURCE_BADGES.map(b => (
            <span key={b.label} className="text-xs px-2 py-0.5 rounded"
                  style={{ background:b.color+'33', color:b.color === '#1A5A8B' ? '#60A0D0' : b.color === '#5A1A1A' ? '#C06060' : b.color === '#1A5A3A' ? '#60B080' : '#C0A060', border:`1px solid ${b.color}55` }}>
              {b.label}
            </span>
          ))}
        </div>
      </div>
    </aside>
  );
}

// ── Top Bar ────────────────────────────────────────────────────────────────
function TopBar({ collection, setCollection, reqId }) {
  return (
    <header className="flex items-center gap-4 px-6 py-3 flex-shrink-0"
            style={{ background:'#111D35', borderBottom:'1px solid #1E2D4A', minHeight:56 }}>
      {/* Collection selector */}
      <div className="relative flex-shrink-0">
        <select value={collection} onChange={e => setCollection(e.target.value)}
                className="pl-3 pr-8 py-1.5 rounded-lg text-sm cursor-pointer"
                style={{ background:'#162240', border:'1px solid #1E2D4A', color:'#E8EAF0', minWidth:220 }}>
          {COLLECTIONS.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
        </select>
        <span className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color:'#C9A84C', fontSize:10 }}>▼</span>
      </div>

      {/* AI Act badge */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs"
           style={{ background:'#8B202022', border:'1px solid #8B202055', color:'#E06060' }}>
        <span>🤖</span>
        <span className="font-medium">Sistema IA ad Alto Rischio</span>
        <span className="text-red-400/60">—</span>
        <span className="italic">Supervisione Umana Richiesta</span>
      </div>

      <div className="flex-1" />

      {/* Request ID */}
      {reqId && <AuditTrail reqId={reqId} />}

      {/* Avatar */}
      <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0"
           style={{ background:'linear-gradient(135deg,#1E2D4A,#263655)', border:'1px solid #C9A84C44', color:'#C9A84C' }}>
        A
      </div>
    </header>
  );
}

// ── CoT Card ───────────────────────────────────────────────────────────────
function CotCard({ phase, content, index, visible }) {
  const [open, setOpen] = useState(index === 4); // RISPOSTA open by default
  if (!visible) return null;
  return (
    <div className="cot-card rounded-lg overflow-hidden animate-slide-in"
         style={{ background:'#162240', border:'1px solid #1E2D4A', animationDelay:`${index*80}ms` }}>
      <button onClick={() => setOpen(o => !o)}
              className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/5 transition-colors">
        <span className="text-lg leading-none">{phase.icon}</span>
        <span className="font-medium text-sm flex-1" style={{ color:'#E8EAF0', fontFamily:'"IBM Plex Sans"' }}>{phase.label}</span>
        <span className="text-xs" style={{ color:'#7A8BA8' }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="px-4 pb-4 pt-0 text-sm leading-relaxed prose-legal" style={{ color:'#BCC5D4' }}>
          {Array.isArray(content) ? (
            <ul className="space-y-1 list-none">
              {content.map((s,i) => <li key={i} className="flex items-start gap-2"><span style={{ color:'#C9A84C' }}>›</span><span>{s}</span></li>)}
            </ul>
          ) : (
            <p>{content}</p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Source Citation Card ───────────────────────────────────────────────────
function SourceCard({ src, index }) {
  return (
    <div className="rounded-lg p-3 text-sm animate-fade-in"
         style={{ background:'#0A1628', border:'1px solid #1E2D4A', animationDelay:`${index*60}ms` }}>
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div className="flex items-center gap-2">
          <span className="font-medium" style={{ color:'#C9A84C' }}>{src.fonte}</span>
          <span className="text-xs px-1.5 py-0.5 rounded" style={{ background:'#1E2D4A', color:'#7A8BA8' }}>{src.numero_atto}</span>
          <span className="text-xs" style={{ color:'#4A5A72' }}>{src.articolo}</span>
        </div>
        <span className="text-xs flex-shrink-0" style={{ color:'#4A5A72' }}>Vigente al {src.data_vigenza}</span>
      </div>
      <p className="text-xs leading-relaxed" style={{ color:'#7A8BA8', fontStyle:'italic' }}>"{src.snippet}"</p>
    </div>
  );
}

// ── Ricerca Giuridica View ─────────────────────────────────────────────────
function RicercaView({ collection }) {
  const [query, setQuery] = useState('');
  const [materie, setMaterie] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [reqId, setReqId] = useState(null);
  const [cotVisible, setCotVisible] = useState({});
  const inputRef = useRef(null);

  // Cmd+K shortcut
  useEffect(() => {
    const handler = e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); inputRef.current?.focus(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const toggleMateria = m => setMaterie(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]);

  const handleSubmit = async e => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true); setResult(null); setCotVisible({});
    const id = newReqId();
    setReqId(id);
    try {
      const data = await MOCK.query({ query, materie, collection });
      setResult(data);
      // Animate CoT phases sequentially
      COT_PHASES.forEach((p, i) => {
        setTimeout(() => setCotVisible(prev => ({ ...prev, [p.key]: true })), i * 200 + 300);
      });
    } finally { setLoading(false); }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Search form */}
      <div className="flex-shrink-0 p-6" style={{ borderBottom:'1px solid #1E2D4A' }}>
        <h1 className="text-2xl font-semibold mb-4" style={{ fontFamily:'"Playfair Display",serif', color:'#E8EAF0' }}>
          Ricerca Giuridica
        </h1>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="relative">
            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-base" style={{ color:'#7A8BA8' }}>⚖️</span>
            <input ref={inputRef} id="search-input" value={query} onChange={e => setQuery(e.target.value)}
                   placeholder="Poni una questione giuridica... (⌘K)"
                   className="w-full pl-11 pr-24 py-3.5 rounded-xl text-sm transition-all duration-200"
                   style={{ background:'#162240', border:'1px solid #1E2D4A', color:'#E8EAF0', outline:'none',
                            boxShadow: query ? '0 0 0 2px rgba(201,168,76,0.25)' : 'none' }}
                   onFocus={e => e.target.style.borderColor='#C9A84C'}
                   onBlur={e => e.target.style.borderColor='#1E2D4A'} />
            <button type="submit" disabled={loading || !query.trim()}
                    className="absolute right-3 top-1/2 -translate-y-1/2 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150"
                    style={{ background: loading || !query.trim() ? '#1E2D4A' : '#C9A84C',
                             color: loading || !query.trim() ? '#4A5A72' : '#0A1628' }}>
              {loading ? '...' : 'Cerca'}
            </button>
          </div>
          {/* Materie chips */}
          <div className="flex flex-wrap gap-2">
            {MATERIE.map(m => (
              <button key={m} type="button" onClick={() => toggleMateria(m)}
                      className="px-3 py-1 rounded-full text-xs transition-all duration-150"
                      style={{ background: materie.includes(m) ? '#C9A84C22' : '#162240',
                               border: `1px solid ${materie.includes(m) ? '#C9A84C' : '#1E2D4A'}`,
                               color: materie.includes(m) ? '#C9A84C' : '#7A8BA8' }}>
                {m}
              </button>
            ))}
          </div>
        </form>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        {loading && (
          <div className="space-y-3">
            <div className="flex items-center gap-3 mb-4">
              <Spinner />
              <span className="text-sm" style={{ color:'#7A8BA8' }}>Analisi in corso — elaborazione CoT...</span>
            </div>
            {[...Array(4)].map((_,i) => (
              <div key={i} className="rounded-lg p-4 space-y-2" style={{ background:'#162240', border:'1px solid #1E2D4A' }}>
                <Skeleton w="40%" h="0.875rem" />
                <Skeleton w="90%" h="0.75rem" />
                <Skeleton w="75%" h="0.75rem" />
              </div>
            ))}
          </div>
        )}

        {!loading && !result && <EmptyState quote='"Ubi societas, ibi ius"' />}

        {!loading && result && (
          <div className="space-y-4 animate-fade-in">
            {/* Human review banner */}
            <DangerBanner>Supervisione umana richiesta — verificare le informazioni con un professionista abilitato prima di qualsiasi utilizzo in sede giudiziaria</DangerBanner>

            {/* Confidence + audit */}
            <Card className="p-4 space-y-3">
              <ConfidenceMeter score={result.confidence_score} />
              <AuditTrail reqId={result.request_id} />
            </Card>

            {/* CoT phases */}
            <div className="space-y-2">
              <p className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color:'#4A5A72' }}>Ragionamento a catena</p>
              {COT_PHASES.map((phase, i) => (
                <CotCard key={phase.key} phase={phase}
                         content={result.cot[phase.key]}
                         index={i}
                         visible={!!cotVisible[phase.key]} />
              ))}
            </div>

            {/* Sources */}
            {result.sources?.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color:'#4A5A72' }}>Fonti e Citazioni ({result.sources.length})</p>
                {result.sources.map((src, i) => <SourceCard key={i} src={src} index={i} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Vigenza View ───────────────────────────────────────────────────────────
function VigenzaView() {
  const [norma, setNorma] = useState('');
  const [dataRif, setDataRif] = useState(new Date().toISOString().split('T')[0]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async e => {
    e.preventDefault();
    if (!norma.trim()) return;
    setLoading(true); setResult(null);
    try { setResult(await MOCK.vigenza({ norma, data_riferimento: dataRif })); }
    finally { setLoading(false); }
  };

  const timelineItems = result ? [
    { date: result.data_entrata_vigore, label:'Entrata in vigore', icon:'🟢', color:'#2A7A4B' },
    ...(result.modifiche || []).map(m => ({ date:m.data, label:m.descrizione, icon:'🔵', color:'#1A5A8B' })),
    ...(result.abrogazione ? [{ date:result.abrogazione, label:'Abrogazione', icon:'🔴', color:'#8B2020' }] : []),
  ] : [];

  const inputStyle = { background:'#162240', border:'1px solid #1E2D4A', color:'#E8EAF0', outline:'none', borderRadius:8, padding:'10px 14px', fontSize:14, width:'100%' };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      <h1 className="text-2xl font-semibold" style={{ fontFamily:'"Playfair Display",serif', color:'#E8EAF0' }}>Verifica Vigenza</h1>

      <Card className="p-5">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color:'#7A8BA8' }}>Norma da verificare</label>
            <input value={norma} onChange={e => setNorma(e.target.value)}
                   placeholder="Es. Art. 1218 Codice Civile, D.Lgs. 231/2001..." style={inputStyle} />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color:'#7A8BA8' }}>Data di riferimento</label>
            <input type="date" value={dataRif} onChange={e => setDataRif(e.target.value)} style={{ ...inputStyle, width:'auto' }} />
          </div>
          <button type="submit" disabled={loading || !norma.trim()} className="px-5 py-2 rounded-lg text-sm font-semibold transition-all"
                  style={{ background: loading || !norma.trim() ? '#1E2D4A' : '#C9A84C', color: loading || !norma.trim() ? '#4A5A72' : '#0A1628' }}>
            {loading ? 'Verifica in corso...' : 'Verifica Vigenza'}
          </button>
        </form>
      </Card>

      {loading && (
        <Card className="p-5 space-y-3">
          <Skeleton w="60%" h="1.5rem" />
          <Skeleton w="100%" h="1rem" />
          <Skeleton w="80%" h="1rem" />
        </Card>
      )}

      {!loading && !result && <EmptyState quote='"Dura lex, sed lex"' />}

      {!loading && result && (
        <div className="space-y-4 animate-fade-in">
          <Card className="p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-12 h-12 rounded-full flex items-center justify-center text-2xl"
                   style={{ background: result.vigente ? '#2A7A4B22' : '#8B202022', border:`2px solid ${result.vigente ? '#2A7A4B' : '#8B2020'}` }}>
                {result.vigente ? '✅' : '❌'}
              </div>
              <div>
                <p className="text-lg font-semibold" style={{ fontFamily:'"Playfair Display",serif', color: result.vigente ? '#60D090' : '#E06060' }}>
                  {result.vigente ? 'Norma Vigente' : 'Norma Abrogata'}
                </p>
                <p className="text-xs" style={{ color:'#7A8BA8' }}>Fonte: {result.fonte} — verificata al {dataRif}</p>
              </div>
            </div>
          </Card>

          {/* Timeline */}
          <Card className="p-5">
            <p className="text-xs font-medium uppercase tracking-wider mb-4" style={{ color:'#4A5A72' }}>Timeline normativa</p>
            <div className="relative pl-6">
              <div className="absolute left-2 top-0 bottom-0 w-px" style={{ background:'#1E2D4A' }} />
              {timelineItems.map((item, i) => (
                <div key={i} className="relative mb-5 last:mb-0 animate-slide-in" style={{ animationDelay:`${i*100}ms` }}>
                  <div className="absolute -left-4 top-0.5 w-4 h-4 rounded-full flex items-center justify-center text-xs"
                       style={{ background:item.color+'33', border:`2px solid ${item.color}` }}>
                    <span style={{ fontSize:8 }}>●</span>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-xs font-mono flex-shrink-0" style={{ color:'#C9A84C', marginTop:1 }}>{item.date}</span>
                    <span className="text-sm" style={{ color:'#BCC5D4' }}>{item.label}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

// ── Contratto View ─────────────────────────────────────────────────────────
function ContrattoView() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const dropRef = useRef(null);

  const onDrop = e => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer?.files?.[0] || e.target.files?.[0];
    if (f && f.type === 'application/pdf') setFile(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true); setResult(null);
    try { setResult(await MOCK.contratto(file)); }
    finally { setLoading(false); }
  };

  return (
    <div className="p-6 space-y-5 overflow-y-auto h-full">
      <h1 className="text-2xl font-semibold" style={{ fontFamily:'"Playfair Display",serif', color:'#E8EAF0' }}>Analisi Contratto</h1>

      {/* Dropzone */}
      <div ref={dropRef}
           onDragOver={e => { e.preventDefault(); setDragging(true); }}
           onDragLeave={() => setDragging(false)}
           onDrop={onDrop}
           className={`rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${dragging ? 'dropzone-active' : ''}`}
           style={{ background:'#162240', border:`2px dashed ${dragging ? '#C9A84C' : '#1E2D4A'}` }}
           onClick={() => document.getElementById('pdf-input').click()}>
        <input id="pdf-input" type="file" accept=".pdf" className="hidden" onChange={onDrop} />
        <div className="text-4xl mb-3">📄</div>
        {file ? (
          <div>
            <p className="font-medium text-sm" style={{ color:'#E8EAF0' }}>{file.name}</p>
            <p className="text-xs mt-1" style={{ color:'#7A8BA8' }}>{(file.size/1024).toFixed(0)} KB — PDF</p>
          </div>
        ) : (
          <div>
            <p className="font-medium text-sm" style={{ color:'#E8EAF0' }}>Trascina il contratto PDF qui</p>
            <p className="text-xs mt-1" style={{ color:'#7A8BA8' }}>oppure fai clic per selezionare il file</p>
            <p className="text-xs mt-2" style={{ color:'#4A5A72' }}>Formati supportati: PDF — Max 50 MB</p>
          </div>
        )}
      </div>

      {file && (
        <button onClick={handleAnalyze} disabled={loading}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold transition-all"
                style={{ background: loading ? '#1E2D4A' : '#C9A84C', color: loading ? '#4A5A72' : '#0A1628' }}>
          {loading ? 'Analisi in corso...' : 'Avvia Analisi Clausole'}
        </button>
      )}

      {loading && (
        <div className="space-y-3">
          <div className="flex items-center gap-3"><Spinner /><span className="text-sm" style={{ color:'#7A8BA8' }}>Analisi AI delle clausole contrattuali...</span></div>
          {[...Array(3)].map((_,i) => <div key={i} className="h-16 rounded-lg skeleton" />)}
        </div>
      )}

      {!loading && !result && !file && <EmptyState quote='"Pacta sunt servanda"' />}

      {!loading && result && (
        <div className="space-y-3 animate-fade-in">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium uppercase tracking-wider" style={{ color:'#4A5A72' }}>
              Clausole analizzate — {result.clauses.length} trovate
            </p>
            <div className="flex items-center gap-2 text-xs" style={{ color:'#7A8BA8' }}>
              <span>🔴 Alto</span><span>🟠 Medio-Alto</span><span>🟡 Medio</span><span>🟢 Basso</span>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl" style={{ border:'1px solid #1E2D4A' }}>
            <table className="w-full text-sm">
              <thead>
                <tr style={{ background:'#162240', borderBottom:'1px solid #1E2D4A' }}>
                  {['Clausola','Tipo','Rischio','Rif. Normativo','Suggerimento'].map(h => (
                    <th key={h} className="text-left px-4 py-3 text-xs font-medium uppercase tracking-wide" style={{ color:'#7A8BA8', whiteSpace:'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.clauses.map((cl, i) => {
                  const rc = riskColor(cl.risk_score);
                  return (
                    <tr key={i} style={{ borderBottom:'1px solid #1E2D4A', background: i%2===0 ? '#111D35' : '#0F1B30' }}>
                      <td className="px-4 py-3 text-xs max-w-xs" style={{ color:'#BCC5D4' }}>
                        <span className="line-clamp-2">{cl.testo}</span>
                      </td>
                      <td className="px-4 py-3 text-xs whitespace-nowrap" style={{ color:'#7A8BA8' }}>{cl.tipo}</td>
                      <td className="px-4 py-3 text-xs whitespace-nowrap">
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full ${rc.bg} ${rc.text}`}>
                          {rc.emoji} {rc.label}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-xs whitespace-nowrap">
                        <span className="font-mono" style={{ color:'#C9A84C' }}>{cl.riferimento}</span>
                      </td>
                      <td className="px-4 py-3 text-xs max-w-xs" style={{ color:'#7A8BA8' }}>
                        <span className="line-clamp-2">{cl.suggerimento}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ── 231 View ───────────────────────────────────────────────────────────────
function View231() {
  const [settore, setSettore] = useState('');
  const [descrizione, setDescrizione] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async e => {
    e.preventDefault();
    if (!settore) return;
    setLoading(true); setResult(null);
    try { setResult(await MOCK.risk231({ settore, descrizione })); }
    finally { setLoading(false); }
  };

  const gauge = result ? (() => {
    const pct = result.risk_score / 100;
    const circumference = 2 * Math.PI * 45;
    const offset = circumference * (1 - pct);
    const color = result.risk_score >= 70 ? '#8B2020' : result.risk_score >= 40 ? '#8B6914' : '#2A7A4B';
    return { circumference, offset, color };
  })() : null;

  const selectStyle = { background:'#162240', border:'1px solid #1E2D4A', color:'#E8EAF0', outline:'none', borderRadius:8, padding:'10px 14px', fontSize:14, width:'100%' };

  return (
    <div className="p-6 space-y-5 overflow-y-auto h-full">
      <h1 className="text-2xl font-semibold" style={{ fontFamily:'"Playfair Display",serif', color:'#E8EAF0' }}>
        Compliance D.Lgs. 231/2001
      </h1>

      <Card className="p-5">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color:'#7A8BA8' }}>Settore di attività</label>
            <div className="relative">
              <select value={settore} onChange={e => setSettore(e.target.value)} style={selectStyle}>
                <option value="">Seleziona il settore...</option>
                {SETTORI_231.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <span className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color:'#C9A84C', fontSize:10 }}>▼</span>
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color:'#7A8BA8' }}>Descrizione attività (opzionale)</label>
            <textarea value={descrizione} onChange={e => setDescrizione(e.target.value)} rows={3}
                      placeholder="Descrivere brevemente l'attività principale dell'ente e i processi sensibili..."
                      style={{ ...selectStyle, resize:'vertical' }} />
          </div>
          <button type="submit" disabled={loading || !settore}
                  className="px-5 py-2.5 rounded-lg text-sm font-semibold transition-all"
                  style={{ background: loading || !settore ? '#1E2D4A' : '#C9A84C', color: loading || !settore ? '#4A5A72' : '#0A1628' }}>
            {loading ? 'Analisi in corso...' : 'Avvia Risk Assessment 231'}
          </button>
        </form>
      </Card>

      {loading && (
        <div className="space-y-3">
          <div className="flex items-center gap-3"><Spinner /><span className="text-sm" style={{ color:'#7A8BA8' }}>Valutazione rischi D.Lgs. 231/2001...</span></div>
          {[...Array(3)].map((_,i) => <div key={i} className="h-20 rounded-lg skeleton" />)}
        </div>
      )}

      {!loading && !result && <EmptyState quote='"Nemo iudex in causa sua"' />}

      {!loading && result && (
        <div className="space-y-4 animate-fade-in">
          {/* Gauge */}
          <Card className="p-5 flex items-center gap-6">
            <svg width="110" height="110" viewBox="0 0 110 110">
              <circle cx="55" cy="55" r="45" fill="none" stroke="#1E2D4A" strokeWidth="10" />
              <circle cx="55" cy="55" r="45" fill="none"
                      stroke={gauge.color} strokeWidth="10"
                      strokeDasharray={gauge.circumference}
                      strokeDashoffset={gauge.offset}
                      strokeLinecap="round"
                      transform="rotate(-90 55 55)"
                      style={{ transition:'stroke-dashoffset 1s ease-out' }} />
              <text x="55" y="50" textAnchor="middle" className="font-bold"
                    style={{ fill:'#E8EAF0', fontSize:22, fontFamily:'IBM Plex Sans', fontWeight:700 }}>{result.risk_score}</text>
              <text x="55" y="66" textAnchor="middle"
                    style={{ fill:'#7A8BA8', fontSize:9, fontFamily:'IBM Plex Sans' }}>RISK SCORE</text>
            </svg>
            <div>
              <p className="font-semibold text-lg" style={{ fontFamily:'"Playfair Display",serif', color: gauge.color }}>
                {result.risk_score >= 70 ? 'Rischio Elevato' : result.risk_score >= 40 ? 'Rischio Moderato' : 'Rischio Contenuto'}
              </p>
              <p className="text-xs mt-1" style={{ color:'#7A8BA8' }}>Settore: <strong style={{ color:'#C9A84C' }}>{settore}</strong></p>
              <p className="text-xs mt-1" style={{ color:'#4A5A72' }}>{result.reati_presupposto.length} reati presupposto identificati</p>
            </div>
          </Card>

          {/* Reati presupposto */}
          <Card className="p-5">
            <p className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color:'#4A5A72' }}>Reati Presupposto</p>
            <div className="space-y-2">
              {result.reati_presupposto.map((r, i) => (
                <div key={i} className="flex items-center gap-3 px-3 py-2.5 rounded-lg"
                     style={{ background:'#0A1628', border:'1px solid #1E2D4A' }}>
                  <span className="font-mono text-xs font-medium flex-shrink-0" style={{ color:'#C9A84C' }}>{r.codice}</span>
                  <span className="text-sm flex-1" style={{ color:'#E8EAF0' }}>{r.descrizione}</span>
                  <span className={`text-xs flex-shrink-0 ${probColor(r.probabilita)}`}>{r.probabilita}</span>
                  <span className="text-xs flex-shrink-0" style={{ color:'#4A5A72' }}>{r.sanzione}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* OdV Raccomandazioni */}
          <Card className="p-5">
            <p className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color:'#4A5A72' }}>Raccomandazioni OdV</p>
            <ul className="space-y-2">
              {result.odv_raccomandazioni.map((r, i) => (
                <li key={i} className="flex items-start gap-2.5 text-sm" style={{ color:'#BCC5D4' }}>
                  <span className="flex-shrink-0 mt-0.5 text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center"
                        style={{ background:'#C9A84C22', color:'#C9A84C', border:'1px solid #C9A84C55' }}>{i+1}</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  );
}

// ── Massimario View ────────────────────────────────────────────────────────
function MassimarioView() {
  return (
    <div className="p-6 h-full flex flex-col items-center justify-center">
      <div className="text-5xl mb-4 opacity-30">📋</div>
      <h2 className="text-xl font-semibold mb-2" style={{ fontFamily:'"Playfair Display",serif', color:'#E8EAF0' }}>Massimario Automatico</h2>
      <p className="text-sm text-center max-w-sm" style={{ color:'#7A8BA8' }}>
        Generazione automatica di massime giurisprudenziali da sentenze della Cassazione.
        <br /><br />
        <span className="italic" style={{ color:'#C9A84C' }}>In sviluppo — disponibile nella prossima release</span>
      </p>
    </div>
  );
}

// ── Settings View ──────────────────────────────────────────────────────────
function SettingsView() {
  return (
    <div className="p-6 space-y-5 overflow-y-auto h-full">
      <h1 className="text-2xl font-semibold" style={{ fontFamily:'"Playfair Display",serif', color:'#E8EAF0' }}>Impostazioni</h1>
      <Card className="p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium" style={{ color:'#E8EAF0' }}>Versione sistema</p>
            <p className="text-xs" style={{ color:'#7A8BA8' }}>RAGForge Italia v2.0.0</p>
          </div>
          <GoldBadge>Stabile</GoldBadge>
        </div>
        <div className="flex items-center justify-between" style={{ borderTop:'1px solid #1E2D4A', paddingTop:16 }}>
          <div>
            <p className="text-sm font-medium" style={{ color:'#E8EAF0' }}>Conformità EU AI Act</p>
            <p className="text-xs" style={{ color:'#7A8BA8' }}>Sistema ad Alto Rischio — Supervisione Umana Attiva</p>
          </div>
          <SuccessPill>✅ Conforme</SuccessPill>
        </div>
        <div className="flex items-center justify-between" style={{ borderTop:'1px solid #1E2D4A', paddingTop:16 }}>
          <div>
            <p className="text-sm font-medium" style={{ color:'#E8EAF0' }}>Audit trail</p>
            <p className="text-xs" style={{ color:'#7A8BA8' }}>Registrazione immutabile con hash SHA-256</p>
          </div>
          <SuccessPill>🔒 Attivo</SuccessPill>
        </div>
        <div className="flex items-center justify-between" style={{ borderTop:'1px solid #1E2D4A', paddingTop:16 }}>
          <div>
            <p className="text-sm font-medium" style={{ color:'#E8EAF0' }}>Data residency</p>
            <p className="text-xs" style={{ color:'#7A8BA8' }}>UE/Italia — GDPR Art. 44 conforme</p>
          </div>
          <SuccessPill>🇮🇹 Italia</SuccessPill>
        </div>
      </Card>
    </div>
  );
}

// ── Root App ───────────────────────────────────────────────────────────────
function App() {
  const [view, setView] = useState('ricerca');
  const [collection, setCollection] = useState('all');
  const [reqId, setReqId] = useState(null);

  const views = {
    ricerca:    <RicercaView collection={collection} onReqId={setReqId} />,
    vigenza:    <VigenzaView />,
    contratti:  <ContrattoView />,
    '231':      <View231 />,
    massimario: <MassimarioView />,
    settings:   <SettingsView />,
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background:'#0A1628', color:'#E8EAF0', fontFamily:'"IBM Plex Sans",system-ui,sans-serif' }}>
      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden">
        <Sidebar view={view} setView={setView} />

        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
          <TopBar collection={collection} setCollection={setCollection} reqId={reqId} />

          <main className="flex-1 overflow-hidden">
            {views[view] || views.ricerca}
          </main>

          {/* Footer disclaimer */}
          <footer className="flex-shrink-0 px-6 py-2.5 flex items-center justify-between"
                  style={{ background:'#0A1628', borderTop:'1px solid #1E2D4A' }}>
            <p className="text-xs italic" style={{ color:'#4A5A72', maxWidth:700 }}>
              ⚖️ Le risposte fornite hanno carattere informativo e non costituiscono parere legale ai sensi dell'art. 2 L. 247/2012
            </p>
            <div className="flex items-center gap-3 flex-shrink-0">
              <span className="text-xs" style={{ color:'#4A5A72' }}>X-Legal-Disclaimer: IT</span>
              <GoldBadge>RAGForge Italia</GoldBadge>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}

// ── Mount ──────────────────────────────────────────────────────────────────
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
