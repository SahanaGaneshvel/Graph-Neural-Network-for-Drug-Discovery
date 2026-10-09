/**
 * AffiniGraph - Drug-Target Affinity Prediction
 * Frontend Application JavaScript
 */

// ============================================
// Application State
// ============================================

// Backend location. Empty = same origin (python scripts/serve_app.py serves both).
// For a separately hosted frontend, frontend/config.js sets window.AFFINIGRAPH_API_BASE.
const API_BASE = (window.AFFINIGRAPH_API_BASE || '').replace(/\/$/, '');

const AppState = {
  data: {
    datasets: {},
    results: [],
    headline: null,
  },
  currentPage: 'dashboard',
  predictionResult: null,
  isLoading: false,
  backend: null,
};

// ============================================
// Sample Protein Sequences
// ============================================

const PROTEIN_SEQUENCES = {
  // Offline fallbacks; replaced by real Davis kinase sequences from /api/examples
  'CDK2': 'MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL',
  'EGFR': 'MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS',
  'ACE2': 'MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYADQSIKVRISLKSALGDKAYEWNDNEMYLFRSSVAYAMRQYFLKVKNQMILFGEEDVRVANLKPRISFNFFVTAPKNVSDIIPRTEVEKAIRMSRSRINDAFRLNDNSLEFLGIQPTLGPPNQPPVSIWLIVFGVVMGVIVVGIVILIFTGIRDRKKKNKARSGENP',
};

// ============================================
// Glossary Data
// ============================================

const GLOSSARY_DATA = [
  {
    term: 'SMILES',
    category: 'chemistry',
    definition: 'Simplified Molecular Input Line Entry System - A notation that represents chemical structures as a line of text. Each atom is represented by its symbol, and bonds are implied or explicitly shown.',
    example: 'CC(=O)Nc1ccc(O)cc1 represents Acetaminophen (Paracetamol)',
  },
  {
    term: 'pKd',
    category: 'chemistry',
    definition: 'The negative logarithm of the dissociation constant (Kd). Higher pKd values indicate stronger binding between a drug and its target. A pKd of 8 means the drug binds tightly (Kd = 10nM).',
    example: 'pKd = 8 → Kd = 10 nM (strong binder); pKd = 5 → Kd = 10 μM (weak binder)',
  },
  {
    term: 'Binding Affinity',
    category: 'chemistry',
    definition: 'The strength of the interaction between a drug molecule and its target protein. Measured by how much drug is needed to occupy 50% of the target binding sites.',
    example: 'High affinity = less drug needed for effect; Low affinity = more drug needed',
  },
  {
    term: 'Molecular Graph',
    category: 'chemistry',
    definition: 'A representation of a molecule where atoms are nodes and chemical bonds are edges. This allows molecules to be processed by Graph Neural Networks.',
    example: 'Benzene: 6 carbon nodes connected in a ring, each with hydrogen edges',
  },
  {
    term: 'Drug-likeness',
    category: 'chemistry',
    definition: 'A qualitative measure of how likely a compound is to be an effective oral drug. Based on molecular properties like size, lipophilicity, and hydrogen bonding.',
    example: "Lipinski's Rule of Five: MW < 500, LogP < 5, HBD < 5, HBA < 10",
  },
  {
    term: 'Amino Acid',
    category: 'biology',
    definition: 'The building blocks of proteins. There are 20 standard amino acids, each with a unique side chain. Represented by single letters (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).',
    example: 'M = Methionine (often starts proteins), G = Glycine (smallest)',
  },
  {
    term: 'Protein Sequence',
    category: 'biology',
    definition: 'The linear chain of amino acids that makes up a protein. The sequence determines how the protein folds and what function it performs.',
    example: 'MVLSPADKTN... (start of hemoglobin sequence)',
  },
  {
    term: 'Binding Site',
    category: 'biology',
    definition: 'The specific region of a protein where a drug molecule attaches. Usually a pocket or groove on the protein surface with complementary shape and chemistry to the drug.',
    example: 'ATP binding site in kinases is where cancer drugs like Imatinib bind',
  },
  {
    term: 'Kinase',
    category: 'biology',
    definition: 'A type of enzyme that transfers phosphate groups to other proteins. Many kinases are drug targets because they control cell growth and are often mutated in cancer.',
    example: 'BCR-ABL kinase (target of Gleevec/Imatinib in leukemia)',
  },
  {
    term: 'Graph Neural Network (GNN)',
    category: 'ml',
    definition: 'A type of neural network that operates on graph-structured data. GNNs learn by aggregating information from neighboring nodes, making them ideal for molecules.',
    example: 'GCN, GAT, and GIN are common GNN architectures for drug discovery',
  },
  {
    term: 'Cross-Attention',
    category: 'ml',
    definition: 'A mechanism that allows the model to learn relationships between two different sequences (like drug atoms and protein residues). Each element attends to relevant elements in the other sequence.',
    example: 'Drug atom → "Which protein residues do I interact with?"',
  },
  {
    term: 'Transformer',
    category: 'ml',
    definition: 'A neural network architecture using self-attention to process sequences. Originally for language, now widely used in biology for proteins and other sequences.',
    example: 'ESM-2 and AlphaFold use Transformer architectures',
  },
  {
    term: 'Embedding',
    category: 'ml',
    definition: 'A learned vector representation of an object (atom, amino acid, molecule). Embeddings capture semantic relationships - similar items have similar vectors.',
    example: 'Each atom in a molecule gets a 128-dimensional embedding vector',
  },
  {
    term: 'MLP (Multi-Layer Perceptron)',
    category: 'ml',
    definition: 'A feedforward neural network with multiple layers. Used as the final prediction head to convert molecular representations to binding affinity scores.',
    example: '256 → 128 → 1 (hidden layers to single output)',
  },
  {
    term: 'Concordance Index (CI)',
    category: 'metrics',
    definition: 'Measures how well the model ranks pairs of drug-target interactions. If drug A actually binds stronger than drug B, CI measures if the model predicts the same ordering.',
    example: 'CI = 0.5 (random), CI = 0.8+ (good), CI = 1.0 (perfect)',
  },
  {
    term: 'RMSE',
    category: 'metrics',
    definition: 'Root Mean Square Error - The square root of the average squared differences between predicted and actual values. Lower is better.',
    example: 'RMSE = 0.5 means predictions are off by 0.5 pKd units on average',
  },
  {
    term: 'Cold Split',
    category: 'metrics',
    definition: "A data splitting strategy where test set contains drugs or proteins not seen during training. Tests the model's ability to generalize to new chemical space.",
    example: 'Cold Drug: test on new molecules; Cold Target: test on new proteins',
  },
  {
    term: 'ADMET',
    category: 'chemistry',
    definition: 'Absorption, Distribution, Metabolism, Excretion, and Toxicity - Properties that determine how a drug behaves in the body. Critical for drug development.',
    example: 'Good ADMET: absorbed orally, distributed well, not toxic',
  },
];

// ============================================
// Dataset Information (fallback when dashboard.json is missing)
// ============================================

const DATASET_INFO = {
  davis: {
    name: 'Davis',
    description: 'Kd of 68 kinase inhibitors measured against 442 kinases (Davis et al., 2011). Affinity used as pKd = 9 - log10(Kd[nM]).',
    pairs: 30056,
    drugs: 68,
    targets: 442,
    status: 'available',
  },
  kiba: {
    name: 'KIBA',
    description: 'KIBA scores that integrate Ki, Kd and IC50 into a single bioactivity value (Tang et al., 2014). Larger and more chemically diverse than Davis.',
    pairs: 118253,
    drugs: 2111,
    targets: 229,
    status: 'available',
  },
};

const MODEL_LABELS = {
  deepdta: 'DeepDTA',
  graphdta_gcn: 'GraphDTA (GCN)',
  graphdta_gat: 'GraphDTA (GAT)',
  graphdta_gin: 'GraphDTA (GIN)',
  proposed: 'Proposed (GIN + Cross-Attn)',
  proposed_concat: 'Ablation (GIN + Concat)',
};

const SPLIT_LABELS = {
  warm: 'Warm',
  cold_drug: 'Cold drug',
  cold_target: 'Cold target',
  cold_both: 'Cold both',
};

// ============================================
// Utility Functions
// ============================================

const escapeHtml = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');

const formatNumber = (value) => {
  if (!Number.isFinite(Number(value))) return '--';
  return Number(value).toLocaleString();
};

const formatMetric = (value, digits = 3) => {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '--';
  return Number(value).toFixed(digits);
};

const formatKd = (nm) => {
  if (!Number.isFinite(nm)) return { value: '--', unit: '' };
  if (nm >= 1e6) return { value: (nm / 1e6).toFixed(2), unit: 'mM' };
  if (nm >= 1e3) return { value: (nm / 1e3).toFixed(2), unit: 'µM' };
  if (nm >= 1) return { value: nm.toFixed(1), unit: 'nM' };
  return { value: (nm * 1e3).toFixed(0), unit: 'pM' };
};

const getMetric = (result, name) => result.metrics?.[name] ?? null;
const getMetricStd = (result, name) => result.metrics_std?.[name] ?? null;

const modelLabel = (key) => MODEL_LABELS[key] || key;
const splitLabel = (key) => SPLIT_LABELS[key] || key;

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function affinityClass(pkd) {
  if (pkd >= 8) return { label: 'Strong Binder', badge: 'strong' };
  if (pkd >= 6) return { label: 'Moderate Binder', badge: 'moderate' };
  if (pkd >= 5.5) return { label: 'Weak Binder', badge: 'weak' };
  return { label: 'Non-Binder', badge: 'weak' };
}

// Interpolate pale -> accent for importance heat colours
function heatColor(score) {
  const s = Math.max(0, Math.min(1, Number(score) || 0));
  const lo = [228, 239, 234];
  const hi = [74, 157, 117];
  const c = lo.map((l, i) => Math.round(l + (hi[i] - l) * s));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// ============================================
// Navigation
// ============================================

function navigateTo(pageId) {
  document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));

  const page = document.getElementById(pageId);
  if (page) {
    page.classList.add('active');
    AppState.currentPage = pageId;
  }

  const navLink = document.querySelector(`.nav-link[data-page="${pageId}"]`);
  if (navLink) navLink.classList.add('active');

  window.scrollTo(0, 0);
}

// ============================================
// Offline fallback (no backend running)
// ============================================

// Very rough property estimates used only when the Python server is not
// running (e.g. the page is opened from a static file server). Clearly
// labelled in the UI as a heuristic, not the trained model.
function heuristicPrediction(smiles, sequence) {
  const heavy = (smiles.match(/[A-Z]/g) || []).length + (smiles.match(/[cnos]/g) || []).length;
  const nitrogen = (smiles.match(/[Nn]/g) || []).length;
  const oxygen = (smiles.match(/[Oo]/g) || []).length;
  const aromatic = (smiles.match(/[cnos]/g) || []).length;

  const mw = heavy * 13.5;
  const logp = aromatic * 0.25 + (smiles.match(/C/g) || []).length * 0.2 - (nitrogen + oxygen) * 0.3;
  const hbd = Math.max(0, Math.round((nitrogen + oxygen) / 3));
  const hba = nitrogen + oxygen;
  const violations = [mw > 500, logp > 5, hbd > 5, hba > 10].filter(Boolean).length;

  let affinity = 5.2 + Math.min(aromatic, 18) * 0.06 + (sequence.length > 250 ? 0.2 : 0) - violations * 0.2;
  affinity = Math.max(4.5, Math.min(8.5, affinity));

  return {
    success: true,
    source: 'offline',
    prediction: {
      binding_affinity: affinity,
      affinity_std: null,
      kd_nm: Math.pow(10, 9 - affinity),
      confidence: 0.2,
      binding_probability: 1 / (1 + Math.exp(-(affinity - 6) * 1.5)),
      interaction_type: affinityClass(affinity).label,
      measured_affinity: null,
    },
    explanation: { atom_symbols: [], atom_importance: [], residue_importance: [], top_regions: [] },
    interactions: [],
    drug_properties: {
      molecular_weight: mw, logp, hbd, hba, tpsa: hbd * 20 + hba * 9,
      rotatable_bonds: null, qed: null, lipinski_violations: violations,
    },
    admet: null,
    notes: [
      'The prediction server is not running, so this is a rough property-based estimate, not the trained GNN.',
      'Start it with: python scripts/serve_app.py, then reload this page.',
    ],
  };
}

// ============================================
// Render: dashboard & results
// ============================================

function resultRow(result, withSeeds = false) {
  const ci = getMetric(result, 'ci');
  const ciStd = getMetricStd(result, 'ci');
  const rmse = getMetric(result, 'rmse');
  const rmseStd = getMetricStd(result, 'rmse');
  const pearson = getMetric(result, 'pearson');
  const pm = (mean, std) => `${formatMetric(mean)}${std !== null && result.n_seeds > 1 ? ` <span class="std">± ${formatMetric(std)}</span>` : ''}`;
  return `
    <tr class="${result.model === 'proposed' ? 'highlight-row' : ''}">
      <td>${escapeHtml(modelLabel(result.model))}</td>
      <td>${escapeHtml(result.dataset)}</td>
      <td>${escapeHtml(splitLabel(result.split))}</td>
      ${withSeeds ? `<td>${escapeHtml(result.n_seeds ?? '--')}</td>` : ''}
      <td>${pm(ci, ciStd)}</td>
      <td>${pm(rmse, rmseStd)}</td>
      ${withSeeds ? `<td>${formatMetric(pearson)}</td>` : ''}
      <td><span class="status">${result.n_seeds > 1 ? `${result.n_seeds} seeds` : '1 run'}</span></td>
    </tr>`;
}

function renderDashboardResults() {
  const tbody = document.getElementById('recent-results');
  const emptyState = document.getElementById('dashboard-empty');
  if (!tbody) return;

  // Show the warm-split comparison on Davis first: it is the headline table
  const results = [...AppState.data.results]
    .sort((a, b) => (a.split === 'warm' ? -1 : 1) - (b.split === 'warm' ? -1 : 1)
      || (getMetric(b, 'ci') || 0) - (getMetric(a, 'ci') || 0))
    .slice(0, 6);

  if (results.length === 0) {
    tbody.innerHTML = '';
    if (emptyState) emptyState.style.display = 'block';
    return;
  }
  if (emptyState) emptyState.style.display = 'none';
  tbody.innerHTML = results.map(r => resultRow(r)).join('');
}

function renderFullResults() {
  const tbody = document.getElementById('full-results-body');
  const emptyState = document.getElementById('results-empty');
  const datasetFilter = document.getElementById('results-dataset-filter')?.value || 'all';
  const metric = document.getElementById('results-metric-filter')?.value || 'ci';
  if (!tbody) return;

  let results = AppState.data.results;
  if (datasetFilter !== 'all') results = results.filter(r => r.dataset === datasetFilter);

  const lowerIsBetter = metric === 'rmse';
  const splitOrder = Object.keys(SPLIT_LABELS);
  results = [...results].sort((a, b) =>
    splitOrder.indexOf(a.split) - splitOrder.indexOf(b.split)
    || (lowerIsBetter
      ? (getMetric(a, metric) ?? Infinity) - (getMetric(b, metric) ?? Infinity)
      : (getMetric(b, metric) ?? -Infinity) - (getMetric(a, metric) ?? -Infinity)));

  if (results.length === 0) {
    tbody.innerHTML = '';
    if (emptyState) emptyState.style.display = 'block';
  } else {
    if (emptyState) emptyState.style.display = 'none';
    tbody.innerHTML = results.map(r => resultRow(r, true)).join('');
  }
  updateResultsOverview(results);
}

function updateResultsOverview(results) {
  const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };

  if (results.length === 0) {
    set('best-ci', '--');
    set('best-rmse', '--');
    set('total-experiments', '0');
    return;
  }

  const best = (metric, lower) => [...results]
    .filter(r => getMetric(r, metric) !== null)
    .sort((a, b) => lower ? getMetric(a, metric) - getMetric(b, metric) : getMetric(b, metric) - getMetric(a, metric))[0];

  const bestCi = best('ci', false);
  if (bestCi) {
    set('best-ci', formatMetric(getMetric(bestCi, 'ci')));
    set('best-ci-detail', `${modelLabel(bestCi.model)} · ${bestCi.dataset} · ${splitLabel(bestCi.split)}`);
  }
  const bestRmse = best('rmse', true);
  if (bestRmse) {
    set('best-rmse', formatMetric(getMetric(bestRmse, 'rmse')));
    set('best-rmse-detail', `${modelLabel(bestRmse.model)} · ${bestRmse.dataset} · ${splitLabel(bestRmse.split)}`);
  }
  const runs = results.reduce((sum, r) => sum + (r.n_seeds || 1), 0);
  set('total-experiments', runs.toString());
}

function renderHeadlineStats() {
  const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
  const datasets = AppState.data.datasets || DATASET_INFO;
  const pairs = Object.values(datasets).reduce((sum, d) => sum + (d.pairs || 0), 0);
  set('stat-datasets', Object.keys(datasets).length.toString());
  set('stat-interactions', pairs ? `${Math.round(pairs / 1000)}K` : '--');

  const models = new Set(AppState.data.results.map(r => r.model));
  if (models.size) set('stat-models', models.size.toString());

  const h = AppState.data.headline;
  if (!h) return;
  set('stat-accuracy', formatMetric(getMetric(h, 'ci')));
  set('stat-accuracy-detail', `Proposed model, Davis warm test set (${h.n_seeds} seed${h.n_seeds === 1 ? '' : 's'})`);

  const scoreCard = (id, metric, toWidth) => {
    const value = getMetric(h, metric);
    if (value === null) return;
    const std = getMetricStd(h, metric);
    set(id, `Our Score: ${formatMetric(value)}${std !== null && h.n_seeds > 1 ? ` ± ${formatMetric(std)}` : ''}`);
    const bar = document.getElementById(`${id}-bar`);
    if (bar) bar.style.width = `${Math.max(0, Math.min(100, toWidth(value)))}%`;
  };
  scoreCard('hiw-ci', 'ci', v => (v - 0.5) / 0.5 * 100);
  scoreCard('hiw-rmse', 'rmse', v => (1 - v / 2) * 100);
  scoreCard('hiw-pearson', 'pearson', v => v * 100);
}

function renderDatasets() {
  const grid = document.getElementById('datasets-grid');
  if (!grid) return;
  const datasets = AppState.data.datasets && Object.keys(AppState.data.datasets).length
    ? AppState.data.datasets : DATASET_INFO;

  grid.innerHTML = Object.values(datasets).map(dataset => `
    <div class="dataset-card">
      <div class="dataset-header">
        <span class="dataset-name">${escapeHtml(dataset.name)}</span>
        <span class="dataset-status available">On disk</span>
      </div>
      <p class="dataset-description">${escapeHtml(dataset.description)}</p>
      <div class="dataset-stats">
        <div class="dataset-stat">
          <span class="value">${formatNumber(dataset.pairs)}</span>
          <span class="label">Pairs</span>
        </div>
        <div class="dataset-stat">
          <span class="value">${formatNumber(dataset.drugs)}</span>
          <span class="label">Drugs</span>
        </div>
        <div class="dataset-stat">
          <span class="value">${formatNumber(dataset.targets ?? dataset.proteins)}</span>
          <span class="label">Proteins</span>
        </div>
      </div>
    </div>
  `).join('');
}

function renderGlossary(filter = 'all', search = '') {
  const list = document.getElementById('glossary-list');
  if (!list) return;

  let items = GLOSSARY_DATA;
  if (filter !== 'all') items = items.filter(item => item.category === filter);
  if (search) {
    const q = search.toLowerCase();
    items = items.filter(item => item.term.toLowerCase().includes(q) || item.definition.toLowerCase().includes(q));
  }

  list.innerHTML = items.map(item => `
    <div class="glossary-item">
      <div class="glossary-item-header">
        <span class="glossary-term">${escapeHtml(item.term)}</span>
        <span class="glossary-category">${escapeHtml(item.category)}</span>
      </div>
      <p class="glossary-definition">${escapeHtml(item.definition)}</p>
      ${item.example ? `
        <div class="glossary-example">
          <strong>Example:</strong>
          <code>${escapeHtml(item.example)}</code>
        </div>` : ''}
    </div>
  `).join('');
}

// ============================================
// Render: prediction
// ============================================

function sourceChip(source) {
  if (source === 'model') return '<span class="source-chip model">Trained GNN ensemble</span>';
  if (source === 'heuristic') return '<span class="source-chip heuristic">Heuristic (no trained model loaded)</span>';
  return '<span class="source-chip heuristic">Offline estimate (server not running)</span>';
}

function renderAtomImportance(expl) {
  const atoms = expl.atom_symbols || [];
  const scores = expl.atom_importance || [];
  if (!atoms.length || atoms.length !== scores.length) return '';
  return `
    <div class="result-block">
      <h4>Atom importance <span class="hint">gradient × input saliency, averaged over the ensemble</span></h4>
      <div class="atom-chips">
        ${atoms.map((sym, i) => `
          <span class="atom-chip" style="background:${heatColor(scores[i])}; color:${scores[i] > 0.55 ? '#fff' : 'var(--text-primary)'}"
                title="Atom ${i + 1} (${escapeHtml(sym)}): ${formatMetric(scores[i], 2)}">${escapeHtml(sym)}<sub>${i + 1}</sub></span>
        `).join('')}
      </div>
    </div>`;
}

function renderResidueImportance(expl, sequenceLength) {
  const scores = expl.residue_importance || [];
  if (!scores.length) return '';
  // Down-sample to at most 160 bins for display
  const bins = Math.min(160, scores.length);
  const size = scores.length / bins;
  const binned = Array.from({ length: bins }, (_, b) => {
    const chunk = scores.slice(Math.floor(b * size), Math.max(Math.floor((b + 1) * size), Math.floor(b * size) + 1));
    return Math.max(...chunk);
  });
  const regions = expl.top_regions || [];
  return `
    <div class="result-block">
      <h4>Protein attention <span class="hint">cross-attention over residues 1–${scores.length}${sequenceLength > scores.length ? ` (sequence truncated from ${sequenceLength})` : ''}</span></h4>
      <div class="residue-strip">
        ${binned.map((s, b) => `<span style="background:${heatColor(s)}" title="~residue ${Math.floor(b * size) + 1}"></span>`).join('')}
      </div>
      ${regions.length ? `
        <div class="region-list">
          ${regions.map(r => `
            <span class="region-chip" title="relative attention ${formatMetric(r.score, 2)}">
              ${r.start}–${r.end} <code>${escapeHtml(r.segment)}</code>
            </span>`).join('')}
        </div>
        <p class="hint">Highest-attention segments. Attention is a learned weighting, not a docked binding pose.</p>` : ''}
    </div>`;
}

function renderPropertyGrid(props) {
  if (!props) return '';
  const cells = [
    ['MW (Da)', formatMetric(props.molecular_weight, 0)],
    ['LogP', formatMetric(props.logp, 2)],
    ['H-bond donors', props.hbd ?? '--'],
    ['H-bond acceptors', props.hba ?? '--'],
    ['TPSA (Å²)', formatMetric(props.tpsa, 0)],
    ['QED', formatMetric(props.qed, 2)],
    ['Rotatable bonds', props.rotatable_bonds ?? '--'],
    ['Lipinski violations', props.lipinski_violations ?? '--'],
  ];
  return `
    <div class="result-block">
      <h4>Molecular properties <span class="hint">RDKit descriptors</span></h4>
      <div class="property-grid">
        ${cells.map(([label, value]) => `
          <div class="property-cell"><span class="value">${escapeHtml(value)}</span><span class="label">${escapeHtml(label)}</span></div>
        `).join('')}
      </div>
    </div>`;
}

function renderAdmet(admet) {
  if (!admet) return '';
  const groups = ['absorption', 'distribution', 'metabolism', 'excretion', 'toxicity'];
  const pretty = (k) => k.replace(/_/g, ' ');
  return `
    <div class="result-block">
      <h4>ADMET flags <span class="hint">rule-based (Lipinski / Veber-style thresholds), not a trained model</span></h4>
      <div class="admet-grid">
        ${groups.map(g => `
          <div class="admet-card">
            <span class="admet-title">${escapeHtml(g)}</span>
            ${Object.entries(admet[g] || {}).map(([k, v]) => `
              <div class="admet-row"><span>${escapeHtml(pretty(k))}</span><strong>${escapeHtml(v)}</strong></div>`).join('')}
          </div>`).join('')}
      </div>
      ${(admet.drug_warnings || []).length ? `
        <ul class="warning-list">${admet.drug_warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('')}</ul>` : ''}
    </div>`;
}

function renderPredictionResult(result) {
  const container = document.getElementById('prediction-results');
  if (!container) return;

  const p = result.prediction;
  const cls = affinityClass(p.binding_affinity);
  const kd = formatKd(p.kd_nm);
  const std = p.affinity_std;
  const measured = p.measured_affinity;

  const domainNotes = [];
  if (p.drug_similarity !== null && p.drug_similarity !== undefined) {
    domainNotes.push(`Nearest training drug: Tanimoto ${formatMetric(p.drug_similarity, 2)}${p.drug_similarity < 0.4 ? ' — outside the training chemistry, treat with caution' : ''}`);
  }
  if (p.protein_in_training_set === false) {
    domainNotes.push('Target sequence is not one of the 442 Davis kinases; predictions for non-kinases are extrapolation.');
  }

  container.innerHTML = `
    <div class="prediction-result">
      <div class="result-header">
        <div>
          <h3>Prediction</h3>
          <div style="margin-top: 6px;">${sourceChip(result.source)}</div>
        </div>
        <span class="result-badge ${cls.badge}">${escapeHtml(p.interaction_type || cls.label)}</span>
      </div>

      <div class="result-metrics">
        <div class="metric-item">
          <span class="label">Predicted affinity</span>
          <span class="value">${formatMetric(p.binding_affinity, 2)}${std !== null && std !== undefined ? `<span class="unit">± ${formatMetric(std, 2)}</span>` : ''}<span class="unit">pKd</span></span>
        </div>
        <div class="metric-item">
          <span class="label">Predicted Kd</span>
          <span class="value">${kd.value}<span class="unit">${kd.unit}</span></span>
        </div>
        <div class="metric-item">
          <span class="label">Measured (Davis${measured === null && (p.measured_values || []).length > 1 ? `: ${p.measured_values.length} entries share this sequence` : ''})</span>
          <span class="value">${measured !== null && measured !== undefined ? formatMetric(measured, 2)
            : (p.measured_values || []).length > 1 ? `${formatMetric(Math.min(...p.measured_values), 1)}–${formatMetric(Math.max(...p.measured_values), 1)}` : '--'}<span class="unit">${measured !== null && measured !== undefined || (p.measured_values || []).length > 1 ? 'pKd' : 'not in dataset'}</span></span>
        </div>
        <div class="metric-item">
          <span class="label">Confidence</span>
          <span class="value">${formatMetric((p.confidence || 0) * 100, 0)}<span class="unit">%</span></span>
        </div>
      </div>

      ${p.member_predictions && p.member_predictions.length > 1 ? `
        <p class="hint" style="margin-bottom: 16px;">Ensemble members: ${p.member_predictions.map(v => formatMetric(v, 2)).join(' · ')} pKd</p>` : ''}
      ${domainNotes.length ? `<ul class="info-list">${domainNotes.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul>` : ''}

      ${renderAtomImportance(result.explanation || {})}
      ${renderResidueImportance(result.explanation || {}, result.sequenceLength || 0)}
      ${renderPropertyGrid(result.drug_properties)}
      ${renderAdmet(result.admet)}

      ${(result.notes || []).length ? `
        <div class="result-block">
          <h4>Notes</h4>
          <div class="note-list">
            ${result.notes.map(note => `<div class="note-item">${escapeHtml(note)}</div>`).join('')}
          </div>
        </div>` : ''}
    </div>
  `;

  updateSimulationPanel(result);
}

// ============================================
// Simulation view
// ============================================

function updateSimulationPanel(result) {
  const panel = document.getElementById('sim-results-panel');
  const metrics = document.getElementById('sim-metrics');
  const canvas = document.getElementById('simulation-canvas');
  const p = result.prediction;

  if (panel && metrics) {
    panel.style.display = 'block';
    metrics.innerHTML = `
      <div class="metric-item">
        <span class="label">Affinity</span>
        <span class="value">${formatMetric(p.binding_affinity, 2)}</span>
      </div>
      <div class="metric-item">
        <span class="label">Type</span>
        <span class="value" style="font-size: 14px;">${escapeHtml(p.interaction_type)}</span>
      </div>`;
  }
  if (canvas) renderSimulationVisualization(canvas, result);
}

function renderSimulationVisualization(canvas, result) {
  const p = result.prediction;
  const expl = result.explanation || {};
  const cls = affinityClass(p.binding_affinity);
  const regions = (expl.top_regions || []).slice(0, 6);
  const atoms = (expl.atom_symbols || []).map((sym, i) => ({ sym, score: (expl.atom_importance || [])[i] || 0 }));

  const colors = {
    drug: '#7eb8a8',
    protein: '#a8d4e6',
    bond: cls.badge === 'strong' ? '#4a9d75' : cls.badge === 'moderate' ? '#d4a64a' : '#d77a7a',
  };

  // Drug atoms on an inner ring, coloured by saliency; attention regions on the outer ring
  const atomDots = atoms.slice(0, 40).map((a, i, arr) => {
    const angle = (i / arr.length) * Math.PI * 2;
    return { x: 50 + Math.cos(angle) * 13, y: 50 + Math.sin(angle) * 13, ...a };
  });
  const regionDots = regions.map((r, i) => {
    const angle = (i / Math.max(regions.length, 1)) * Math.PI * 2 - Math.PI / 2;
    return { x: 50 + Math.cos(angle) * 38, y: 50 + Math.sin(angle) * 38, ...r };
  });
  const topAtoms = [...atomDots].sort((a, b) => b.score - a.score).slice(0, Math.max(1, regionDots.length));

  canvas.innerHTML = `
    <div class="simulation-result" style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px;">
      <div style="position: relative; width: 320px; max-width: 100%; aspect-ratio: 1; margin-bottom: 20px;">
        <svg viewBox="0 0 100 100" style="width: 100%; height: 100%;" role="img" aria-label="Drug-protein attention map">
          <circle cx="50" cy="50" r="45" fill="none" stroke="${colors.protein}" stroke-width="0.5" opacity="0.4"/>
          <circle cx="50" cy="50" r="38" fill="none" stroke="${colors.protein}" stroke-width="0.3" opacity="0.4" stroke-dasharray="1,1"/>
          ${regionDots.map((r, i) => {
            const a = topAtoms[i % topAtoms.length];
            return a ? `<line x1="${a.x}" y1="${a.y}" x2="${r.x}" y2="${r.y}" stroke="${colors.bond}" stroke-width="${0.4 + r.score * 1.2}" opacity="${0.35 + r.score * 0.5}" stroke-dasharray="2,1">
                <animate attributeName="opacity" values="${0.3 + r.score * 0.4};${0.6 + r.score * 0.4};${0.3 + r.score * 0.4}" dur="2.4s" begin="${i * 0.3}s" repeatCount="indefinite"/>
              </line>` : '';
          }).join('')}
          <circle cx="50" cy="50" r="17" fill="${colors.drug}" opacity="0.12"/>
          ${atomDots.map(a => `
            <circle cx="${a.x}" cy="${a.y}" r="${1.4 + a.score * 1.6}" fill="${heatColor(a.score)}" stroke="${colors.drug}" stroke-width="0.2">
              <title>${escapeHtml(a.sym)} · saliency ${formatMetric(a.score, 2)}</title>
            </circle>`).join('')}
          <text x="50" y="51.5" text-anchor="middle" fill="${colors.drug}" font-size="3.6" font-weight="700">Drug</text>
          ${regionDots.map(r => `
            <g>
              <circle cx="${r.x}" cy="${r.y}" r="${3.5 + r.score * 2.5}" fill="${colors.protein}" opacity="0.9"><title>Residues ${r.start}-${r.end}: ${escapeHtml(r.segment)}</title></circle>
              <text x="${r.x}" y="${r.y + 1}" text-anchor="middle" fill="#2d4a3e" font-size="2.6" font-weight="600">${r.start}</text>
            </g>`).join('')}
          <text x="50" y="97" text-anchor="middle" fill="${colors.bond}" font-size="3.6" font-weight="600">${escapeHtml(p.interaction_type)}</text>
        </svg>
      </div>
      <div class="simulation-summary" style="text-align: center; max-width: 440px;">
        <h4 style="color: var(--text-primary); margin-bottom: 12px; font-size: 16px;">Interaction map</h4>
        <p style="color: var(--text-secondary); font-size: 13px; line-height: 1.6;">
          ${regionDots.length
            ? `Inner ring: drug atoms coloured by saliency. Outer ring: the ${regionDots.length} protein segments that receive the most cross-attention (labelled by first residue). Line width shows relative attention.`
            : 'Load a trained model (python scripts/serve_app.py) to see atom saliency and protein attention regions here.'}
        </p>
      </div>
    </div>`;
}

// ============================================
// Data Loading
// ============================================

async function loadDashboardData() {
  try {
    const response = await fetch(`data/dashboard.json?ts=${Date.now()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    AppState.data = {
      datasets: data.datasets || DATASET_INFO,
      results: Array.isArray(data.results) ? data.results : [],
      headline: data.headline || null,
    };
  } catch (error) {
    console.warn('dashboard.json not available:', error);
    AppState.data = { datasets: DATASET_INFO, results: [], headline: null };
  }
  renderDashboardResults();
  renderFullResults();
  renderHeadlineStats();
  renderDatasets();
}

async function checkBackend() {
  const dot = document.getElementById('model-status-dot');
  const text = document.getElementById('model-status-text');
  try {
    const response = await fetch(`${API_BASE}/api/model`);
    if (!response.ok) throw new Error('no api');
    const info = await response.json();
    AppState.backend = info;
    if (info.loaded) {
      dot?.classList.add('online');
      if (text) text.textContent = `GNN ×${info.ensemble_size} ready`;
    } else {
      if (text) text.textContent = 'No trained model';
    }
  } catch (e) {
    AppState.backend = null;
    if (text) text.textContent = 'Server offline';
  }
}

async function loadExamples() {
  try {
    const response = await fetch(`${API_BASE}/api/examples`);
    if (!response.ok) return;
    const { drugs, targets } = await response.json();
    const drugBox = document.getElementById('drug-examples');
    const targetBox = document.getElementById('target-examples');
    if (drugBox && drugs.length) {
      drugBox.innerHTML = '<span class="example-label">Davis drugs:</span>' + drugs.map(d =>
        `<button class="example-btn" data-smiles="${escapeHtml(d.smiles)}" title="PubChem CID ${escapeHtml(d.id)}">${escapeHtml(d.name)}</button>`).join('');
    }
    if (targetBox && targets.length) {
      targets.forEach(t => { PROTEIN_SEQUENCES[t.id] = t.sequence; });
      targetBox.innerHTML = '<span class="example-label">Davis kinases:</span>' + targets.map(t =>
        `<button class="example-btn" data-sequence="${escapeHtml(t.id)}">${escapeHtml(t.name)}</button>`).join('');
    }
    bindExampleButtons();
  } catch (e) {
    // Static hosting: keep the built-in examples
  }
}

// ============================================
// Event Handlers
// ============================================

function bindExampleButtons() {
  document.querySelectorAll('.example-btn[data-smiles]').forEach(btn => {
    btn.onclick = () => {
      const input = document.getElementById('drug-input');
      if (input) input.value = btn.dataset.smiles;
    };
  });
  document.querySelectorAll('.example-btn[data-sequence]').forEach(btn => {
    btn.onclick = () => {
      const input = document.getElementById('protein-input');
      const seq = PROTEIN_SEQUENCES[btn.dataset.sequence];
      if (input && seq) {
        input.value = seq;
        AppState.selectedTarget = { id: btn.dataset.sequence, sequence: seq };
      }
    };
  });
}

function setupEventListeners() {
  document.querySelectorAll('[data-navigate]').forEach(el => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTo(el.dataset.navigate);
    });
  });
  document.querySelectorAll('.nav-link[data-page]').forEach(el => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTo(el.dataset.page);
      history.replaceState(null, '', `#${el.dataset.page}`);
    });
  });

  document.getElementById('refresh-btn')?.addEventListener('click', loadDashboardData);
  document.getElementById('predict-btn')?.addEventListener('click', handlePrediction);
  document.getElementById('results-dataset-filter')?.addEventListener('change', renderFullResults);
  document.getElementById('results-metric-filter')?.addEventListener('change', renderFullResults);

  bindExampleButtons();

  const glossarySearch = document.getElementById('glossary-search');
  glossarySearch?.addEventListener('input', (e) => {
    const activeCategory = document.querySelector('.glossary-nav-btn.active')?.dataset.category || 'all';
    renderGlossary(activeCategory, e.target.value);
  });
  document.querySelectorAll('.glossary-nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.glossary-nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderGlossary(btn.dataset.category, glossarySearch?.value || '');
    });
  });

  document.querySelectorAll('[data-tooltip]').forEach(el => {
    el.addEventListener('mouseenter', showTooltip);
    el.addEventListener('mouseleave', hideTooltip);
  });
  document.querySelectorAll('.term').forEach(el => {
    el.addEventListener('mouseenter', (e) => {
      const termId = el.dataset.term;
      // Match "gnn" to "Graph Neural Network (GNN)", "pkd" to "pKd", etc.
      const termData = GLOSSARY_DATA.find(g => {
        const slug = g.term.toLowerCase().replace(/[()]/g, '').trim().replace(/\s+/g, '-');
        return slug === termId || slug.startsWith(`${termId}-`) || slug.split('-').includes(termId);
      });
      if (termData) showTermTooltip(e, termData);
    });
    el.addEventListener('mouseleave', hideTooltip);
  });

  document.querySelectorAll('.toolbar-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toolbar-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  window.addEventListener('hashchange', () => {
    const hash = window.location.hash.slice(1);
    if (hash) navigateTo(hash);
  });
}

function showPredictionError(message) {
  const container = document.getElementById('prediction-results');
  if (!container) return;
  container.innerHTML = `
    <div class="prediction-result">
      <div class="result-header"><h3>Could not run prediction</h3></div>
      <div class="note-item error">${escapeHtml(message)}</div>
    </div>`;
}

async function handlePrediction() {
  const smiles = document.getElementById('drug-input')?.value.trim();
  const sequence = (document.getElementById('protein-input')?.value || '').replace(/\s+/g, '').toUpperCase();
  const predictBtn = document.getElementById('predict-btn');
  const loadingOverlay = document.getElementById('loading-overlay');

  if (!smiles) return showPredictionError('Please enter a drug SMILES string.');
  if (!sequence) return showPredictionError('Please enter a protein sequence.');
  if (!/^[A-Z]+$/.test(sequence)) return showPredictionError('The protein sequence should only contain one-letter amino-acid codes.');

  predictBtn?.classList.add('loading');
  loadingOverlay?.classList.add('visible');

  let result;
  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        smiles,
        protein_sequence: sequence,
        // Davis id of a clicked example (only if the sequence was not edited since)
        protein_id: AppState.selectedTarget?.sequence === sequence ? AppState.selectedTarget.id : undefined,
      }),
    });
    const body = await response.json().catch(() => ({}));
    if (response.ok && body.success) {
      result = body;
    } else if (response.status === 422 || response.status === 400) {
      predictBtn?.classList.remove('loading');
      loadingOverlay?.classList.remove('visible');
      return showPredictionError(body.error || 'Invalid input.');
    } else {
      throw new Error(body.error || `HTTP ${response.status}`);
    }
  } catch (e) {
    console.warn('Prediction API unavailable, using offline estimate:', e);
    await sleep(400);
    result = heuristicPrediction(smiles, sequence);
  }

  result.sequenceLength = sequence.length;
  AppState.predictionResult = result;

  predictBtn?.classList.remove('loading');
  loadingOverlay?.classList.remove('visible');
  renderPredictionResult(result);
}

function positionTooltip(tooltip, target) {
  const rect = target.getBoundingClientRect();
  const left = Math.min(rect.left, window.innerWidth - 320);
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${rect.bottom + 10}px`;
}

function showTooltip(e) {
  const tooltip = document.getElementById('tooltip');
  if (!tooltip) return;
  tooltip.textContent = e.currentTarget.dataset.tooltip;
  tooltip.classList.add('visible');
  positionTooltip(tooltip, e.currentTarget);
}

function showTermTooltip(e, termData) {
  const tooltip = document.getElementById('tooltip');
  if (!tooltip) return;
  tooltip.innerHTML = `<strong>${escapeHtml(termData.term)}</strong><br>${escapeHtml(termData.definition)}`;
  tooltip.classList.add('visible');
  positionTooltip(tooltip, e.currentTarget);
}

function hideTooltip() {
  document.getElementById('tooltip')?.classList.remove('visible');
}

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  renderGlossary();
  renderDatasets();
  loadDashboardData();
  checkBackend();
  loadExamples();

  const hash = window.location.hash.slice(1);
  if (hash) navigateTo(hash);
});
