# app_32.py — Tic Tac Toe com SVM | T1 IA 2026
# Humano (X) vs Máquina aleatória (O) · SVM classifica o estado a cada jogada
#
# Como rodar:
#   pip install flask joblib scikit-learn numpy
#   python app_32.py
#   Acesse: http://localhost:5032

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import random

app_32 = Flask(__name__)

# ── Carrega modelo ─────────────────────────────────────────────────────────────
try:
    modelo_svm_32 = joblib.load('svm_model_32.pkl')
    print("✓ Modelo SVM carregado com sucesso!")
except FileNotFoundError:
    print("✗ ERRO: svm_model_32.pkl não encontrado.")
    print("  → Execute a célula de exportação no Colab e coloque o arquivo aqui.")
    modelo_svm_32 = None

# ── Estado real do tabuleiro ───────────────────────────────────────────────────
def estado_real_32(tab):
    combos_32 = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for c in combos_32:
        if tab[c[0]] == tab[c[1]] == tab[c[2]] == 1:  return 'X vence'
        if tab[c[0]] == tab[c[1]] == tab[c[2]] == -1: return 'O vence'
    if 0 not in tab:
        return 'Empate'
    for jogador_32 in [1, -1]:
        for c in combos_32:
            vals_32 = [tab[i] for i in c]
            if vals_32.count(jogador_32) == 2 and vals_32.count(0) == 1:
                return 'Possibilidade de Fim de Jogo'
    return 'Tem jogo'

# ── HTML do jogo ───────────────────────────────────────────────────────────────
HTML_32 = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tic Tac Toe — IA com SVM</title>
<style>
:root {
  --bg:      #0f1117;
  --card:    #1a1d2e;
  --border:  #2d3147;
  --accent:  #6c63ff;
  --x-col:   #ff6b6b;
  --o-col:   #4ecdc4;
  --text:    #e0e0e0;
  --muted:   #7a7f9a;
  --ok:      #4ade80;
  --warn:    #fbbf24;
  --err:     #f87171;
  --info:    #60a5fa;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 1rem;
  gap: 0.5rem;
}
h1 { font-size: 1.5rem; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }
.sub { font-size: 0.82rem; color: var(--muted); margin-bottom: 1.8rem; }

/* Layout */
.layout {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
  flex-wrap: wrap;
  justify-content: center;
  width: 100%;
  max-width: 860px;
}

/* Tabuleiro */
.board-wrap { display: flex; flex-direction: column; align-items: center; gap: 1rem; }
.board {
  display: grid;
  grid-template-columns: repeat(3, 108px);
  grid-template-rows: repeat(3, 108px);
  gap: 6px;
  background: var(--border);
  padding: 6px;
  border-radius: 14px;
}
.cell {
  background: var(--card);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.8rem;
  font-weight: 900;
  cursor: pointer;
  transition: background 0.15s, transform 0.1s;
  user-select: none;
}
.cell:hover:not(.filled):not(.disabled) { background: #22263a; transform: scale(1.03); }
.cell.x     { color: var(--x-col); cursor: default; }
.cell.o     { color: var(--o-col); cursor: default; }
.cell.disabled { cursor: not-allowed; opacity: 0.6; }
.cell.filled   { cursor: default; }

/* Turn indicator */
.turn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--text);
}
.dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  animation: pulse 1.2s infinite;
}
.dot.x { background: var(--x-col); }
.dot.o { background: var(--o-col); animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

/* Painel lateral */
.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.4rem;
  min-width: 290px;
  max-width: 330px;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.panel-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--muted);
  font-weight: 700;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.875rem;
}
.row-label { color: var(--muted); }
.row-val   { font-weight: 600; }

/* Badges */
.badge {
  padding: 0.18rem 0.65rem;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
}
.b-neutral { background: #22263a; color: var(--text); }
.b-ok      { background: #0f2e1a; color: var(--ok);   }
.b-warn    { background: #2e1f0a; color: var(--warn);  }
.b-err     { background: #2e0f0f; color: var(--err);   }
.b-info    { background: #111e36; color: var(--info);  }

/* Barra de acurácia */
.bar-bg {
  height: 7px;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 4px;
  transition: width 0.4s ease;
}

/* Caixa de mensagem */
.msg {
  padding: 0.75rem 0.9rem;
  border-radius: 8px;
  font-size: 0.855rem;
  line-height: 1.55;
  border-left: 3px solid;
}
.msg.info    { background: #101825; border-color: var(--info);  color: #93c5fd; }
.msg.ok      { background: #0a1e12; border-color: var(--ok);    color: #86efac; }
.msg.warn    { background: #1e1408; border-color: var(--warn);  color: #fde68a; }
.msg.err     { background: #1e0a0a; border-color: var(--err);   color: #fca5a5; }

/* Histórico */
.hist {
  max-height: 190px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0;
}
.hist-item {
  font-size: 0.77rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.32rem 0;
  border-bottom: 1px solid #1a1d2e;
  color: var(--muted);
  gap: 0.5rem;
}
.hist-item .ok-tag  { color: var(--ok);  font-weight: 600; }
.hist-item .err-tag { color: var(--err); font-weight: 600; }
.hist::-webkit-scrollbar { width: 3px; }
.hist::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Botão */
.btn {
  background: var(--accent);
  color: #fff;
  border: none;
  padding: 0.65rem 1.2rem;
  border-radius: 8px;
  font-size: 0.88rem;
  font-weight: 600;
  cursor: pointer;
  width: 100%;
  transition: opacity 0.2s, transform 0.1s;
}
.btn:hover  { opacity: 0.82; transform: translateY(-1px); }
.btn:active { transform: translateY(0); }
</style>
</head>
<body>

<h1>Tic Tac Toe — IA com SVM</h1>
<p class="sub">Humano (X) vs Máquina aleatória (O) &nbsp;·&nbsp; SVM classifica o estado a cada jogada</p>

<div class="layout">

  <!-- Tabuleiro -->
  <div class="board-wrap">
    <div class="board" id="board32">
      <div class="cell" id="c0" onclick="jogada_32(0)"></div>
      <div class="cell" id="c1" onclick="jogada_32(1)"></div>
      <div class="cell" id="c2" onclick="jogada_32(2)"></div>
      <div class="cell" id="c3" onclick="jogada_32(3)"></div>
      <div class="cell" id="c4" onclick="jogada_32(4)"></div>
      <div class="cell" id="c5" onclick="jogada_32(5)"></div>
      <div class="cell" id="c6" onclick="jogada_32(6)"></div>
      <div class="cell" id="c7" onclick="jogada_32(7)"></div>
      <div class="cell" id="c8" onclick="jogada_32(8)"></div>
    </div>
    <div class="turn" id="turn32">
      <div class="dot x"></div>
      <span>Sua vez &nbsp;(X)</span>
    </div>
  </div>

  <!-- Painel -->
  <div class="panel">
    <div class="panel-label">Análise SVM em Tempo Real</div>

    <div class="row">
      <span class="row-label">Predição SVM</span>
      <span class="badge b-neutral" id="pred32">—</span>
    </div>
    <div class="row">
      <span class="row-label">Estado Real</span>
      <span class="badge b-neutral" id="real32">—</span>
    </div>
    <div class="row">
      <span class="row-label">Última classificação</span>
      <span class="row-val" id="last32" style="color:var(--muted)">—</span>
    </div>

    <!-- Acurácia -->
    <div>
      <div class="row" style="margin-bottom:.4rem">
        <span class="row-label">Acurácia no jogo</span>
        <span class="row-val" id="acc32">—</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" id="bar32" style="width:0%"></div>
      </div>
      <div class="row" style="margin-top:.35rem;font-size:.78rem">
        <span class="row-label">Acertos / Total</span>
        <span id="counts32" style="color:var(--muted)">0 / 0</span>
      </div>
    </div>

    <div class="msg info" id="msg32">Faça sua jogada clicando em uma célula.</div>

    <div class="panel-label" style="margin-top:.2rem">Histórico</div>
    <div class="hist" id="hist32"></div>

    <button class="btn" onclick="novoJogo_32()">Novo Jogo</button>
  </div>

</div>

<script>
// ── Estado ─────────────────────────────────────────────────────────────────────
let tab32         = Array(9).fill(0);  // 1=X  -1=O  0=vazio
let vez32         = 'X';
let fimJogo32     = false;
let aguardando32  = false;
let jogada32Num   = 0;
let acertos32     = 0;
let total32       = 0;

// ── Helpers ────────────────────────────────────────────────────────────────────
function badgeClass32(estado) {
  if (['X vence', 'O vence'].includes(estado)) return 'b-err';
  if (estado === 'Empate')                      return 'b-warn';
  if (estado === 'Possibilidade de Fim de Jogo') return 'b-info';
  return 'b-neutral';
}

function setMsg32(txt, tipo) {
  const el = document.getElementById('msg32');
  el.textContent = txt;
  el.className = 'msg ' + tipo;
}

function renderBoard32() {
  const bloqueado = fimJogo32 || aguardando32;
  for (let i = 0; i < 9; i++) {
    const el = document.getElementById('c' + i);
    if (tab32[i] === 1) {
      el.textContent = 'X';
      el.className = 'cell x filled';
    } else if (tab32[i] === -1) {
      el.textContent = 'O';
      el.className = 'cell o filled';
    } else {
      el.textContent = '';
      el.className = 'cell' + (bloqueado ? ' disabled' : '');
    }
  }
}

function atualizarAcc32(acertou) {
  total32++;
  if (acertou) acertos32++;
  const pct = total32 > 0 ? (acertos32 / total32 * 100).toFixed(1) : 0;
  document.getElementById('acc32').textContent  = pct + '%';
  document.getElementById('bar32').style.width  = pct + '%';
  document.getElementById('counts32').textContent = acertos32 + ' / ' + total32;
  const last = document.getElementById('last32');
  last.textContent  = acertou ? '✓ Correto' : '✗ Errado';
  last.style.color  = acertou ? 'var(--ok)' : 'var(--err)';
}

function addHist32(num, pred, real, acertou) {
  const hist = document.getElementById('hist32');
  const item = document.createElement('div');
  item.className = 'hist-item';
  item.innerHTML =
    `<span>#${num} &nbsp;SVM: <b>${pred}</b></span>
     <span class="${acertou ? 'ok-tag' : 'err-tag'}">${acertou ? '✓' : '✗'} ${real}</span>`;
  hist.prepend(item);
}

// ── API calls ──────────────────────────────────────────────────────────────────
async function classificar32() {
  const resp = await fetch('/api/jogar_32', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tabuleiro: tab32})
  });
  return await resp.json();  // {predicao_svm, estado_real}
}

async function maquinaJoga32() {
  const resp = await fetch('/api/jogada_maquina_32', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tabuleiro: tab32})
  });
  return await resp.json();  // {posicao}
}

// ── Processa classificação e decide continuação ────────────────────────────────
const FIM_32 = new Set(['Empate', 'O vence', 'X vence']);

async function processarClassificacao32(quem) {
  jogada32Num++;
  const {predicao_svm: pred, estado_real: real} = await classificar32();
  const acertou = pred === real;

  // Atualiza badges
  const predEl = document.getElementById('pred32');
  predEl.textContent = pred;
  predEl.className   = 'badge ' + badgeClass32(pred);

  const realEl = document.getElementById('real32');
  realEl.textContent = real;
  realEl.className   = 'badge ' + badgeClass32(real);

  atualizarAcc32(acertou);
  addHist32(jogada32Num, pred, real, acertou);

  // ─── Regra de encerramento ───────────────────────────────────────────────────
  // Caso 1: SVM detecta fim → jogo encerra (independente de acertar)
  if (FIM_32.has(pred)) {
    fimJogo32 = true;
    renderBoard32();
    document.getElementById('turn32').innerHTML =
      '<span style="color:var(--muted)">Jogo encerrado</span>';

    if (acertou) {
      const msgs = {
        'X vence': ['Você venceu! SVM classificou corretamente "X vence".', 'ok'],
        'O vence': ['Máquina venceu! SVM classificou corretamente "O vence".', 'err'],
        'Empate':  ['Empate! SVM classificou corretamente.', 'warn'],
      };
      setMsg32(...msgs[pred]);
    } else {
      setMsg32(
        `SVM detectou "${pred}" (real: "${real}"). Classificação encerrou o jogo.`,
        'warn'
      );
    }
    return false;  // jogo encerrou
  }

  // Caso 2: estado real é fim, mas SVM não detectou → jogo continua forçado
  if (FIM_32.has(real) && !FIM_32.has(pred)) {
    setMsg32(
      `Estado real é "${real}", mas SVM previu "${pred}". Jogo continua por falha do classificador!`,
      'warn'
    );
    return true;
  }

  // Caso 3: jogo normal
  if (!acertou) {
    setMsg32(`SVM errou: previu "${pred}", real é "${real}". Continue jogando.`, 'warn');
  } else {
    const msgMap = {
      'Tem jogo': 'Jogo em andamento.',
      'Possibilidade de Fim de Jogo': 'Atenção — possibilidade de fim de jogo!'
    };
    setMsg32((msgMap[pred] || 'Sua vez!') + ' SVM acertou.', 'info');
  }
  return true;  // jogo continua
}

// ── Turno da máquina ───────────────────────────────────────────────────────────
async function turnoMaquina32() {
  document.getElementById('turn32').innerHTML =
    '<div class="dot o"></div><span>Máquina jogando...</span>';

  await new Promise(r => setTimeout(r, 500));

  const {posicao} = await maquinaJoga32();
  if (posicao === -1) { fimJogo32 = true; return; }

  tab32[posicao] = -1;
  renderBoard32();

  const continua = await processarClassificacao32('O');
  if (continua) {
    vez32 = 'X';
    aguardando32 = false;
    renderBoard32();
    document.getElementById('turn32').innerHTML =
      '<div class="dot x"></div><span>Sua vez &nbsp;(X)</span>';
  }
}

// ── Jogada do humano ───────────────────────────────────────────────────────────
async function jogada_32(pos) {
  if (fimJogo32 || tab32[pos] !== 0 || vez32 !== 'X' || aguardando32) return;

  aguardando32 = true;
  tab32[pos] = 1;
  vez32 = 'O';
  renderBoard32();

  const continua = await processarClassificacao32('X');
  if (!continua) { aguardando32 = false; return; }

  // Máquina joga
  await turnoMaquina32();
}

// ── Novo jogo ──────────────────────────────────────────────────────────────────
function novoJogo_32() {
  tab32        = Array(9).fill(0);
  vez32        = 'X';
  fimJogo32    = false;
  aguardando32 = false;
  jogada32Num  = 0;
  acertos32    = 0;
  total32      = 0;

  ['pred32','real32'].forEach(id => {
    document.getElementById(id).textContent = '—';
    document.getElementById(id).className   = 'badge b-neutral';
  });
  document.getElementById('acc32').textContent    = '—';
  document.getElementById('bar32').style.width    = '0%';
  document.getElementById('counts32').textContent = '0 / 0';
  document.getElementById('last32').textContent   = '—';
  document.getElementById('last32').style.color   = 'var(--muted)';
  document.getElementById('hist32').innerHTML     = '';
  document.getElementById('turn32').innerHTML =
    '<div class="dot x"></div><span>Sua vez &nbsp;(X)</span>';
  setMsg32('Nova partida. Faça sua jogada!', 'info');

  renderBoard32();
}

// Init
renderBoard32();
</script>
</body>
</html>"""

# ── Rotas ──────────────────────────────────────────────────────────────────────
@app_32.route('/')
def index_32():
    return render_template_string(HTML_32)

@app_32.route('/api/jogar_32', methods=['POST'])
def api_jogar_32():
    data_32       = request.get_json()
    tabuleiro_32  = data_32['tabuleiro']
    predicao_32   = modelo_svm_32.predict([tabuleiro_32])[0]
    real_32       = estado_real_32(tabuleiro_32)
    return jsonify({'predicao_svm': predicao_32, 'estado_real': real_32})

@app_32.route('/api/jogada_maquina_32', methods=['POST'])
def api_maquina_32():
    data_32      = request.get_json()
    tabuleiro_32 = data_32['tabuleiro']
    vazias_32    = [i for i, v in enumerate(tabuleiro_32) if v == 0]
    posicao_32   = random.choice(vazias_32) if vazias_32 else -1
    return jsonify({'posicao': posicao_32})

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "─"*50)
    print("  Tic Tac Toe — IA com SVM")
    print("  http://localhost:5032")
    print("─"*50 + "\n")
    app_32.run(debug=True, port=5032)
