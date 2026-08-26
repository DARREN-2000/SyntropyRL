
const ctx = document.getElementById('rlChart').getContext('2d');
Chart.defaults.font.family = 'Inter';

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Reward', borderColor: '#0ea5e9', borderWidth: 2, pointRadius: 0, tension: 0.1, data: [], yAxisID: 'y' },
            { label: 'KL Div', borderColor: '#64748b', borderDash: [4, 4], borderWidth: 2, pointRadius: 0, tension: 0.1, data: [], yAxisID: 'y1' },
            { label: 'Entropy', borderColor: '#f43f5e', borderWidth: 2, pointRadius: 0, tension: 0.1, data: [], yAxisID: 'y2' }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
            x: { grid: { color: '#f1f5f9' } },
            y: { display: true, position: 'left', grid: { color: '#f1f5f9' } },
            y1: { display: false, min: 0, max: 10 },
            y2: { display: false, min: 0, max: 2 }
        }
    }
});

let isRunning = false;
let step = 0;
let loopId = null;
let currentReward = 0;
let currentKL = 0.1;
let currentEntropy = 1.0;
let activeAnomaly = null;
let anomalyStepStart = 0;
const terminal = document.getElementById('terminal');

function logTerminal(message, type = 'info') {
    const div = document.createElement('div');
    const time = new Date().toISOString().split('T')[1].slice(0,-1);
    let colorClass = 'text-slate-300';
    if (type === 'critical') colorClass = 'text-red-400 font-bold';
    if (type === 'warn') colorClass = 'text-amber-400';
    if (type === 'success') colorClass = 'text-emerald-400';
    div.innerHTML = `<span class="text-slate-500">[${time}]</span> <span class="${colorClass}">${message}</span>`;
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight;
}

function updateSimulation() {
    step++;
    currentReward += (Math.random() * 0.1) - 0.02;
    currentKL = Math.max(0, currentKL + (Math.random() * 0.02 - 0.01));
    currentEntropy = Math.max(0, currentEntropy - 0.001);
    
    if (activeAnomaly === 'hack') { currentReward += 2.5; currentKL += 0.5; } 
    else if (activeAnomaly === 'collapse') { currentEntropy -= 0.1; currentKL += 0.1; currentEntropy = Math.max(0, currentEntropy); } 
    else if (activeAnomaly === 'kl') { currentKL += 1.2; currentReward -= 0.5; }
    
    chart.data.labels.push(step);
    chart.data.datasets[0].data.push(currentReward);
    chart.data.datasets[1].data.push(currentKL);
    chart.data.datasets[2].data.push(currentEntropy);
    
    if (chart.data.labels.length > 50) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(d => d.data.shift());
    }
    chart.update();
    
    if (activeAnomaly && (step - anomalyStepStart) === 5) {
        if (activeAnomaly === 'hack') logTerminal(`[RLD-014] REWARD HACKING DETECTED! Severity: CRITICAL. Proxy logic exploitation identified.`, 'critical');
        else if (activeAnomaly === 'collapse') logTerminal(`[RLD-019] DISTRIBUTIONAL SHIFT! Severity: CRITICAL. Policy Entropy collapsed by >30%.`, 'critical');
        else if (activeAnomaly === 'kl') logTerminal(`[RLD-002] KL DIVERGENCE BOUND BREACHED! Severity: WARN. Trust region bounds violated.`, 'warn');
        activeAnomaly = null;
    }
}

document.getElementById('btn-start').onclick = () => {
    if (isRunning) return;
    isRunning = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    logTerminal('Telemetry streaming initiated.', 'info');
    loopId = setInterval(updateSimulation, 200);
};

document.getElementById('btn-stop').onclick = () => {
    isRunning = false;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    clearInterval(loopId);
    logTerminal('Telemetry stream paused.', 'info');
};

document.getElementById('btn-hack').onclick = () => {
    if (!isRunning) return logTerminal('Cannot inject: stream inactive.', 'warn');
    activeAnomaly = 'hack'; anomalyStepStart = step;
    logTerminal('Simulating reward hacking topology...', 'info');
};
document.getElementById('btn-collapse').onclick = () => {
    if (!isRunning) return logTerminal('Cannot inject: stream inactive.', 'warn');
    activeAnomaly = 'collapse'; anomalyStepStart = step;
    logTerminal('Simulating sudden policy entropy collapse...', 'info');
};
document.getElementById('btn-kl').onclick = () => {
    if (!isRunning) return logTerminal('Cannot inject: stream inactive.', 'warn');
    activeAnomaly = 'kl'; anomalyStepStart = step;
    logTerminal('Simulating reference model divergence...', 'info');
};
