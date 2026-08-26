
// Chart setup
const ctx = document.getElementById('rlChart').getContext('2d');
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = 'Inter';

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Reward',
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
                data: [],
                yAxisID: 'y'
            },
            {
                label: 'KL Divergence',
                borderColor: '#3b82f6',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                tension: 0.4,
                data: [],
                yAxisID: 'y1'
            },
            {
                label: 'Entropy',
                borderColor: '#f43f5e',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
                data: [],
                yAxisID: 'y2'
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
            x: { grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { 
                type: 'linear', display: true, position: 'left',
                grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y1: { 
                type: 'linear', display: false, position: 'right',
                min: 0, max: 10
            },
            y2: {
                type: 'linear', display: false, position: 'right',
                min: 0, max: 2
            }
        },
        plugins: {
            legend: { position: 'top' }
        }
    }
});

// Simulation State
let isRunning = false;
let step = 0;
let loopId = null;

let currentReward = 0;
let currentKL = 0.1;
let currentEntropy = 1.0;

// Anomalies
let activeAnomaly = null;
let anomalyStepStart = 0;

// DOM
const terminal = document.getElementById('terminal');
const statReward = document.getElementById('stat-reward');
const statKl = document.getElementById('stat-kl');
const statEntropy = document.getElementById('stat-entropy');

function logTerminal(message, type = 'info') {
    const div = document.createElement('div');
    div.className = 'log-enter break-words';
    
    const time = new Date().toISOString().split('T')[1].slice(0,-1);
    
    let colorClass = 'text-slate-300';
    if (type === 'critical') colorClass = 'text-rose-400 font-bold';
    if (type === 'warn') colorClass = 'text-orange-400';
    if (type === 'success') colorClass = 'text-emerald-400';
    
    div.innerHTML = `<span class="text-slate-600">[${time}]</span> <span class="${colorClass}">${message}</span>`;
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight;
}

function updateSimulation() {
    step++;
    
    // Base healthy drift
    currentReward += (Math.random() * 0.1) - 0.02;
    currentKL = Math.max(0, currentKL + (Math.random() * 0.02 - 0.01));
    currentEntropy = Math.max(0, currentEntropy - 0.001); // slow natural decay
    
    // Apply anomalies
    if (activeAnomaly === 'hack') {
        currentReward += 2.5; // unnatural spike
        currentKL += 0.5;
    } else if (activeAnomaly === 'collapse') {
        currentEntropy -= 0.1;
        currentKL += 0.1;
        currentEntropy = Math.max(0, currentEntropy);
    } else if (activeAnomaly === 'kl') {
        currentKL += 1.2;
        currentReward -= 0.5;
    }
    
    // Chart update
    chart.data.labels.push(step);
    chart.data.datasets[0].data.push(currentReward);
    chart.data.datasets[1].data.push(currentKL);
    chart.data.datasets[2].data.push(currentEntropy);
    
    if (chart.data.labels.length > 50) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(d => d.data.shift());
    }
    
    chart.update();
    
    // DOM stats
    statReward.innerText = currentReward.toFixed(2);
    statKl.innerText = currentKL.toFixed(2);
    statEntropy.innerText = currentEntropy.toFixed(2);
    
    // SyntropyRL Detector Logic (Simulated)
    if (activeAnomaly && (step - anomalyStepStart) === 5) { // Catch it fast
        if (activeAnomaly === 'hack') {
            logTerminal(`[RLD-014] REWARD HACKING DETECTED!<br>Severity: CRITICAL<br>Details: Reward gradient > 500% over baseline while KL divergence spiked rapidly. Policy is exploiting proxy logic.`, 'critical');
        } else if (activeAnomaly === 'collapse') {
            logTerminal(`[RLD-019] DISTRIBUTIONAL SHIFT!<br>Severity: CRITICAL<br>Details: Policy Entropy collapsed by >30% in window. The agent has converged to a deterministic, catastrophic sub-optimal policy.`, 'critical');
        } else if (activeAnomaly === 'kl') {
            logTerminal(`[RLD-002] KL DIVERGENCE BOUND BREACHED!<br>Severity: WARN<br>Details: Reference model log-probability deviation exceeded trust region bounds (TRPO/PPO constraint failure).`, 'warn');
        }
        activeAnomaly = null; // reset so it doesn't spam
    }
}

document.getElementById('btn-start').onclick = () => {
    if (isRunning) return;
    isRunning = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-start').classList.add('opacity-50');
    document.getElementById('btn-stop').disabled = false;
    logTerminal('Starting PPO Training simulation...', 'info');
    loopId = setInterval(updateSimulation, 200);
};

document.getElementById('btn-stop').onclick = () => {
    isRunning = false;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').classList.remove('opacity-50');
    document.getElementById('btn-stop').disabled = true;
    clearInterval(loopId);
    logTerminal('Training paused.', 'info');
};

document.getElementById('btn-hack').onclick = () => {
    if (!isRunning) return logTerminal('Start training first.', 'warn');
    activeAnomaly = 'hack';
    anomalyStepStart = step;
    logTerminal('Injecting simulated reward hacking topology...', 'info');
};

document.getElementById('btn-collapse').onclick = () => {
    if (!isRunning) return logTerminal('Start training first.', 'warn');
    activeAnomaly = 'collapse';
    anomalyStepStart = step;
    logTerminal('Injecting sudden policy entropy collapse...', 'info');
};

document.getElementById('btn-kl').onclick = () => {
    if (!isRunning) return logTerminal('Start training first.', 'warn');
    activeAnomaly = 'kl';
    anomalyStepStart = step;
    logTerminal('Simulating massive reference model divergence...', 'info');
};
