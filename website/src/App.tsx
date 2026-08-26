import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, ShieldAlert, Terminal, Play, Square, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function App() {
  const [data, setData] = useState<any[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>(['[System] SyntropyRL Telemetry Engine Ready.']);

  useEffect(() => {
    if (!isRunning) return;
    let step = data.length;
    let currentReward = data.length ? data[data.length-1].reward : 0;
    let currentKL = data.length ? data[data.length-1].kl : 0.1;
    
    const interval = setInterval(() => {
      step++;
      currentReward += (Math.random() * 0.1) - 0.02;
      currentKL = Math.max(0, currentKL + (Math.random() * 0.02 - 0.01));
      
      setData(prev => {
        const next = [...prev, { step, reward: currentReward, kl: currentKL }];
        if (next.length > 50) next.shift();
        return next;
      });
    }, 200);
    return () => clearInterval(interval);
  }, [isRunning, data]);

  const injectAnomaly = () => {
    if (!isRunning) return;
    setLogs(prev => [...prev, '[Alert] Injecting Reward Hacking Topology...']);
    setTimeout(() => {
      setData(prev => prev.map((d, i) => i === prev.length - 1 ? { ...d, reward: d.reward + 2.5, kl: d.kl + 0.8 } : d));
      setLogs(prev => [...prev, '[RLD-014] CRITICAL: Reward Hacking Detected! KL Divergence spiked.']);
    }, 600);
  };

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background */}
      <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] rounded-full bg-purple-600/20 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-blue-600/20 blur-[120px] pointer-events-none" />

      {/* Nav */}
      <nav className="glass sticky top-0 z-50 px-8 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2 font-bold text-xl">
          <Activity className="text-purple-400" /> SyntropyRL
        </div>
        <div className="flex gap-6 text-sm font-medium text-slate-300">
          <a href="#" className="hover:text-white transition-colors">Docs</a>
          <a href="https://github.com/DARREN-2000/SyntropyRL" className="hover:text-white transition-colors">GitHub</a>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-8 grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
        
        {/* Hero Left */}
        <div className="lg:col-span-5 flex flex-col justify-center space-y-6">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-300 text-sm font-medium w-fit">
            <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" /> Live Telemetry Demo
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="text-5xl font-bold tracking-tight leading-tight">
            Diagnose <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400">RL Failures</span> in Real-Time.
          </motion.h1>
          <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="text-lg text-slate-400">
            A lightweight, fault-tolerant diagnostic library for catching KL-divergence and reward hacking in post-training runs.
          </motion.p>
          
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass p-6 rounded-2xl space-y-4">
            <h3 className="font-semibold flex items-center gap-2"><Terminal size={18} /> Interactive Playground</h3>
            <div className="flex gap-4">
              <button onClick={() => setIsRunning(!isRunning)} className={`flex-1 py-2 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors ${isRunning ? 'bg-slate-800 text-white hover:bg-slate-700' : 'bg-white text-black hover:bg-slate-200'}`}>
                {isRunning ? <><Square size={16}/> Pause</> : <><Play size={16}/> Start Training</>}
              </button>
              <button onClick={injectAnomaly} disabled={!isRunning} className="flex-1 bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20 py-2 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors disabled:opacity-50">
                <AlertTriangle size={16}/> Inject Failure
              </button>
            </div>
          </motion.div>
        </div>

        {/* Dashboard Right */}
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }} className="lg:col-span-7 flex flex-col gap-6">
          <div className="glass rounded-2xl p-6 h-[350px] flex flex-col">
            <h3 className="font-semibold text-slate-300 mb-4 flex items-center gap-2">
              <Activity size={18} className="text-blue-400" /> Reward vs KL Divergence
            </h3>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                  <XAxis dataKey="step" hide />
                  <YAxis yAxisId="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                  <Line yAxisId="left" type="monotone" dataKey="reward" stroke="#8b5cf6" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line yAxisId="right" type="monotone" dataKey="kl" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass rounded-2xl p-6 flex-1 flex flex-col min-h-[200px]">
            <h3 className="font-semibold text-slate-300 mb-4 flex items-center gap-2">
              <ShieldAlert size={18} className="text-rose-400" /> SyntropyRL Audit Logs
            </h3>
            <div className="flex-1 overflow-y-auto font-mono text-sm space-y-2 bg-[#09090b] p-4 rounded-lg border border-white/5">
              {logs.map((log, i) => (
                <div key={i} className={`${log.includes('CRITICAL') ? 'text-rose-400 font-bold' : log.includes('Alert') ? 'text-amber-400' : 'text-slate-400'}`}>
                  {log}
                </div>
              ))}
            </div>
          </div>
        </motion.div>

      </main>
    </div>
  );
}