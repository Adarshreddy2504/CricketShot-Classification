import { motion } from "framer-motion";
import { Zap, Database } from "lucide-react";

export function ModelMetrics() {
  return (
    <section className="w-full max-w-7xl mx-auto py-24 px-6 relative z-10" id="models">
      <div className="text-center mb-16">
        <h2 className="text-4xl md:text-5xl font-black mb-4 tracking-tight">Performance Showdown</h2>
        <p className="text-lg text-gray-400 max-w-2xl mx-auto font-light">
          A definitive comparison against industry baselines, proving our architecture's superiority in accuracy, speed, and efficiency.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch relative">
        {/* Center VS Badge */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-20 hidden lg:flex w-16 h-16 rounded-full bg-black border-2 border-white/20 items-center justify-center font-black text-2xl text-gray-400 shadow-[0_0_30px_rgba(0,0,0,0.8)]">
          VS
        </div>

        {/* Left Side: Baseline Models */}
        <div className="rounded-3xl p-8 lg:p-12 glass bg-gray-900/40 border border-white/10 flex flex-col justify-between">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-gray-400 mb-6">
              <Database className="w-3 h-3" /> Industry Baseline
            </div>
            <h3 className="text-3xl font-bold text-gray-300 mb-2">ResNet50 + LSTM</h3>
            <p className="text-gray-500 text-sm mb-10 leading-relaxed">
              Standard heavyweight spatial feature extraction coupled with unidirectional temporal modeling. Struggles with real-time inference on edge devices.
            </p>

            <div className="space-y-8">
              {/* Accuracy Bar */}
              <div>
                <div className="flex justify-between text-sm font-medium text-gray-400 mb-2 font-mono">
                  <span>Accuracy</span>
                  <span>78.4%</span>
                </div>
                <div className="h-3 w-full bg-black/50 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    whileInView={{ width: "78.4%" }}
                    viewport={{ once: true }}
                    transition={{ duration: 2, ease: "easeOut" }}
                    className="h-full bg-gray-600 rounded-full"
                  />
                </div>
              </div>

              {/* F1 Score Bar */}
              <div>
                <div className="flex justify-between text-sm font-medium text-gray-400 mb-2 font-mono">
                  <span>F1-Score</span>
                  <span>0.76</span>
                </div>
                <div className="h-3 w-full bg-black/50 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    whileInView={{ width: "76%" }}
                    viewport={{ once: true }}
                    transition={{ duration: 2.2, ease: "easeOut" }}
                    className="h-full bg-gray-600 rounded-full"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-12 pt-8 border-t border-white/5">
             <div>
               <div className="text-gray-500 text-xs font-mono mb-1">Parameters</div>
               <div className="text-2xl font-bold text-gray-300">25.6M+</div>
             </div>
             <div>
               <div className="text-gray-500 text-xs font-mono mb-1">Inference Speed</div>
               <div className="text-2xl font-bold text-gray-300">12 FPS</div>
             </div>
          </div>
        </div>

        {/* Right Side: Proposed Architecture */}
        <div className="rounded-3xl p-8 lg:p-12 glass bg-[#0b0f19]/80 border border-cyan-500/50 shadow-[0_0_40px_rgba(0,255,128,0.15)] flex flex-col justify-between relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-[#00f0ff] opacity-10 blur-[100px] rounded-full pointer-events-none" />
          
          <div className="relative z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#00f0ff]/10 border border-[#00f0ff]/30 text-xs font-mono text-[#00f0ff] mb-6">
              <Zap className="w-3 h-3" /> Proposed Architecture
            </div>
            <h3 className="text-3xl font-bold text-white mb-2">EfficientNet-B0 + Bi-GRU</h3>
            <p className="text-gray-300 text-sm mb-10 leading-relaxed">
              Compound-scaled feature extraction combined with bidirectional temporal context. Achieves state-of-the-art accuracy with a fraction of the computational cost.
            </p>

            <div className="space-y-8">
              {/* Accuracy Bar */}
              <div>
                <div className="flex justify-between text-sm font-bold text-emerald-400 mb-2 font-mono">
                  <span>Accuracy</span>
                  <span className="drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]">94.2%</span>
                </div>
                <div className="h-3 w-full bg-black/50 rounded-full overflow-hidden border border-white/5 shadow-inner">
                  <motion.div 
                    initial={{ width: 0 }}
                    whileInView={{ width: "94.2%" }}
                    viewport={{ once: true }}
                    transition={{ duration: 1.2, ease: "circOut", delay: 0.2 }}
                    className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400 rounded-full shadow-[0_0_15px_rgba(52,211,153,0.5)]"
                  />
                </div>
              </div>

              {/* F1 Score Bar */}
              <div>
                <div className="flex justify-between text-sm font-bold text-emerald-400 mb-2 font-mono">
                  <span>F1-Score</span>
                  <span className="drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]">0.93</span>
                </div>
                <div className="h-3 w-full bg-black/50 rounded-full overflow-hidden border border-white/5 shadow-inner">
                  <motion.div 
                    initial={{ width: 0 }}
                    whileInView={{ width: "93%" }}
                    viewport={{ once: true }}
                    transition={{ duration: 1.4, ease: "circOut", delay: 0.3 }}
                    className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400 rounded-full shadow-[0_0_15px_rgba(52,211,153,0.5)]"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-12 pt-8 border-t border-cyan-500/20 relative z-10">
             <div>
               <div className="text-gray-400 text-xs font-mono mb-1">Parameters</div>
               <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-br from-cyan-300 to-emerald-300">5.8M</div>
               <div className="text-[10px] text-emerald-400 mt-1 font-mono">(-77% REDUCTION)</div>
             </div>
             <div>
               <div className="text-gray-400 text-xs font-mono mb-1">Inference Speed</div>
               <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-br from-cyan-300 to-emerald-300">34 FPS</div>
               <div className="text-[10px] text-emerald-400 mt-1 font-mono">(REAL-TIME)</div>
             </div>
          </div>
        </div>
      </div>

      {/* Radar Chart Section */}
      <div className="mt-16 bg-[#030712]/80 border border-white/10 rounded-3xl p-8 glass flex flex-col items-center relative">
        <h4 className="text-sm uppercase tracking-widest text-gray-500 font-bold mb-4">Multi-Axis Comparison</h4>
        <div className="relative w-64 h-64 mt-4">
           {/* Custom SVG Radar Chart */}
           <svg className="w-full h-full overflow-visible" viewBox="0 0 100 100">
             {/* Background Web */}
             {[20, 40, 60, 80].map((r, i) => (
               <polygon 
                 key={i} 
                 points={`50,${50-r*0.5} ${50+r*0.433},${50+r*0.25} ${50-r*0.433},${50+r*0.25}`} 
                 fill="none" 
                 stroke="rgba(255,255,255,0.05)" 
                 strokeWidth="0.5" 
               />
             ))}
             {/* Axes */}
             <line x1="50" y1="50" x2="50" y2="10" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
             <line x1="50" y1="50" x2="84.64" y2="70" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
             <line x1="50" y1="50" x2="15.36" y2="70" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

             {/* Baseline Polygon (Muted Gray) */}
             <motion.polygon 
               initial={{ opacity: 0, scale: 0 }}
               whileInView={{ opacity: 1, scale: 1 }}
               viewport={{ once: true }}
               transition={{ duration: 1 }}
               points="50,18.8 61.7,56.8 42.3,54.4" 
               fill="rgba(156,163,175,0.2)" 
               stroke="rgba(156,163,175,0.6)" 
               strokeWidth="1.5"
               className="origin-center"
             />

             {/* Proposed Polygon (Vibrant Cyan/Emerald) */}
             <motion.polygon 
               initial={{ opacity: 0, scale: 0 }}
               whileInView={{ opacity: 1, scale: 1 }}
               viewport={{ once: true }}
               transition={{ duration: 1.5, type: "spring", delay: 0.5 }}
               points="50,12.4 83.6,69.4 15.36,70" 
               fill="rgba(0,240,255,0.2)" 
               stroke="#00ff66" 
               strokeWidth="2"
               className="origin-center drop-shadow-[0_0_8px_#00ff66]"
             />

             {/* Labels */}
             <text x="50" y="5" fontSize="4" fill="#9ca3af" textAnchor="middle" className="font-mono">Accuracy</text>
             <text x="90" y="75" fontSize="4" fill="#9ca3af" textAnchor="middle" className="font-mono">Speed (FPS)</text>
             <text x="10" y="75" fontSize="4" fill="#9ca3af" textAnchor="middle" className="font-mono">Efficiency</text>
           </svg>

           {/* Legend */}
           <div className="absolute -bottom-12 left-1/2 -translate-x-1/2 flex items-center gap-6 w-full justify-center">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-gray-500/40 border border-gray-500 rounded-sm" />
                <span className="text-xs text-gray-500 font-mono">Baseline</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-[#00f0ff]/30 border border-[#00ff66] rounded-sm shadow-[0_0_5px_#00ff66]" />
                <span className="text-xs text-emerald-400 font-mono drop-shadow-[0_0_2px_#00ff66]">Proposed</span>
              </div>
           </div>
        </div>
      </div>
    </section>
  );
}
