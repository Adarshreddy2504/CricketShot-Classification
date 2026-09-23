import { motion } from "framer-motion";
import { Upload, Database, Video, Layers, Brain, CheckCircle2 } from "lucide-react";
import type { InferenceResult } from "../services/api";

interface Props {
  result?: InferenceResult | null;
}

export function SystemArchitectureBoard({ result }: Props) {
  return (
    <section className="w-full max-w-[90vw] mx-auto py-24 relative z-10" id="system-architecture">
      <div className="text-center mb-16">
        <h2 className="text-4xl md:text-5xl font-black mb-4 tracking-tight">System Architecture</h2>
        <p className="text-lg text-gray-400 max-w-2xl mx-auto font-light">
          End-to-End Pipeline: Client Request → SOTA FastAPI Inference Engine → Visualization
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 relative">
        
        {/* Animated Connections Desktop */}
        <div className="absolute top-1/2 left-[25%] -translate-y-1/2 -translate-x-1/2 z-0 hidden lg:block">
           <motion.div 
             className="flex items-center text-[#00f0ff]"
             animate={{ x: [0, 10, 0], opacity: [0.3, 1, 0.3] }}
             transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
           >
              <div className="h-[2px] w-6 bg-[#00f0ff]" />
              <div className="w-0 h-0 border-y-[6px] border-y-transparent border-l-[8px] border-l-[#00f0ff]" />
           </motion.div>
        </div>
        <div className="absolute top-1/2 left-[75%] -translate-y-1/2 -translate-x-1/2 z-0 hidden lg:block">
           <motion.div 
             className="flex items-center text-[#00ff66]"
             animate={{ x: [0, 10, 0], opacity: [0.3, 1, 0.3] }}
             transition={{ duration: 2, repeat: Infinity, delay: 1, ease: "easeInOut" }}
           >
              <div className="h-[2px] w-6 bg-[#00ff66]" />
              <div className="w-0 h-0 border-y-[6px] border-y-transparent border-l-[8px] border-l-[#00ff66]" />
           </motion.div>
        </div>

        {/* Zone 1: Client */}
        <div className="lg:col-span-1 flex flex-col gap-4 z-10">
           <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-2">1. Client / Frontend</h3>
           <div className="glass bg-[#0b0f19]/60 p-6 rounded-2xl border border-white/10 h-[400px] flex flex-col items-center justify-center relative overflow-hidden shadow-2xl">
             <div className="absolute top-0 w-full h-1 bg-gradient-to-r from-[#00f0ff] to-transparent" />
             <div className="w-24 h-24 rounded-full border-2 border-dashed border-[#00f0ff]/50 flex items-center justify-center mb-8 relative">
                <motion.div 
                  className="absolute inset-0 rounded-full bg-[#00f0ff]/10"
                  animate={{ scale: [1, 1.3, 1], opacity: [0.3, 0.8, 0.3] }}
                  transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
                />
                <Upload className="w-10 h-10 text-[#00f0ff]" />
             </div>
             <div className="w-full bg-white/5 border border-white/10 rounded-lg backdrop-blur-md p-4 font-mono text-xs leading-relaxed shadow-inner">
               <div><span className="text-cyan-400">const</span> <span className="text-gray-300">formData</span> <span className="text-gray-500">=</span> <span className="text-emerald-400">new</span> <span className="text-gray-300">FormData</span><span className="text-gray-500">();</span></div>
               <div className="my-1"><span className="text-gray-300">formData.append</span><span className="text-gray-500">(</span><span className="text-emerald-400">'video'</span><span className="text-gray-500">,</span> <span className="text-gray-300">file</span><span className="text-gray-500">);</span></div>
               <div className="mt-3"><span className="text-cyan-400">await</span> <span className="text-gray-300">axios.post</span><span className="text-gray-500">(</span><span className="text-emerald-400">'/api/predict'</span><span className="text-gray-500">)</span></div>
             </div>
           </div>
        </div>

        {/* Zone 2: Server (FastAPI) & SOTA Pipeline */}
        <div className="lg:col-span-2 flex flex-col gap-4 z-10">
           <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-2">2. FastAPI & SOTA Inference Pipeline</h3>
           <div className="glass bg-[#0b0f19]/60 p-6 rounded-2xl border border-white/10 h-[400px] flex flex-col relative overflow-hidden shadow-[0_0_50px_rgba(0,240,255,0.05)]">
             <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 via-[#00f0ff] to-[#00ff66]" />
             
             {/* API Endpoints */}
             <div className="flex justify-center gap-4 mb-10">
               <div className="px-4 py-1.5 bg-purple-900/30 border border-purple-500/40 rounded-full text-[11px] font-mono text-purple-300 shadow-[0_0_15px_rgba(168,85,247,0.25)]">POST /api/predict</div>
               <div className="px-4 py-1.5 bg-purple-900/30 border border-purple-500/40 rounded-full text-[11px] font-mono text-purple-300">GET /api/health</div>
               <div className="px-4 py-1.5 bg-purple-900/30 border border-purple-500/40 rounded-full text-[11px] font-mono text-purple-300">GET /api/visualization</div>
             </div>

             {/* The Pipeline */}
             <div className="flex-1 flex flex-col justify-center relative py-4">
                {/* Connecting Pulse Line (Absolute behind nodes) */}
                <div className="absolute left-8 right-8 top-[2.75rem] h-[2px] bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 z-[-1] overflow-hidden rounded-full">
                   <motion.div 
                     className="w-1/3 h-full bg-white shadow-[0_0_15px_white]"
                     animate={{ x: ["-100%", "300%"] }}
                     transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                   />
                </div>

                {/* Spatial Attention Node (Pinned to Conduit) */}
                <div className="absolute top-[2.75rem] -translate-y-1/2 left-[60%] -translate-x-1/2 w-8 h-8 rounded-full bg-red-500/20 border-2 border-red-500 z-20 flex items-center justify-center shadow-[0_0_20px_rgba(239,68,68,0.7)] backdrop-blur-sm animate-pulse" title="Spatial Attention">
                   <div className="w-2 h-2 bg-red-400 rounded-full" />
                </div>

                {/* Temporal Attention Node (Pinned to Conduit) */}
                <div className="absolute top-[2.75rem] -translate-y-1/2 left-[80%] -translate-x-1/2 w-8 h-8 rounded-full bg-purple-500/20 border-2 border-purple-500 z-20 flex items-center justify-center shadow-[0_0_20px_rgba(168,85,247,0.7)] backdrop-blur-sm animate-pulse" title="Temporal Attention">
                   <div className="w-2 h-2 bg-purple-400 rounded-full" />
                </div>

                <div className="grid grid-cols-5 gap-2 w-full relative z-10 pt-4">
                  {/* OpenCV */}
                  <div className="flex flex-col items-center justify-start group">
                    <div className="w-14 h-14 rounded-xl bg-white/5 backdrop-blur-md border border-white/20 flex items-center justify-center mb-2 shadow-lg shrink-0">
                      <Video className="w-6 h-6 text-gray-300" />
                    </div>
                    <span className="text-[11px] font-bold text-gray-200">OpenCV</span>
                    <div className="mt-4 w-full bg-white/5 border border-white/10 rounded-md px-1 py-3 text-center flex flex-col gap-1">
                       <span className="text-[9px] font-mono text-gray-400 leading-tight">30 Frames Extracted</span>
                    </div>
                  </div>

                  {/* Preprocessing */}
                  <div className="flex flex-col items-center justify-start group">
                    <div className="w-14 h-14 rounded-xl bg-[#0b0f19] border border-cyan-500/50 flex items-center justify-center mb-2 shadow-[0_0_20px_rgba(0,240,255,0.2)] shrink-0">
                      <Layers className="w-6 h-6 text-[#00f0ff]" />
                    </div>
                    <span className="text-[11px] font-bold text-[#00f0ff]">Preprocess</span>
                    <div className="mt-4 w-full bg-white/5 border border-white/10 rounded-md px-1 py-3 text-center flex flex-col gap-1">
                       <span className="text-[8px] font-mono text-cyan-300">Resize: 224x224</span>
                       <span className="text-[8px] font-mono text-cyan-400">Tensor: (B, 30, 3, 224, 224)</span>
                    </div>
                  </div>

                  {/* EfficientNet-B0 */}
                  <div className="flex flex-col items-center justify-start">
                    <div className="w-16 h-16 rounded-xl bg-[#0b0f19] border border-cyan-400 flex items-center justify-center mb-1 shadow-[0_0_25px_rgba(0,240,255,0.3)] shrink-0">
                      <span className="text-[11px] font-bold text-white text-center leading-tight">EfficientNet<br/>B0</span>
                    </div>
                    <div className="mt-4 w-full bg-white/5 border border-white/10 rounded-md px-1 py-3 text-center flex flex-col gap-1">
                       <span className="text-[9px] font-mono text-cyan-300">1280-D Features</span>
                    </div>
                  </div>

                  {/* Bi-GRU */}
                  <div className="flex flex-col items-center justify-start">
                    <div className="w-16 h-16 rounded-xl bg-[#0b0f19] border border-emerald-400 flex items-center justify-center mb-1 shadow-[0_0_25px_rgba(0,255,102,0.3)] shrink-0">
                      <span className="text-[11px] font-bold text-white text-center leading-tight">2-Layer<br/>Bi-GRU</span>
                    </div>
                    <div className="mt-4 w-full bg-white/5 border border-white/10 rounded-md px-1 py-3 text-center flex flex-col gap-1">
                       <span className="text-[9px] font-mono text-emerald-300">hidden=256</span>
                       <span className="text-[8px] font-mono text-emerald-400">(B, 30, 512)</span>
                    </div>
                  </div>

                  {/* Classification Head */}
                  <div className="flex flex-col items-center justify-start">
                    <div className="w-14 h-14 rounded-xl bg-white/5 backdrop-blur-md border border-white/20 flex items-center justify-center mb-2 shadow-lg shrink-0">
                      <Brain className="w-6 h-6 text-gray-200" />
                    </div>
                    <span className="text-[11px] font-bold text-gray-200">Classifier</span>
                    <div className="mt-4 w-full bg-white/5 border border-white/10 rounded-md px-1 py-3 text-center flex flex-col gap-1">
                       <span className="text-[9px] font-mono text-gray-400">Out: (B, 10)</span>
                    </div>
                  </div>
                </div>
             </div>
           </div>
        </div>

        {/* Zone 3 & 4 Container */}
        <div className="lg:col-span-1 flex flex-col gap-6 z-10">
           
           {/* Zone 3: Model Repository */}
           <div className="flex flex-col gap-2 h-[150px]">
             <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-1">3. Model Repo</h3>
             <div className="glass bg-[#0b0f19]/60 p-5 rounded-2xl border border-white/10 flex items-center gap-5 h-full shadow-lg">
                <div className="w-14 h-14 rounded-xl bg-blue-900/30 border border-blue-500/40 flex items-center justify-center shadow-inner">
                   <Database className="w-6 h-6 text-blue-400" />
                </div>
                <div className="flex flex-col gap-1.5">
                   <span className="text-[11px] font-mono text-blue-300 flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-emerald-500"/> weights_v2.ckpt</span>
                   <span className="text-[11px] font-mono text-blue-300 flex items-center gap-2"><CheckCircle2 className="w-3.5 h-3.5 text-emerald-500"/> config.yaml</span>
                   <span className="text-[10px] font-mono text-gray-400 ml-5.5 pl-5">↳ 10 Classes</span>
                </div>
             </div>
           </div>

           {/* Zone 4: Output & Viz */}
           <div className="flex flex-col gap-2 flex-1">
             <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-1">4. Output</h3>
             <div className="glass bg-[#0b0f19]/60 p-6 rounded-2xl border border-[#00ff66]/40 flex-1 flex flex-col relative overflow-hidden shadow-[0_0_30px_rgba(0,255,102,0.15)]">
                <div className="absolute top-0 right-0 w-32 h-32 bg-[#00ff66] opacity-10 blur-[50px] rounded-full pointer-events-none" />
                
                <div className="bg-white/5 border border-white/10 rounded-lg backdrop-blur-md p-4 flex-1 flex flex-col justify-center z-10 shadow-inner font-mono text-xs leading-relaxed">
                  <div className="text-gray-500 mb-2">{"{"}</div>
                  <div className="pl-4 border-l border-white/5 flex flex-col gap-1">
                    <div>
                      <span className="text-cyan-400">"prediction"</span><span className="text-gray-500">: </span>
                      <span className="text-emerald-400">"{result ? result.predictedClass : "Hook Shot"}"</span><span className="text-gray-500">,</span>
                    </div>
                    <div className="mb-3">
                      <span className="text-cyan-400">"confidence"</span><span className="text-gray-500">: </span>
                      <span className="text-emerald-400">{result ? (result.confidence / 100).toFixed(3) : "0.923"}</span><span className="text-gray-500">,</span>
                    </div>
                    
                    {/* Mini probability bars */}
                    <div className="flex flex-col gap-1.5 w-full pr-2">
                       {result && result.class_probabilities ? Object.entries(result.class_probabilities).slice(0, 3).map(([cls, prob], idx) => (
                         <div key={cls} className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden flex" title={`${cls}: ${(prob * 100).toFixed(1)}%`}>
                            <div 
                              className={`h-full ${idx === 0 ? 'bg-emerald-400 shadow-[0_0_10px_#00ff66]' : 'bg-gray-400'}`} 
                              style={{ width: `${prob * 100}%` }} 
                            />
                         </div>
                       )) : (
                         <>
                           <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden flex">
                              <div className="h-full bg-emerald-400 w-[92%] shadow-[0_0_10px_#00ff66]" />
                           </div>
                           <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden flex">
                              <div className="h-full bg-gray-400 w-[5%]" />
                           </div>
                           <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden flex">
                              <div className="h-full bg-gray-400 w-[2%]" />
                           </div>
                         </>
                       )}
                    </div>
                  </div>
                  <div className="text-gray-500 mt-2">{"}"}</div>
                </div>
                
                {/* Frame timeline mock */}
                <div className="absolute bottom-0 left-0 right-0 h-10 bg-white/5 backdrop-blur-lg flex items-center justify-between px-3 gap-1.5 border-t border-[#00ff66]/30 z-10">
                   {[...Array(8)].map((_, i) => (
                     <div key={i} className="flex-1 h-4 bg-[#00ff66]/20 border border-[#00ff66]/40 rounded-[2px]" />
                   ))}
                </div>
             </div>
           </div>

        </div>
      </div>
    </section>
  );
}
