import { motion } from "framer-motion";

export function BiGruVisualizer() {
  const steps = [1, 2, 3, 4, 5];

  return (
    <div className="relative w-full h-full flex flex-col justify-between flex-1 pt-6 pb-2">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(0,255,102,0.05)_0%,transparent_70%)] pointer-events-none" />

      {/* Top: Concat & Softmax Centerpiece */}
      <div className="flex flex-col items-center w-full z-20 relative mt-2">
        <div className="px-8 py-4 border-2 border-white/20 bg-gradient-to-r from-[#00f0ff]/10 via-white/5 to-[#00ff66]/10 rounded-xl text-center shadow-[0_0_30px_rgba(255,255,255,0.1)] backdrop-blur-md relative">
          <span className="font-mono text-[13px] text-gray-200 tracking-wider block font-bold mb-2">Concat [h_fwd, h_bwd]</span>
          <span className="font-mono text-[12px] text-white tracking-widest block bg-white/10 py-1.5 px-4 rounded-md shadow-inner">→ Softmax (10 Classes)</span>
        </div>
        
        {/* Converging Paths from Forward(5) and Backward(1) */}
        <svg className="absolute top-full left-0 w-full h-16 pointer-events-none z-0" preserveAspectRatio="none">
           {/* Cyan from right */}
           <path d="M 85% 100% Q 70% 20% 50% 0%" stroke="#00f0ff" strokeWidth="3" fill="none" className="drop-shadow-[0_0_8px_#00f0ff]" />
           {/* Emerald from left */}
           <path d="M 15% 100% Q 30% 20% 50% 0%" stroke="#00ff66" strokeWidth="3" fill="none" className="drop-shadow-[0_0_8px_#00ff66]" />
        </svg>
      </div>

      <div className="relative w-full flex flex-col gap-8 px-2 sm:px-6 mt-16 mb-4">
        
        {/* Forward Stream (Cyan) */}
        <div className="flex justify-between items-center relative w-full">
          {/* Static thick track */}
          <div className="absolute left-6 right-6 h-1 bg-[#00f0ff]/20 z-0 rounded-full" />
          
          {/* Animated pulsing arrow left-to-right */}
          <motion.div
            className="absolute h-1.5 bg-[#00f0ff] shadow-[0_0_15px_#00f0ff] rounded-full z-0"
            animate={{ left: ["0%", "100%"], width: ["0%", "40%", "0%"] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
          
          {steps.map((t) => (
            <div key={t} className="w-16 h-12 border-2 border-[#00f0ff] bg-[#00f0ff]/10 rounded-lg flex items-center justify-center z-10 shadow-[0_0_15px_rgba(0,240,255,0.25)] relative backdrop-blur-md">
              <span className="font-mono text-[14px] text-[#00f0ff] font-bold tracking-wider">h→{t}</span>
              {t < 5 && <span className="absolute -right-8 text-[#00f0ff] text-[18px] font-black drop-shadow-[0_0_5px_#00f0ff]">→</span>}
            </div>
          ))}
        </div>

        {/* Backward Stream (Emerald) */}
        <div className="flex justify-between items-center relative w-full">
          {/* Static thick track */}
          <div className="absolute left-6 right-6 h-1 bg-[#00ff66]/20 z-0 rounded-full" />
          
          {/* Animated pulsing arrow right-to-left */}
          <motion.div
            className="absolute h-1.5 bg-[#00ff66] shadow-[0_0_15px_#00ff66] rounded-full z-0"
            animate={{ right: ["0%", "100%"], width: ["0%", "40%", "0%"] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
          
          {steps.map((t) => (
            <div key={t} className="w-16 h-12 border-2 border-[#00ff66] bg-[#00ff66]/10 rounded-lg flex items-center justify-center z-10 shadow-[0_0_15px_rgba(0,255,102,0.25)] relative backdrop-blur-md">
              <span className="font-mono text-[14px] text-[#00ff66] font-bold tracking-wider">←h{t}</span>
              {t > 1 && <span className="absolute -left-8 text-[#00ff66] text-[18px] font-black drop-shadow-[0_0_5px_#00ff66]">←</span>}
            </div>
          ))}
        </div>

        {/* Input Row */}
        <div className="flex justify-between items-center relative w-full mt-6">
          {steps.map((t) => (
            <div key={t} className="flex flex-col items-center relative z-10 w-16">
              <div className="absolute -top-16 w-1 h-16 bg-gradient-to-t from-white/10 to-transparent" />
              <div className="w-12 h-12 border-2 border-white/30 bg-white/10 rounded-full flex items-center justify-center shadow-[0_0_15px_rgba(255,255,255,0.1)] backdrop-blur-md z-10 relative">
                <span className="font-mono text-[14px] text-white font-bold tracking-widest">x{t}</span>
                {/* Highlight dot */}
                <div className="absolute top-1 right-2 w-1.5 h-1.5 rounded-full bg-white/50" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
