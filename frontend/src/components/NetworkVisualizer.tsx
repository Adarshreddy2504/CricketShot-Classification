import { motion } from "framer-motion";

export function NetworkVisualizer() {
  const stages = [
    { w: 90, h: 30 },
    { w: 75, h: 36 },
    { w: 55, h: 46 },
    { w: 35, h: 60 }
  ];

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-between pt-2 pb-2 flex-1">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(0,240,255,0.05)_0%,transparent_70%)] pointer-events-none" />

      {/* Input Frame */}
      <div className="flex flex-col items-center z-10 w-full mb-4">
        <div className="w-32 h-20 border border-dashed border-[#00f0ff]/50 bg-[#00f0ff]/10 flex items-center justify-center shadow-[0_0_20px_rgba(0,240,255,0.15)] relative overflow-hidden backdrop-blur-sm">
           <span className="font-mono text-[12px] text-gray-200 tracking-wider font-semibold z-10">224x224x3</span>
           <div className="absolute inset-0 bg-[linear-gradient(to_right,#00f0ff20_1px,transparent_1px),linear-gradient(to_bottom,#00f0ff20_1px,transparent_1px)] bg-[size:10px_10px]" />
        </div>
      </div>

      {/* 3D Slabs Stack */}
      <div className="flex flex-col items-center gap-3 relative w-full flex-1 justify-center my-2">
        {stages.map((stage, i) => (
          <div 
            key={i} 
            className="relative border border-[#00f0ff]/50 bg-gradient-to-b from-[#00f0ff]/20 to-[#00f0ff]/5 flex items-center justify-center backdrop-blur-md shadow-[0_8px_15px_rgba(0,240,255,0.15)]"
            style={{ width: `${stage.w}%`, height: `${stage.h}px`, borderRadius: '4px' }}
          >
            {/* Top highlight */}
            <div className="absolute top-0 left-0 right-0 h-px bg-white/40" />
            <span className="font-mono text-[10px] text-cyan-200/50 hidden md:block">MBConv Block {i+1}</span>
          </div>
        ))}
        {/* Animated Scanner */}
        <motion.div
          animate={{ top: ["0%", "100%"] }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="absolute left-0 right-0 h-16 bg-gradient-to-b from-transparent via-[#00f0ff]/40 to-transparent z-50 pointer-events-none blur-md"
        />
      </div>
      
      {/* Feature Vector */}
      <div className="flex flex-col items-center relative z-10 w-full mt-4">
         <div className="w-8 h-20 sm:h-28 bg-gradient-to-b from-[#00f0ff] to-[#00f0ff]/10 rounded shadow-[0_0_25px_rgba(0,240,255,0.7)] border border-[#00f0ff]" />
         <span className="mt-3 font-mono text-[12px] text-[#00f0ff] tracking-widest font-black uppercase">1280-D Embedding</span>
      </div>
    </div>
  );
}
