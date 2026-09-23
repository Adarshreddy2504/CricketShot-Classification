import { useState, useRef, type MouseEvent } from "react";
import { motion } from "framer-motion";
import { Cpu, Network, Frame } from "lucide-react";
import { cn } from "../lib/utils";
import { NetworkVisualizer } from "./NetworkVisualizer";
import { BiGruVisualizer } from "./BiGruVisualizer";

// Spotlight Card component
function BentoCard({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const divRef = useRef<HTMLDivElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!divRef.current || isFocused) return;

    const div = divRef.current;
    const rect = div.getBoundingClientRect();

    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleFocus = () => {
    setIsFocused(true);
    setOpacity(1);
  };

  const handleBlur = () => {
    setIsFocused(false);
    setOpacity(0);
  };

  const handleMouseEnter = () => {
    setOpacity(1);
  };

  const handleMouseLeave = () => {
    setOpacity(0);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ type: "spring", stiffness: 100, damping: 20 }}
      ref={divRef}
      onMouseMove={handleMouseMove}
      onFocus={handleFocus}
      onBlur={handleBlur}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={cn(
        "relative rounded-3xl overflow-hidden glass bg-[#030712] transition-colors duration-500 border-t border-white/20 shadow-2xl shadow-cyan-900/20",
        className
      )}
    >
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition duration-300"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(0,240,255,.15), transparent 40%)`,
        }}
      />
      {/* Content wrapper */}
      <div className="relative z-10 p-8 h-full flex flex-col">{children}</div>
    </motion.div>
  );
}

export function ArchitectureExplainer() {
  return (
    <section className="w-full max-w-7xl mx-auto py-24 px-6 relative">
      <motion.div 
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
        className="text-center mb-20"
      >
        <h2 className="text-4xl md:text-6xl font-black mb-6 tracking-tight">Under the Hood</h2>
        <p className="text-xl text-gray-400 max-w-3xl mx-auto font-light leading-relaxed">
          Unlike baseline models that only look at static images, our architecture 
          understands motion through time.
        </p>
      </motion.div>

      {/* Bento Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
        
        {/* Card 1: Top Left (Raw Frame Sampling) */}
        <BentoCard className="group flex flex-col">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-10 h-10 rounded-xl glass flex items-center justify-center bg-white/5 shadow-inner">
              <Frame className="w-5 h-5 text-[#00f0ff]" />
            </div>
            <h3 className="text-xl font-bold tracking-tight">Raw Frame Sequence</h3>
          </div>
          <p className="text-gray-400 text-sm mb-6 max-w-sm font-light">
            Uniform temporal sampling: 16 equidistant frames extracted at 30 FPS.
          </p>

          <div className="mt-auto relative w-full flex flex-col gap-4 flex-1 justify-end">
            {/* High-tech preview canvas */}
            <div className="w-full h-44 bg-[#020408] rounded-xl border border-[#00f0ff]/20 overflow-hidden relative shadow-[inset_0_0_20px_rgba(0,240,255,0.05)]">
              {/* Pitch grid */}
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#00f0ff1a_1px,transparent_1px),linear-gradient(to_bottom,#00f0ff1a_1px,transparent_1px)] bg-[size:20px_20px]" />
              
              {/* Bounding box target */}
              <motion.div 
                className="absolute border border-[#00ff66] bg-[#00ff66]/10 flex flex-col justify-end shadow-[0_0_15px_rgba(0,255,102,0.2)]"
                animate={{ 
                  left: ["35%", "40%", "35%"],
                  top: ["20%", "25%", "20%"],
                  width: ["30%", "28%", "30%"],
                  height: ["60%", "58%", "60%"]
                }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              >
                 {/* Reticle corners */}
                 <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-[#00ff66]" />
                 <div className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2 border-[#00ff66]" />
                 <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2 border-[#00ff66]" />
                 <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-[#00ff66]" />
                 
                 {/* Coordinate label */}
                 <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap bg-[#020408]/80 px-2 py-0.5 border border-[#00ff66]/40 rounded text-center backdrop-blur-sm">
                    <span className="font-mono text-[10px] text-[#00ff66] tracking-wider">BBox: [x:142, y:88, w:110, h:210]</span>
                 </div>
              </motion.div>
            </div>

            {/* 5-segment filmstrip with waveforms */}
            <div className="relative w-[calc(100%+4rem)] -mx-8 h-20 flex items-end justify-between border-y border-[#00f0ff]/30 bg-[#00f0ff]/5 backdrop-blur-md overflow-hidden">
               {/* Film sprockets */}
               <div className="absolute top-1 left-0 right-0 flex justify-between px-4 gap-1 w-full">
                  {[...Array(24)].map((_, j) => <div key={j} className="flex-1 max-w-[4px] h-1.5 bg-black/90 rounded-[1px] shadow-inner" />)}
               </div>
               <div className="absolute bottom-1 left-0 right-0 flex justify-between px-4 gap-1 w-full">
                  {[...Array(24)].map((_, j) => <div key={j} className="flex-1 max-w-[4px] h-1.5 bg-black/90 rounded-[1px] shadow-inner" />)}
               </div>

               {/* Waveform content */}
               <div className="w-full h-[60%] flex justify-around items-end z-10 px-8 pb-3 gap-2">
                 {[...Array(5)].map((_, i) => (
                   <div key={i} className="flex-1 h-full max-w-[18%] bg-black/40 border border-[#00f0ff]/20 flex items-end justify-around px-1.5 pb-1 gap-[2px] rounded-sm">
                     {[...Array(5)].map((_, j) => (
                        <motion.div 
                          key={j} 
                          className="w-full bg-[#00f0ff]/60 rounded-t-[1px]"
                          animate={{ height: [`${Math.random() * 40 + 20}%`, `${Math.random() * 60 + 40}%`, `${Math.random() * 40 + 20}%`] }}
                          transition={{ duration: 1.5 + Math.random(), repeat: Infinity, ease: "easeInOut" }}
                        />
                     ))}
                   </div>
                 ))}
               </div>

               {/* Playhead Scanner */}
               <motion.div
                  className="absolute top-0 bottom-0 w-[2px] bg-[#00ffcc] z-20 shadow-[0_0_15px_3px_#00ffcc]"
                  animate={{ left: ["-10%", "110%"] }}
                  transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                >
                  <div className="absolute top-0 bottom-0 right-full w-24 bg-gradient-to-l from-[#00ffcc]/40 to-transparent pointer-events-none" />
                </motion.div>
            </div>
          </div>
        </BentoCard>

        {/* Card 2: Top Right (EfficientNet Scanner) */}
        <BentoCard className="group flex flex-col">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-10 h-10 rounded-xl glass flex items-center justify-center bg-white/5 shadow-inner">
              <Network className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-xl font-bold tracking-tight">EfficientNet-B0</h3>
          </div>
          <p className="text-gray-400 text-sm mb-6 font-light">
            Compound-scaled CNN extracting 1280-D spatial embeddings per frame.
          </p>
          <div className="flex-1 flex flex-col justify-end">
            <NetworkVisualizer />
          </div>
        </BentoCard>

        {/* Card 3: Full Width Bottom (GRU Temporal Sequence) */}
        <BentoCard className="lg:col-span-2 group">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-10 h-10 rounded-xl glass flex items-center justify-center bg-[#00ff66]/10 shadow-inner">
              <Cpu className="w-5 h-5 text-[#00ff66]" />
            </div>
            <h3 className="text-xl font-bold tracking-tight">Bidirectional GRU (Bi-GRU)</h3>
          </div>
          <p className="text-gray-400 text-sm mb-8 max-w-2xl font-light">
            Processes spatial features across time in both directions. By analyzing the video frames forwards and backwards simultaneously, the Bi-GRU captures the complete context of the stroke, from backlift to follow-through.
          </p>
          
          <BiGruVisualizer />
        </BentoCard>

      </div>
    </section>
  );
}
