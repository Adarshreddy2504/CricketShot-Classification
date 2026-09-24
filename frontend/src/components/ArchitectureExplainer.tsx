import { useState, useRef, type MouseEvent } from "react";
import { motion } from "framer-motion";
import { Cpu, Network, Frame, Play, ImageIcon, ArrowRight, ArrowLeft } from "lucide-react";
import { cn } from "../lib/utils";

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
            <h3 className="text-xl font-bold text-white">Raw Frame Sequence</h3>
          </div>
          <p className="text-sm text-gray-300 mb-6 max-w-sm">
            Uniform temporal sampling: 16 equidistant frames extracted at 30 FPS.
          </p>

          <div className="mt-auto relative w-full flex flex-col gap-4 flex-1 justify-center py-6">
            <div className="flex justify-center gap-4 w-full">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="flex-1 aspect-[4/3] max-w-[120px] rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                  <Play className="w-8 h-8 text-[#00f0ff] opacity-80" fill="currentColor" />
                </div>
              ))}
            </div>
            <p className="font-bold text-sm text-center text-white mt-2">30 Frames Sampled Uniformly</p>
          </div>
        </BentoCard>

        {/* Card 2: Top Right (EfficientNet Scanner) */}
        <BentoCard className="group flex flex-col">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-10 h-10 rounded-xl glass flex items-center justify-center bg-white/5 shadow-inner">
              <Network className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white">EfficientNet-B0</h3>
          </div>
          <p className="text-sm text-gray-300 mb-6">
            Compound-scaled CNN extracting 1280-D spatial embeddings per frame.
          </p>
          <div className="flex-1 flex items-center justify-center gap-6">
            <div className="flex flex-col items-center gap-2">
              <div className="w-16 h-16 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                <ImageIcon className="w-8 h-8 text-white" />
              </div>
              <span className="text-xs font-bold text-white uppercase">Image</span>
            </div>
            <ArrowRight className="w-6 h-6 text-gray-500" />
            <div className="flex flex-col items-center gap-2">
              <div className="w-16 h-16 flex items-center justify-center">
                <div className="w-0 h-0 border-l-[24px] border-l-transparent border-t-[48px] border-t-[#00f0ff] border-r-[24px] border-r-transparent filter drop-shadow-[0_0_10px_rgba(0,240,255,0.4)]" />
              </div>
              <span className="text-xs font-bold text-white uppercase">CNN</span>
            </div>
            <ArrowRight className="w-6 h-6 text-gray-500" />
            <div className="flex flex-col items-center gap-2">
              <div className="w-24 h-6 rounded-full bg-white/5 border border-white/20 flex items-center justify-center px-2">
                 <div className="w-full flex justify-between gap-1">
                   {[...Array(5)].map((_, i) => <div key={i} className="flex-1 h-2 bg-white rounded-full" />)}
                 </div>
              </div>
              <span className="text-xs font-bold text-white uppercase">1280-D Vector</span>
            </div>
          </div>
        </BentoCard>

        {/* Card 3: Full Width Bottom (GRU Temporal Sequence) */}
        <BentoCard className="lg:col-span-2 group">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-10 h-10 rounded-xl glass flex items-center justify-center bg-[#00ff66]/10 shadow-inner">
              <Cpu className="w-5 h-5 text-[#00ff66]" />
            </div>
            <h3 className="text-xl font-bold text-white">Bidirectional GRU (Bi-GRU)</h3>
          </div>
          <p className="text-sm text-gray-300 mb-8 max-w-2xl">
            Processes spatial features across time in both directions. By analyzing the video frames forwards and backwards simultaneously, the Bi-GRU captures the complete context of the stroke, from backlift to follow-through.
          </p>
          
          <div className="mt-4 relative w-full max-w-3xl mx-auto py-16 flex flex-col items-center justify-center">
            <span className="absolute top-0 text-sm font-bold text-white uppercase tracking-widest">Bidirectional Temporal Processing</span>
            <div className="relative w-full flex items-center justify-between mt-6">
              <div className="absolute left-0 right-0 h-1 bg-white/20 top-1/2 -translate-y-1/2 rounded-full" />
              
              <div className="absolute left-10 right-10 -top-10 flex justify-between px-4">
                {[...Array(4)].map((_, i) => (
                  <ArrowRight key={`r-${i}`} className="w-6 h-6 text-[#00f0ff]" />
                ))}
              </div>

              <div className="absolute left-10 right-10 -bottom-10 flex justify-between px-4">
                {[...Array(4)].map((_, i) => (
                  <ArrowLeft key={`l-${i}`} className="w-6 h-6 text-[#00ff66]" />
                ))}
              </div>

              {[...Array(5)].map((_, i) => (
                <div key={`node-${i}`} className="w-12 h-12 rounded-full bg-[#030712] border-4 border-white z-10 flex items-center justify-center shadow-[0_0_15px_rgba(255,255,255,0.1)]">
                  <div className="w-3 h-3 rounded-full bg-white" />
                </div>
              ))}
            </div>
          </div>
        </BentoCard>

      </div>
    </section>
  );
}
