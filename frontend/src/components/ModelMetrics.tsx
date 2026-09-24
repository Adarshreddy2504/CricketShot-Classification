import { motion, useMotionValue, useTransform, animate, useInView } from "framer-motion";
import { useRef, useEffect } from "react";

function AnimatedNumber({ 
  value, 
  suffix = "", 
  format = (v: number) => v.toFixed(0) 
}: { 
  value: number; 
  suffix?: string; 
  format?: (v: number) => string;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });
  const motionValue = useMotionValue(0);
  const displayValue = useTransform(motionValue, (v) => format(v) + suffix);

  useEffect(() => {
    if (isInView) {
      const controls = animate(motionValue, value, {
        duration: 2,
        ease: "easeOut",
      });
      return controls.stop;
    }
  }, [isInView, value, motionValue]);

  return <motion.span ref={ref}>{displayValue}</motion.span>;
}

export function ModelMetrics() {
  return (
    <section className="w-full max-w-7xl mx-auto py-24 px-6 relative z-10" id="models">
      {/* Hero Metric Card */}
      <div className="max-w-5xl mx-auto bg-[#0b0f19]/80 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-10 shadow-[0_0_40px_rgba(0,255,255,0.1)]">
        <h2 className="text-2xl font-bold text-white text-center mb-2">
          EfficientNet-B0 + Bi-GRU
        </h2>
        <p className="text-sm text-gray-400 text-center mb-10">
          State-of-the-Art Architecture Performance Metrics
        </p>

        {/* 4-Metric Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          
          {/* Metric 1: Accuracy */}
          <div className="flex flex-col items-center text-center">
            <div className="text-cyan-400 text-4xl font-bold mb-4 drop-shadow-[0_0_15px_rgba(34,211,238,0.5)]">
              <AnimatedNumber value={94.2} suffix="%" format={(v) => v.toFixed(1)} />
            </div>
            <div className="w-full max-w-[120px] mb-2">
               <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                 <motion.div 
                   initial={{ width: 0 }}
                   whileInView={{ width: "94.2%" }}
                   viewport={{ once: true, margin: "-50px" }}
                   transition={{ duration: 1.5, ease: "easeOut", delay: 0.2 }}
                   className="h-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.8)] rounded-full"
                 />
               </div>
            </div>
            <span className="text-gray-300 font-medium text-sm">Test Accuracy</span>
          </div>

          {/* Metric 2: F1-Score */}
          <div className="flex flex-col items-center text-center">
            <div className="text-emerald-400 text-4xl font-bold mb-4 drop-shadow-[0_0_15px_rgba(52,211,153,0.5)]">
              <AnimatedNumber value={0.93} format={(v) => v.toFixed(2)} />
            </div>
            <div className="w-full max-w-[120px] mb-2">
               <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                 <motion.div 
                   initial={{ width: 0 }}
                   whileInView={{ width: "93%" }}
                   viewport={{ once: true, margin: "-50px" }}
                   transition={{ duration: 1.5, ease: "easeOut", delay: 0.4 }}
                   className="h-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)] rounded-full"
                 />
               </div>
            </div>
            <span className="text-gray-300 font-medium text-sm">Macro F1-Score</span>
          </div>

          {/* Metric 3: Parameters */}
          <div className="flex flex-col items-center text-center">
            <div className="text-white text-4xl font-bold mb-4 drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]">
              <AnimatedNumber value={5.8} suffix="M" format={(v) => v.toFixed(1)} />
            </div>
            <div className="h-1.5 mb-2" /> {/* Spacer to align with progress bars */}
            <span className="text-gray-300 font-medium text-sm">Lightweight Parameters</span>
          </div>

          {/* Metric 4: Speed */}
          <div className="flex flex-col items-center text-center">
            <div className="text-purple-400 text-4xl font-bold mb-4 drop-shadow-[0_0_15px_rgba(192,132,252,0.5)]">
              <AnimatedNumber value={34} suffix=" FPS" />
            </div>
            <div className="h-1.5 mb-2" /> {/* Spacer */}
            <span className="text-gray-300 font-medium text-sm">Real-Time Inference</span>
          </div>

        </div>
      </div>
    </section>
  );
}
