import { motion, type Variants } from "framer-motion";
import { Upload, ChevronRight, Play } from "lucide-react";

export function Hero() {
  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.2 },
    },
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: "spring", stiffness: 100, damping: 15 },
    },
  };

  return (
    <section className="relative min-h-[90vh] flex flex-col items-center justify-center overflow-hidden px-4 pt-12">
      {/* Dynamic Background Glows */}
      <motion.div
        animate={{ scale: [1, 1.05, 1], opacity: [0.1, 0.15, 0.1] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] bg-[#00f0ff] blur-[200px] rounded-full pointer-events-none"
      />
      <motion.div
        animate={{ scale: [1, 1.1, 1], opacity: [0.1, 0.15, 0.1] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        className="absolute top-1/4 right-1/4 w-[500px] h-[500px] bg-[#00ff66] blur-[150px] rounded-full pointer-events-none"
      />

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="z-10 text-center max-w-5xl mx-auto flex flex-col items-center"
      >
        <motion.div
          variants={itemVariants}
          className="inline-flex items-center gap-3 px-6 py-2.5 rounded-full glass mb-10 border-[#00f0ff]/40 text-[#00f0ff] font-semibold tracking-widest text-xs uppercase shadow-[0_0_20px_rgba(0,240,255,0.2)]"
        >
          <span className="w-2 h-2 rounded-full bg-[#00f0ff] shadow-[0_0_10px_#00f0ff] animate-pulse" />
          EfficientNet-B0 + Bi-GRU Engine
        </motion.div>

        <motion.h1
          variants={itemVariants}
          className="text-6xl md:text-8xl font-black tracking-tighter mb-8 leading-[1.1]"
        >
          Decode Every Shot <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#00f0ff] via-white to-[#00ff66] animate-[gradient_8s_ease_infinite] bg-[length:200%_auto]">
            with AI Precision
          </span>
        </motion.h1>

        <motion.p
          variants={itemVariants}
          className="text-xl md:text-2xl text-gray-400 mb-14 max-w-3xl leading-relaxed font-light"
        >
          State-of-the-art Deep Learning architecture for cricket shot classification.
          Upload a video and let our spatial-temporal models break down the technique frame-by-frame.
        </motion.p>

        <motion.div
          variants={itemVariants}
          className="flex flex-col sm:flex-row gap-6 w-full sm:w-auto"
        >
          <a
            href="#demo-zone"
            className="group relative px-10 py-5 bg-[#00f0ff] text-black font-extrabold text-lg rounded-2xl overflow-hidden transition-all hover:scale-105 active:scale-95 shadow-[0_0_30px_rgba(0,240,255,0.3)] hover:shadow-[0_0_50px_rgba(0,240,255,0.6)] flex items-center justify-center gap-3"
          >
            <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
            <Upload className="w-6 h-6" />
            Analyze Video
          </a>

          <button className="group px-10 py-5 glass-hover glass rounded-2xl font-bold text-lg text-white flex items-center justify-center gap-3">
            <Play className="w-5 h-5 text-[#00ff66]" />
            Watch Demo
            <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-white group-hover:translate-x-1 transition-all" />
          </button>
        </motion.div>
      </motion.div>
    </section>
  );
}
