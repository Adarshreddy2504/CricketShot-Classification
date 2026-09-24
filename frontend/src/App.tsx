import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Lenis from "@studio-freight/lenis";
import { Hero } from "./components/Hero";
import { UploadZone } from "./components/UploadZone";
import { AnalysisDashboard } from "./components/AnalysisDashboard";
import { ArchitectureExplainer } from "./components/ArchitectureExplainer";
import { ShotClasses } from "./components/ShotClasses";
import { ModelMetrics } from "./components/ModelMetrics";

import { CricketVideoBackground } from "./components/CricketVideoBackground";
import { type InferenceResult } from "./services/api";

function App() {
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: "vertical",
      gestureOrientation: "vertical",
      smoothWheel: true,
      touchMultiplier: 2,
    });

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    return () => {
      lenis.destroy();
    };
  }, []);
  const handleUploadComplete = (file: File, inferenceData: InferenceResult) => {
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setResult(inferenceData);
    setTimeout(() => {
      document.getElementById('demo-zone')?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  return (
    <div className="min-h-screen text-white selection:bg-[#00f0ff] selection:text-black font-sans relative">
      <CricketVideoBackground />

      <motion.nav 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="fixed top-6 inset-x-0 mx-auto w-[90%] max-w-4xl z-50 flex items-center justify-between px-5 py-3 rounded-full bg-[#0b0f19]/60 backdrop-blur-2xl border border-white/10 shadow-[0_8px_32px_rgba(0,255,255,0.15)]"
      >
        {/* Left: AI Status Logo */}
        <div className="flex items-center gap-3 z-10">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00f0ff] to-[#00ff66] flex items-center justify-center font-black text-black shadow-[0_0_15px_rgba(0,240,255,0.4)]">
            C
          </div>
          <div className="flex items-center">
            <span className="font-bold text-xl tracking-wide text-white">CricShot AI</span>
            <div className="hidden sm:flex items-center ml-3 text-[10px] uppercase font-mono text-emerald-400 bg-emerald-900/30 px-2 py-1 rounded-full border border-emerald-500/30 gap-1.5">
              <motion.div 
                className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
              />
              Bi-GRU Online
            </div>
          </div>
        </div>

        {/* Center: Navigation Links */}
        <div className="hidden md:flex items-center gap-6 z-0">
          <a href="#models" className="text-sm font-semibold text-gray-300 transition-colors duration-300 hover:text-cyan-400 hover:drop-shadow-[0_0_8px_rgba(0,255,255,0.8)]">Models</a>
          <a href="https://github.com/your-username/cricshot" target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-gray-300 transition-colors duration-300 hover:text-cyan-400 hover:drop-shadow-[0_0_8px_rgba(0,255,255,0.8)]">GitHub</a>
        </div>

        {/* Right: CTA Button */}
        <div className="z-10">
          <a 
            href="#demo-zone"
            className="inline-block text-center bg-gradient-to-r from-cyan-500 to-emerald-500 text-white text-sm font-bold px-6 py-2.5 rounded-full hover:scale-105 active:scale-95 transition-transform shadow-[0_0_15px_rgba(0,255,128,0.4)]"
          >
            Launch Demo
          </a>
        </div>
      </motion.nav>

      <main className="pt-24 flex flex-col gap-24 pb-24 relative z-10">
        <Hero />
        
        <div className="flex flex-col gap-24">
          <div id="demo-zone">
            <UploadZone 
              onUploadComplete={handleUploadComplete} 
            />
            
            {videoUrl && (
              <AnalysisDashboard result={result} videoUrl={videoUrl} />
            )}
          </div>

          <div>
            <ShotClasses />
          </div>

          <div id="architecture">
            <ArchitectureExplainer />
          </div>

          <ModelMetrics />
        </div>
      </main>

      <footer className="border-t border-white/10 py-12 text-center text-gray-500 text-sm glass bg-[#030712]/50 backdrop-blur-xl relative z-10">
        <p className="font-medium">© 2026 CricShot AI. Precision Deep Learning for Cricket.</p>
      </footer>
    </div>
  );
}

export default App;
