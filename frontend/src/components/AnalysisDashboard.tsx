import { motion } from "framer-motion";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, Target, Zap } from "lucide-react";
import type { InferenceResult } from "../services/api";
import { PoseWireframe } from "./PoseWireframe";
import { CricketPitchMap } from "./CricketPitchMap";

interface AnalysisDashboardProps {
  result: InferenceResult | null;
  videoUrl: string | null;
}

export function AnalysisDashboard({ result, videoUrl }: AnalysisDashboardProps) {
  if (!result || !videoUrl) return null;

  return (
    <motion.section
      initial={{ opacity: 0, y: 60 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ type: "spring", stiffness: 80, damping: 20 }}
      className="w-full max-w-7xl mx-auto px-6 py-12"
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        
        {/* Video Player */}
        <motion.div 
          initial={{ opacity: 0, x: -30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ type: "spring", stiffness: 100, damping: 15, delay: 0.1 }}
          className="lg:col-span-2 glass rounded-[2rem] p-6 flex flex-col gap-6"
        >
          <div className="flex items-center justify-between px-2">
            <h3 className="text-2xl font-bold flex items-center gap-3">
              <Activity className="text-[#00f0ff] w-7 h-7" />
              Shot Playback
            </h3>
            <span className="text-xs font-mono bg-[#00f0ff]/10 text-[#00f0ff] px-4 py-2 rounded-full border border-[#00f0ff]/30 shadow-[0_0_15px_rgba(0,240,255,0.2)]">
              FRAME ANALYSIS ACTIVE
            </span>
          </div>
          <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden border border-white/10 shadow-2xl group">
            <video
              src={videoUrl}
              controls
              autoPlay
              loop
              className="w-full h-full object-cover"
            />
            
            {/* AI Pose Estimation Overlay */}
            <PoseWireframe />
            
            {/* Overlay Grid simulating AI scanning */}
            <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(to_right,#00f0ff08_1px,transparent_1px),linear-gradient(to_bottom,#00f0ff08_1px,transparent_1px)] bg-[size:40px_40px] mix-blend-overlay" />
          </div>
        </motion.div>

        {/* Real-time Inference Widget */}
        <motion.div 
          initial={{ opacity: 0, x: 30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ type: "spring", stiffness: 100, damping: 15, delay: 0.2 }}
          className="glass rounded-[2rem] p-8 flex flex-col items-center justify-center relative overflow-hidden group"
        >
          <div className="absolute -top-24 -right-24 w-64 h-64 bg-[#00ff66] opacity-10 blur-[60px] rounded-full pointer-events-none group-hover:opacity-20 group-hover:scale-110 transition-all duration-700" />
          
          <h3 className="text-2xl font-bold mb-10 flex items-center gap-3 w-full">
            <Target className="text-[#00ff66] w-7 h-7" />
            Prediction Map
          </h3>

          {/* Cricket Pitch Field Map */}
          <CricketPitchMap predictedClass={result.predictedClass} />

          <div className="w-full text-center p-6 rounded-2xl bg-black/40 border border-white/10 group-hover:border-[#00ff66]/30 transition-colors duration-500 shadow-inner flex flex-col gap-1 mt-auto">
            <span className="text-sm text-gray-500 block font-medium">Confidence: {result.confidence.toFixed(1)}%</span>
            <span className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-[#00f0ff] to-[#00ff66]">
              {result.predictedClass}
            </span>
          </div>
        </motion.div>

        {/* Temporal Analysis Graph */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ type: "spring", stiffness: 100, damping: 15, delay: 0.3 }}
          className="lg:col-span-3 glass rounded-[2rem] p-8 mt-6"
        >
          <div className="flex items-center justify-between mb-10">
            <h3 className="text-2xl font-bold flex items-center gap-3">
              <Zap className="text-[#00f0ff] w-7 h-7" />
              Temporal Analysis (Bi-GRU Sequence)
            </h3>
          </div>
          <div className="w-full h-[350px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={result.timeline} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f0ff" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#00f0ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="frame" 
                  stroke="#ffffff40" 
                  tick={{ fill: '#ffffff80', fontSize: 13, fontWeight: 500 }} 
                  tickFormatter={(val) => `F${val}`}
                  tickMargin={10}
                />
                <YAxis 
                  stroke="#ffffff40" 
                  tick={{ fill: '#ffffff80', fontSize: 13, fontWeight: 500 }}
                  domain={[0, 100]}
                  tickMargin={10}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(3, 7, 18, 0.95)', 
                    border: '1px solid rgba(0, 240, 255, 0.4)',
                    borderRadius: '12px',
                    boxShadow: '0 10px 30px rgba(0, 240, 255, 0.2)',
                    backdropFilter: 'blur(10px)'
                  }} 
                  itemStyle={{ color: '#00f0ff', fontWeight: 'bold' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="confidence" 
                  stroke="#00f0ff" 
                  strokeWidth={4}
                  fillOpacity={1} 
                  fill="url(#colorConfidence)" 
                  animationDuration={2500}
                  animationEasing="ease-out"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

      </div>
    </motion.section>
  );
}
