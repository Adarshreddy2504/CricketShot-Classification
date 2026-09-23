import { motion } from "framer-motion";

interface CricketPitchMapProps {
  predictedClass: string;
}

export function CricketPitchMap({ predictedClass }: CricketPitchMapProps) {
  // Map shots to general field zones
  const getHighlightZone = () => {
    const shot = predictedClass.toLowerCase();
    if (shot.includes("cover") || shot.includes("drive")) return "off-side";
    if (shot.includes("pull") || shot.includes("hook") || shot.includes("leg")) return "leg-side";
    if (shot.includes("straight") || shot.includes("drive")) return "straight";
    if (shot.includes("cut") || shot.includes("square")) return "point";
    return "straight";
  };

  const zone = getHighlightZone();

  return (
    <div className="relative w-64 h-64 mx-auto mb-8 rounded-full border border-white/10 bg-[#030712] overflow-hidden shadow-inner flex items-center justify-center">
      
      {/* Field Grass Gradient */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#00ff66]/10 via-transparent to-transparent opacity-50" />
      
      {/* 30 Yard Circle */}
      <div className="absolute w-40 h-40 rounded-full border border-white/20 border-dashed" />
      
      {/* Pitch */}
      <div className="absolute w-8 h-20 bg-[#f5d5a6]/20 border border-[#f5d5a6]/40 rounded-sm flex flex-col justify-between py-1 items-center">
        {/* Wickets */}
        <div className="w-4 h-1 border-b border-white/50" />
        <div className="w-4 h-1 border-t border-white/50" />
      </div>

      {/* Dynamic Highlight Zones */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: zone === "off-side" ? 1 : 0 }}
        transition={{ duration: 1 }}
        className="absolute top-1/4 left-0 w-1/2 h-1/2 bg-[radial-gradient(circle,_rgba(0,240,255,0.4)_0%,_transparent_70%)] blur-md"
      />
      
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: zone === "leg-side" ? 1 : 0 }}
        transition={{ duration: 1 }}
        className="absolute top-1/4 right-0 w-1/2 h-1/2 bg-[radial-gradient(circle,_rgba(0,240,255,0.4)_0%,_transparent_70%)] blur-md"
      />
      
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: zone === "straight" ? 1 : 0 }}
        transition={{ duration: 1 }}
        className="absolute bottom-0 left-1/4 w-1/2 h-1/2 bg-[radial-gradient(circle,_rgba(0,240,255,0.4)_0%,_transparent_70%)] blur-md"
      />

      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: zone === "point" ? 1 : 0 }}
        transition={{ duration: 1 }}
        className="absolute top-1/2 left-0 w-1/2 h-1/2 -translate-y-1/2 bg-[radial-gradient(circle,_rgba(0,240,255,0.4)_0%,_transparent_70%)] blur-md"
      />

      {/* Center Radar Sweep */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
        className="absolute inset-0 rounded-full"
        style={{
          background: "conic-gradient(from 0deg, transparent 70%, rgba(0, 255, 102, 0.2) 100%)",
        }}
      />
    </div>
  );
}
