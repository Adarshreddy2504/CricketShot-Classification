import { motion } from "framer-motion";

export function PoseWireframe() {
  // SVG paths for a swinging motion
  const pathStart = "M 50 10 L 40 30 L 30 50 L 50 70 L 60 90";
  const pathEnd = "M 50 10 L 60 30 L 70 50 L 50 70 L 40 90";
  
  const batStart = "M 30 50 L 10 70";
  const batEnd = "M 70 50 L 90 30";

  return (
    <div className="absolute inset-0 z-10 pointer-events-none flex items-center justify-center bg-black/40 backdrop-blur-[2px]">
      <svg 
        viewBox="0 0 100 100" 
        className="w-3/4 h-3/4 drop-shadow-[0_0_15px_rgba(0,255,102,0.8)]"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Nodes */}
        <motion.circle cx="50" cy="10" r="3" fill="#00ff66" />
        <motion.circle 
          cx="40" cy="30" r="2" fill="#00ff66" 
          animate={{ cx: [40, 60, 40] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle 
          cx="30" cy="50" r="2" fill="#00ff66"
          animate={{ cx: [30, 70, 30], cy: [50, 45, 50] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle 
          cx="50" cy="70" r="2" fill="#00ff66" 
          animate={{ cx: [50, 50, 50] }}
        />
        <motion.circle 
          cx="60" cy="90" r="2" fill="#00ff66" 
          animate={{ cx: [60, 40, 60] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        />
        
        {/* Skeleton Lines */}
        <motion.path
          d={pathStart}
          stroke="#00ff66"
          strokeWidth="1.5"
          fill="transparent"
          strokeLinecap="round"
          strokeLinejoin="round"
          animate={{ d: [pathStart, pathEnd, pathStart] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        />
        
        {/* Bat Line */}
        <motion.path
          d={batStart}
          stroke="#00f0ff"
          strokeWidth="3"
          fill="transparent"
          strokeLinecap="round"
          animate={{ d: [batStart, batEnd, batStart] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="drop-shadow-[0_0_15px_rgba(0,240,255,0.8)]"
        />
        
        {/* Bat Node */}
        <motion.circle 
          cx="10" cy="70" r="3" fill="#00f0ff"
          animate={{ cx: [10, 90, 10], cy: [70, 30, 70] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="drop-shadow-[0_0_15px_rgba(0,240,255,0.8)]"
        />
      </svg>
      
      {/* Scanning laser line */}
      <motion.div 
        className="absolute left-0 right-0 h-0.5 bg-[#00ff66] shadow-[0_0_15px_rgba(0,255,102,1)]"
        animate={{ top: ["10%", "90%", "10%"] }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}
