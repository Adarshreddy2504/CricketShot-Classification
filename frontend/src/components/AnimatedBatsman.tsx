import { useEffect, useState, useRef } from "react";
import { Lottie } from "lottie-react";
import type { LottieHandle } from "lottie-react";
import { motion } from "framer-motion";
import type { ShotData } from "../data/cricshot10";

// A placeholder URL for a Cricket Batsman Lottie JSON.
// Replace this with a real Lottie JSON URL from IconScout or LottieFiles.
const LOTTIE_BATSMAN_URL = "https://assets9.lottiefiles.com/packages/lf20_cricket_placeholder.json";

interface AnimatedBatsmanProps {
  shot: ShotData;
}

export function AnimatedBatsman({ shot }: AnimatedBatsmanProps) {
  const [animationData, setAnimationData] = useState<any>(null);
  const lottieRef = useRef<LottieHandle>(null);

  // Fetch the Lottie JSON
  useEffect(() => {
    const fetchLottie = async () => {
      try {
        const response = await fetch(LOTTIE_BATSMAN_URL);
        if (response.ok) {
          const data = await response.json();
          setAnimationData(data);
        } else {
          throw new Error("Placeholder URL not found");
        }
      } catch (err) {
        // Fallback minimal valid Lottie JSON to prevent crash if placeholder is 404
        setAnimationData({
          v: "5.5.2",
          fr: 30,
          ip: 0,
          op: 60,
          w: 100,
          h: 100,
          nm: "Placeholder",
          ddd: 0,
          assets: [],
          layers: []
        });
        console.warn("Could not load Lottie animation. Replace LOTTIE_BATSMAN_URL with a valid JSON url.");
      }
    };
    
    fetchLottie();
  }, []);

  return (
    <div className="relative z-10 w-36 h-36 flex items-center justify-center my-2 pointer-events-none">
      
      {/* Silhouette Glow - Positioned behind */}
      <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full z-0 overflow-visible">
        <motion.circle 
          cx="50" cy="50" r="30" 
          fill={shot.color} 
          className="blur-2xl opacity-20"
          animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.3, 0.1] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
      </svg>

      {/* Lottie Animation Layer */}
      {/* 
        We use CSS filters to force the Lottie to a white/monochromatic silhouette, 
        ensuring it maintains the sleek, premium aesthetic regardless of the source asset colors.
      */}
      <div 
        className="absolute inset-0 w-full h-full flex items-center justify-center transition-opacity duration-500 z-10"
        style={{ 
          opacity: 1,
          filter: "brightness(0) invert(1) drop-shadow(0 0 4px rgba(255,255,255,0.3))" 
        }}
      >
        {animationData && (
          <Lottie 
            lottieRef={lottieRef}
            src={animationData} 
            loop={true} 
            autoplay={true}
            className="w-full h-full"
          />
        )}
      </div>

      {/* Neon Trajectory Overlaid on Top */}
      <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full overflow-visible z-20">
        <g style={{ filter: `drop-shadow(0 0 8px ${shot.color})` }}>
          <motion.path
            d={shot.svgPath}
            stroke={shot.color}
            strokeWidth="3.5"
            fill="transparent"
            strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ 
              pathLength: [0, 1],
              opacity: [0, 1, 0] 
            }}
            transition={{ 
              duration: 1.5, 
              repeat: Infinity, 
              ease: "easeInOut"
            }}
          />
        </g>
      </svg>

    </div>
  );
}
