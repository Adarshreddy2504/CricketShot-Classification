import { useState, useRef } from "react";
import type { MouseEvent } from "react";
import { motion } from "framer-motion";
import type { ShotData } from "../data/cricshot10";
import { cn } from "../lib/utils";
import { AnimatedBatsman } from "./AnimatedBatsman";

interface HolographicCardProps {
  shot: ShotData;
}

export function HolographicCard({ shot }: HolographicCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [rotateX, setRotateX] = useState(0);
  const [rotateY, setRotateY] = useState(0);
  const [mouseX, setMouseX] = useState(0);
  const [mouseY, setMouseY] = useState(0);
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;

    const card = cardRef.current;
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMouseX(x);
    setMouseY(y);

    // Calculate rotation (-15 to +15 degrees)
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const rotateXValue = ((y - centerY) / centerY) * -15;
    const rotateYValue = ((x - centerX) / centerX) * 15;

    setRotateX(rotateXValue);
    setRotateY(rotateYValue);
  };

  const handleMouseLeave = () => {
    setRotateX(0);
    setRotateY(0);
    setIsHovered(false);
  };

  return (
    <motion.div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={handleMouseLeave}
      animate={{
        rotateX: isHovered ? rotateX : 0,
        rotateY: isHovered ? rotateY : 0,
        scale: isHovered ? 1.05 : 1,
      }}
      transition={{ 
        type: "spring", 
        stiffness: isHovered ? 400 : 100, 
        damping: isHovered ? 30 : 20 
      }}
      className={cn(
        "relative w-full min-h-[420px] rounded-3xl glass p-4 flex flex-col justify-between overflow-hidden flex-shrink-0 cursor-pointer transform-gpu transition-shadow duration-500",
        isHovered ? "shadow-2xl" : "shadow-lg"
      )}
      style={{
        boxShadow: isHovered ? `0 20px 50px -10px ${shot.color}40, inset 0 0 20px ${shot.color}20` : undefined,
        borderColor: isHovered ? `${shot.color}80` : undefined,
      }}
    >
      {/* Background glow based on active state */}
      <div 
        className="absolute inset-0 opacity-20 transition-opacity duration-500"
        style={{
          background: `radial-gradient(circle at center, ${shot.color} 0%, transparent 70%)`
        }}
      />

      {/* Glare Effect */}
      {isHovered && (
        <div 
          className="absolute inset-0 pointer-events-none z-20 rounded-3xl transition-opacity duration-300"
          style={{
            background: `radial-gradient(circle at ${mouseX}px ${mouseY}px, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 60%)`,
            mixBlendMode: "overlay"
          }}
        />
      )}

      {/* Header */}
      <div className="relative z-10 w-full text-center mt-4">
        <h3 
          className="text-base xl:text-lg font-bold uppercase tracking-wider mb-2"
          style={{ 
            color: isHovered ? shot.color : '#ffffff',
            textShadow: isHovered ? `0 0 15px ${shot.color}80` : 'none'
          }}
        >
          {shot.name}
        </h3>
        <p className="text-xs text-gray-400 line-clamp-3 px-2">
          {shot.description}
        </p>
      </div>

      {/* Swing Trace SVG */}
      <AnimatedBatsman shot={shot} />

      {/* Mini Pitch Map */}
      <div className="relative z-10 w-full flex items-center justify-between border-t border-white/10 pt-4 mt-auto">
        <span className="text-xs font-mono text-gray-400 uppercase tracking-widest">
          Zone
        </span>
        <div className="relative w-16 h-16 rounded-full border border-white/20 bg-black/50 flex items-center justify-center overflow-hidden">
          {/* Mini Pitch */}
          <div className="absolute w-2 h-6 bg-white/10 border border-white/20 rounded-[1px]" />
          {/* Glowing Zone Indicator */}
          <motion.div
            className="absolute w-6 h-6 rounded-full blur-md"
            style={{ backgroundColor: shot.color }}
            animate={{ opacity: isHovered ? [0.4, 1, 0.4] : 0.2 }}
            transition={{ duration: isHovered ? 0.4 : 2, repeat: Infinity }}
            initial={false}
            {...(shot.zone === "off-side" && { style: { left: "-4px", top: "25%", backgroundColor: shot.color } })}
            {...(shot.zone === "leg-side" && { style: { right: "-4px", top: "25%", backgroundColor: shot.color } })}
            {...(shot.zone === "straight" && { style: { bottom: "-4px", left: "25%", backgroundColor: shot.color } })}
            {...(shot.zone === "point" && { style: { left: "-4px", top: "50%", transform: "translateY(-50%)", backgroundColor: shot.color } })}
          />
        </div>
      </div>
    </motion.div>
  );
}
