import { motion, useScroll, useTransform } from "framer-motion";

export function CricketVideoBackground() {
  const { scrollYProgress } = useScroll();
  
  // Parallax effects linked to scroll
  // Parallax effect linked to scroll (scale only)
  const scale = useTransform(scrollYProgress, [0, 1], [1, 0.95]);

  return (
    <div className="fixed top-0 left-0 w-full h-full -z-50 pointer-events-none overflow-hidden bg-black">
      
      {/* Background Video with Scroll-Linked Parallax */}
      <motion.video 
        autoPlay 
        loop 
        muted 
        playsInline 
        className="absolute top-0 left-0 w-full h-full object-cover"
        src="/cricket-bg.mp4"
        style={{ scale }}
      />

      <div className="absolute inset-0 bg-black/60 z-[-45]" />
      <div className="absolute inset-0 z-[-44] opacity-30 bg-[linear-gradient(to_right,#ffffff1a_1px,transparent_1px),linear-gradient(to_bottom,#ffffff1a_1px,transparent_1px)] bg-[size:30px_30px] pointer-events-none" />

      {/* 2. Computer Vision Scanning Laser */}
      <motion.div 
        className="absolute top-0 left-0 h-[1px] w-full bg-cyan-400/50 shadow-[0_0_15px_#00f0ff]"
        animate={{ y: ["0vh", "100vh"] }}
        transition={{ 
          duration: 3.5, 
          repeat: Infinity, 
          ease: "linear" 
        }}
      />
    </div>
  );
}
