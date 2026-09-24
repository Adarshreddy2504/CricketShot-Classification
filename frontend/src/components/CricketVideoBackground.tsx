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
        muted 
        playsInline 
        className="absolute top-0 left-0 w-full h-full object-cover"
        src="/cricket-bg.mp4"
        style={{ scale }}
      />

      <div className="absolute inset-0 bg-black/60 z-[-45]" />
    </div>
  );
}
