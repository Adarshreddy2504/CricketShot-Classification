import { cricshot10Data } from "../data/cricshot10";
import { HolographicCard } from "./HolographicCard";
import { BookOpen } from "lucide-react";
import { motion } from "framer-motion";
import type { Variants } from "framer-motion";

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 40 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

export function ShotDictionary() {
  return (
    <section className="w-full py-24 relative overflow-hidden" id="shot-dictionary">
      <div className="max-w-7xl mx-auto px-6 mb-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="inline-flex items-center justify-center gap-3 mb-6"
        >
          <BookOpen className="w-8 h-8 text-[#00f0ff]" />
          <h2 className="text-4xl md:text-5xl font-black tracking-tight">
            The CricShot10 Dictionary
          </h2>
        </motion.div>
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="text-xl text-gray-400 font-light max-w-2xl mx-auto"
        >
          Explore the 10 distinct biomechanical classes our AI model is trained to recognize. 
          Hover over each shot to view its swing mechanics.
        </motion.p>
      </div>

      {/* Grid Container */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-100px" }}
        className="w-full max-w-[96vw] 2xl:max-w-[1600px] mx-auto grid grid-cols-2 lg:grid-cols-5 gap-4 xl:gap-6 p-4"
      >
        {cricshot10Data.map((shot) => (
          <motion.div key={shot.id} variants={itemVariants} className="w-full h-full flex justify-center">
            <HolographicCard shot={shot} />
          </motion.div>
        ))}
      </motion.div>
    </section>
  );
}
