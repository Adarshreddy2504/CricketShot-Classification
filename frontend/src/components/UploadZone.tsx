import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileVideo, X, Loader2 } from "lucide-react";
import { cn } from "../lib/utils";

interface UploadZoneProps {
  onUpload: (file: File) => void;
  isLoading: boolean;
  onHoverStart?: () => void;
  onHoverEnd?: () => void;
}

export function UploadZone({ onUpload, isLoading, onHoverStart, onHoverEnd }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0 && files[0].type.startsWith("video/")) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleProcess = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <motion.section 
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ type: "spring", stiffness: 100, damping: 15 }}
      className="w-full max-w-5xl mx-auto py-12 px-6 relative" 
      id="upload-zone"
    >
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">Analyze Your Shot</h2>
        <p className="text-xl text-gray-400">Upload a video to see our AI in action.</p>
      </div>

      <div
        onMouseEnter={onHoverStart}
        onMouseLeave={onHoverEnd}
        className="relative"
      >
        {/* Rotating Dashed Border */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
          className="absolute inset-[-4px] rounded-[2rem] border-2 border-dashed border-[#00f0ff]/30 opacity-50 pointer-events-none"
        />

        <div
          className={cn(
            "relative rounded-[2rem] transition-all duration-500 overflow-hidden glass p-12 flex flex-col items-center justify-center min-h-[450px] z-10 bg-[#030712]/80",
            isDragging ? "border-[#00f0ff] bg-[#00f0ff]/10 scale-[1.02] shadow-[0_0_50px_rgba(0,240,255,0.2)]" : "border border-white/10 hover:border-white/30 hover:shadow-[0_0_40px_rgba(255,255,255,0.05)]",
            isLoading && "border-[#00ff66] bg-[#00ff66]/5"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <AnimatePresence mode="wait">
            {!selectedFile ? (
              <motion.div
                key="upload-prompt"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
                className="flex flex-col items-center text-center w-full"
              >
                <div className="w-24 h-24 rounded-2xl glass flex items-center justify-center mb-8 text-[#00f0ff] shadow-[0_0_30px_rgba(0,240,255,0.2)] group-hover:shadow-[0_0_50px_rgba(0,240,255,0.4)] transition-shadow duration-500">
                  <UploadCloud className="w-12 h-12" />
                </div>
                <h3 className="text-3xl font-bold mb-3">Drag & Drop Video</h3>
                <p className="text-lg text-gray-400 mb-10 max-w-md">
                  Supported formats: MP4, MOV, AVI. Maximum file size: 50MB.
                </p>
                
                <label className="cursor-pointer relative group">
                  <input
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileSelect}
                  />
                  <div className="px-10 py-4 rounded-xl bg-white/10 text-white font-bold text-lg group-hover:bg-white/20 transition-all duration-300 border border-white/10 group-hover:border-[#00f0ff]/50 hover:shadow-[0_0_20px_rgba(0,240,255,0.3)]">
                    Browse Files
                  </div>
                </label>
              </motion.div>
            ) : (
              <motion.div
                key="file-selected"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
                className="flex flex-col items-center w-full max-w-lg"
              >
                <div className="w-full glass rounded-2xl p-8 relative">
                  {!isLoading && (
                    <button
                      onClick={() => setSelectedFile(null)}
                      className="absolute top-6 right-6 text-gray-400 hover:text-white transition-colors bg-white/5 hover:bg-white/20 rounded-full p-2"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  )}
                  
                  <div className="flex items-center gap-6 mb-8">
                    <div className="p-5 rounded-xl bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/20">
                      <FileVideo className="w-10 h-10" />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <h4 className="font-bold text-xl truncate text-white" title={selectedFile.name}>
                        {selectedFile.name}
                      </h4>
                      <p className="text-gray-400 font-medium mt-1">
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>

                  {isLoading ? (
                    <div className="space-y-6">
                      <div className="flex items-center justify-between text-base font-bold text-[#00ff66]">
                        <span className="flex items-center gap-3">
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Extracting Spatial-Temporal Features...
                        </span>
                        <span className="animate-pulse opacity-80">Extracting frames</span>
                      </div>
                      <div className="w-full h-3 bg-white/10 rounded-full overflow-hidden shadow-inner">
                        <motion.div
                          className="h-full bg-gradient-to-r from-[#00f0ff] to-[#00ff66] relative"
                          initial={{ width: "0%" }}
                          animate={{ width: "100%" }}
                          transition={{ duration: 2, ease: "easeInOut" }}
                        >
                          <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]" />
                        </motion.div>
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={handleProcess}
                      className="w-full py-5 rounded-xl bg-gradient-to-r from-[#00f0ff] to-[#00ff66] text-black font-extrabold text-xl hover:shadow-[0_0_30px_rgba(0,255,102,0.5)] transition-all hover:scale-[1.02]"
                    >
                      Analyze Shot
                    </button>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.section>
  );
}
