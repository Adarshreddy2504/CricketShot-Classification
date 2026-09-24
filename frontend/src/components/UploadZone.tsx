import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileVideo, X } from "lucide-react";
import { cn } from "../lib/utils";
import { api, type InferenceResult } from "../services/api";

interface UploadZoneProps {
  onUploadComplete: (file: File, result: InferenceResult) => void;
  onHoverStart?: () => void;
  onHoverEnd?: () => void;
}

export function UploadZone({
  onUploadComplete,
  onHoverStart,
  onHoverEnd,
}: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("Initializing...");

  /*
   * Stores the current backend task ID.
   */
  const taskIdRef = useRef<string | null>(null);

  /*
   * Stores the active SSE connection.
   */
  const eventSourceRef = useRef<EventSource | null>(null);

  /*
   * Prevents callbacks from showing an error
   * after the user has intentionally cancelled.
   */
  const cancellingRef = useRef(false);

  /*
   * Cleanup the current SSE connection and task reference.
   */
  const cleanupTask = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    taskIdRef.current = null;
  }, []);

  /*
   * Drag and drop handlers
   */
  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(true);
    },
    []
  );

  const handleDragLeave = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
    },
    []
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      const files = e.dataTransfer.files;

      if (!files || files.length === 0) {
        return;
      }

      const file = files[0];

      if (!file.type.startsWith("video/")) {
        alert("Please select a video file.");
        return;
      }

      setSelectedFile(file);
      setProgress(0);
      setStatusText("Ready to analyze");
    },
    []
  );

  /*
   * File picker
   */
  const handleFileSelect = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];

    if (!file) {
      return;
    }

    if (!file.type.startsWith("video/")) {
      alert("Please select a video file.");
      return;
    }

    setSelectedFile(file);
    setProgress(0);
    setStatusText("Ready to analyze");

    /*
     * Allows selecting the same file again later.
     */
    e.target.value = "";
  };

  /*
   * Start inference.
   */
  const handleProcess = async () => {
    if (!selectedFile || isLoading) {
      return;
    }

    try {
      setIsLoading(true);
      setIsCancelling(false);
      setProgress(0);
      setStatusText("Uploading video...");

      cancellingRef.current = false;

      /*
       * Step 1:
       * Upload the video and receive task_id.
       */
      const taskResponse = await api.predictShot(selectedFile);

      const taskId = taskResponse.task_id;

      if (!taskId) {
        throw new Error(
          "Backend did not return an inference task ID."
        );
      }

      taskIdRef.current = taskId;

      setStatusText("Starting AI inference...");

      /*
       * Step 2:
       * Listen to backend SSE progress.
       */
      eventSourceRef.current = api.streamProgress(
        taskId,

        /*
         * Progress callback
         */
        (update) => {
          /*
           * Backend has confirmed cancellation.
           */
          if (update.cancelled) {
            setProgress(update.progress ?? 0);
            setStatusText("Inference cancelled");

            setIsLoading(false);
            setIsCancelling(false);

            cancellingRef.current = false;

            cleanupTask();

            return;
          }

          /*
           * Normal progress update.
           */
          setProgress(
            Math.min(
              Math.max(update.progress ?? 0, 0),
              100
            )
          );

          setStatusText(
            update.status || "Processing video..."
          );
        },

        /*
         * Complete callback
         */
        (result) => {
          setProgress(100);
          setStatusText("Analysis complete");

          setIsLoading(false);
          setIsCancelling(false);

          cancellingRef.current = false;

          cleanupTask();

          /*
           * Send result back to App.tsx.
           */
          onUploadComplete(selectedFile, result);
        },

        /*
         * Error callback
         */
        (error) => {
          /*
           * If the user cancelled, don't show
           * a misleading connection error.
           */
          if (cancellingRef.current) {
            setStatusText("Inference cancelled");

            setIsLoading(false);
            setIsCancelling(false);

            cancellingRef.current = false;

            cleanupTask();

            return;
          }

          console.error(
            "Inference error:",
            error
          );

          setIsLoading(false);
          setIsCancelling(false);

          cleanupTask();

          setStatusText("Inference failed");

          alert(
            error.message ||
              "An error occurred while processing the video."
          );
        }
      );
    } catch (error) {
      console.error(
        "Failed to start inference:",
        error
      );

      setIsLoading(false);
      setIsCancelling(false);

      cancellingRef.current = false;

      cleanupTask();

      const message =
        error instanceof Error
          ? error.message
          : "An error occurred while uploading the video.";

      alert(message);
    }
  };

  /*
   * Cancel the backend task.
   */
  const handleCancel = async () => {
    const taskId = taskIdRef.current;

    if (!taskId || isCancelling) {
      return;
    }

    try {
      /*
       * Prevent multiple cancellation requests.
       */
      setIsCancelling(true);

      cancellingRef.current = true;

      setStatusText("Cancelling inference...");

      /*
       * This sends:
       *
       * DELETE /api/cancel/{task_id}
       */
      await api.cancelTask(taskId);

      /*
       * IMPORTANT:
       *
       * We do NOT immediately set isLoading(false).
       *
       * We wait for the backend SSE message:
       *
       * cancelled: true
       *
       * Then the progress callback performs
       * the final UI cleanup.
       */
    } catch (error) {
      console.error(
        "Cancellation error:",
        error
      );

      setIsCancelling(false);

      cancellingRef.current = false;

      const message =
        error instanceof Error
          ? error.message
          : "Unable to cancel the inference task.";

      alert(message);
    }
  };

  /*
   * Remove selected file.
   */
  const handleRemoveFile = () => {
    if (isLoading) {
      return;
    }

    setSelectedFile(null);
    setProgress(0);
    setStatusText("Initializing...");
    setIsCancelling(false);

    cancellingRef.current = false;

    cleanupTask();
  };

  return (
    <motion.section
      initial={{
        opacity: 0,
        y: 50,
      }}
      whileInView={{
        opacity: 1,
        y: 0,
      }}
      viewport={{
        once: true,
        margin: "-100px",
      }}
      transition={{
        type: "spring",
        stiffness: 100,
        damping: 15,
      }}
      className="w-full max-w-5xl mx-auto py-12 px-6 relative"
      id="upload-zone"
    >
      {/* Header */}

      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">
          Analyze Your Shot
        </h2>

        <p className="text-xl text-gray-400">
          Upload a video to see our AI in action.
        </p>
      </div>

      <div
        onMouseEnter={onHoverStart}
        onMouseLeave={onHoverEnd}
        className="relative"
      >
        {/* Rotating border */}

        <motion.div
          animate={{
            rotate: 360,
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: "linear",
          }}
          className="absolute inset-[-4px] rounded-[2rem] border-2 border-dashed border-[#00f0ff]/30 opacity-50 pointer-events-none"
        />

        {/* Main upload card */}

        <div
          className={cn(
            "relative rounded-[2rem] transition-all duration-500 overflow-hidden glass p-12 flex flex-col items-center justify-center min-h-[450px] z-10 bg-[#030712]/80",

            isDragging
              ? "border-[#00f0ff] bg-[#00f0ff]/10 scale-[1.02] shadow-[0_0_50px_rgba(0,240,255,0.2)]"
              : "border border-white/10 hover:border-white/30 hover:shadow-[0_0_40px_rgba(255,255,255,0.05)]",

            isLoading &&
              "border-[#00ff66] bg-[#00ff66]/5"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <AnimatePresence mode="wait">
            {/* ====================================================== */}
            {/* NO FILE SELECTED                                      */}
            {/* ====================================================== */}

            {!selectedFile ? (
              <motion.div
                key="upload-prompt"
                initial={{
                  opacity: 0,
                  scale: 0.95,
                }}
                animate={{
                  opacity: 1,
                  scale: 1,
                }}
                exit={{
                  opacity: 0,
                  scale: 0.95,
                }}
                transition={{
                  type: "spring",
                  stiffness: 200,
                  damping: 20,
                }}
                className="flex flex-col items-center text-center w-full"
              >
                {/* Upload icon */}

                <div className="w-24 h-24 rounded-2xl glass flex items-center justify-center mb-8 text-[#00f0ff] shadow-[0_0_30px_rgba(0,240,255,0.2)]">
                  <UploadCloud className="w-12 h-12" />
                </div>

                <h3 className="text-3xl font-bold mb-3">
                  Drag & Drop Video
                </h3>

                <p className="text-lg text-gray-400 mb-10 max-w-md">
                  Supported formats: MP4, MOV, AVI.
                  Maximum file size: 50MB.
                </p>

                {/* Browse button */}

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
              /* ==================================================== */
              /* FILE SELECTED                                        */
              /* ==================================================== */

              <motion.div
                key="file-selected"
                initial={{
                  opacity: 0,
                  scale: 0.95,
                }}
                animate={{
                  opacity: 1,
                  scale: 1,
                }}
                transition={{
                  type: "spring",
                  stiffness: 200,
                  damping: 20,
                }}
                className="flex flex-col items-center w-full max-w-lg"
              >
                <div className="w-full glass rounded-2xl p-8 relative">
                  {/* Remove file button */}

                  {!isLoading && (
                    <button
                      type="button"
                      onClick={handleRemoveFile}
                      className="absolute top-6 right-6 text-gray-400 hover:text-white transition-colors bg-white/5 hover:bg-white/20 rounded-full p-2"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  )}

                  {/* File information */}

                  <div className="flex items-center gap-6 mb-8">
                    <div className="relative">
                      {isLoading && (
                        <div className="absolute inset-0 rounded-xl bg-cyan-400/30 animate-ping" />
                      )}

                      <div
                        className={cn(
                          "p-5 rounded-xl border relative z-10 transition-colors",

                          isLoading
                            ? "bg-[#0b0f19] text-cyan-400 border-cyan-400/50"
                            : "bg-[#00f0ff]/10 text-[#00f0ff] border-[#00f0ff]/20"
                        )}
                      >
                        <FileVideo className="w-10 h-10" />
                      </div>
                    </div>

                    <div className="flex-1 overflow-hidden">
                      <h4
                        className="font-bold text-xl truncate text-white"
                        title={selectedFile.name}
                      >
                        {selectedFile.name}
                      </h4>

                      <p className="text-gray-400 font-medium mt-1">
                        {(
                          selectedFile.size /
                          (1024 * 1024)
                        ).toFixed(2)}{" "}
                        MB
                      </p>
                    </div>
                  </div>

                  {/* ================================================= */}
                  {/* PROCESSING STATE                                  */}
                  {/* ================================================= */}

                  {isLoading ? (
                    <div className="w-full mt-4">
                      {/* Status */}

                      <div className="flex items-center justify-center mb-5">
                        <span
                          className={cn(
                            "text-sm font-mono tracking-wide text-center",
                            isCancelling
                              ? "text-red-400"
                              : "text-cyan-400"
                          )}
                        >
                          {statusText}
                        </span>
                      </div>

                      {/* Progress bar */}

                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden relative">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-emerald-400 to-cyan-500 transition-all duration-500"
                          style={{
                            width: `${Math.min(
                              Math.max(progress, 0),
                              100
                            )}%`,
                          }}
                        />
                      </div>

                      {/* Percentage */}

                      <div className="flex justify-center mt-3">
                        <span className="text-xs font-mono text-gray-400">
                          {Math.round(progress)}%
                        </span>
                      </div>

                      {/* Cancel button */}

                      <div className="relative mt-8">
                        <button
                          type="button"
                          onClick={handleCancel}
                          disabled={isCancelling}
                          className="group relative flex items-center justify-center gap-2 px-6 py-2.5 mx-auto rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 font-mono text-xs font-bold uppercase tracking-widest backdrop-blur-md overflow-hidden transition-all duration-300 hover:bg-red-500/20 hover:border-red-500/50 hover:shadow-[0_0_20px_rgba(239,68,68,0.3)] disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {/* Stop icon */}

                          <svg
                            className="w-4 h-4 text-red-500"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path
                              fillRule="evenodd"
                              d="M4.5 4.5a1 1 0 011-1h9a1 1 0 011 1v9a1 1 0 01-1 1h-9a1 1 0 01-1-1v-9z"
                              clipRule="evenodd"
                            />
                          </svg>

                          {isCancelling
                            ? "Cancelling..."
                            : "Abort Inference"}
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* =============================================== */
                    /* READY TO ANALYZE                                */
                    /* =============================================== */

                    <button
                      type="button"
                      onClick={handleProcess}
                      className="w-full py-5 mt-4 rounded-xl bg-gradient-to-r from-[#00f0ff] to-[#00ff66] text-black font-extrabold text-xl hover:shadow-[0_0_30px_rgba(0,255,102,0.5)] transition-all hover:scale-[1.02]"
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