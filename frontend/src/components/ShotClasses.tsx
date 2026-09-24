export function ShotClasses() {
  const classes = [
    "Cover Drive", "Forward Defense", "Flick", "Hook", "Late Cut", 
    "Lofted Drive", "Pull", "Square Cut", "Straight Drive", "Sweep"
  ];

  return (
    <div className="max-w-4xl mx-auto py-12">
      <h2 className="text-sm font-mono text-gray-400 uppercase tracking-widest mb-6 text-center">
        Supported Classification Classes
      </h2>
      <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 shadow-2xl flex flex-wrap justify-center gap-4">
        {classes.map((shotClass) => (
          <div 
            key={shotClass}
            className="group flex items-center gap-3 px-5 py-2.5 rounded-full bg-[#0b0f19]/80 border border-white/20 text-gray-200 font-mono text-xs uppercase tracking-widest transition-all duration-300 hover:-translate-y-1 hover:border-cyan-400 hover:shadow-[0_4px_20px_rgba(0,255,255,0.25)] hover:text-cyan-300 cursor-default"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,255,255,0.8)] transition-all duration-300 group-hover:animate-[pulse_0.5s_cubic-bezier(0.4,0,0.6,1)_infinite]"></span>
            {shotClass}
          </div>
        ))}
      </div>
    </div>
  );
}
