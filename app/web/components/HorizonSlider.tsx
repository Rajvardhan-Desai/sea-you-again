"use client";

interface Props {
  horizons:         string[];   // ["2026-04-26", ...]
  selectedHorizon:  number;     // 0-based index
  onChange:         (idx: number) => void;
}

export function HorizonSlider({ horizons, selectedHorizon, onChange }: Props) {
  return (
    <div className="flex flex-col gap-1 bg-ocean-900/80 rounded-lg p-3 min-w-[180px]">
      <span className="text-xs text-ocean-100/60 font-medium uppercase tracking-wide mb-1">
        Forecast Horizon
      </span>
      {horizons.map((date, idx) => (
        <button
          key={date}
          onClick={() => onChange(idx)}
          className={`text-left px-3 py-1.5 rounded text-sm transition ${
            idx === selectedHorizon
              ? "bg-ocean-500 text-white font-semibold"
              : "hover:bg-ocean-100/10 text-ocean-100/70"
          }`}
        >
          D+{idx + 1} &middot; {date}
        </button>
      ))}
    </div>
  );
}
