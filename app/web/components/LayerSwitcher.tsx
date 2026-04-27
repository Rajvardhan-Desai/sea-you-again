"use client";

export type LayerKey = "bloom" | "eri" | "impact" | "forecast" | "recon";

const LAYERS: { key: LayerKey; label: string; desc: string }[] = [
  { key: "bloom",    label: "Bloom Probability", desc: "HAB risk per day 0–1" },
  { key: "eri",      label: "Ecosystem Risk",    desc: "ERI class 0–4" },
  { key: "impact",   label: "Impact Score",      desc: "Combined ecosystem impact 0–1" },
  { key: "forecast", label: "Chl-a Forecast",   desc: "mg/m³ log-scaled" },
  { key: "recon",    label: "Chl-a Recon",       desc: "Gap-filled last step" },
];

interface Props {
  activeLayer: LayerKey;
  onChange:    (key: LayerKey) => void;
}

export function LayerSwitcher({ activeLayer, onChange }: Props) {
  return (
    <div className="flex flex-col gap-1 bg-ocean-900/80 rounded-lg p-3 min-w-[200px]">
      <span className="text-xs text-ocean-100/60 font-medium uppercase tracking-wide mb-1">
        Layer
      </span>
      {LAYERS.map(({ key, label, desc }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          title={desc}
          className={`text-left px-3 py-1.5 rounded text-sm transition ${
            key === activeLayer
              ? "bg-ocean-500 text-white font-semibold"
              : "hover:bg-ocean-100/10 text-ocean-100/70"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
