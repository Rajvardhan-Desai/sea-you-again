import type { PlaybookBand } from "@/lib/api";

const SEV_COLORS: Record<string, string> = {
  low:      "border-green-500  bg-green-900/30  text-green-300",
  moderate: "border-yellow-500 bg-yellow-900/30 text-yellow-300",
  high:     "border-orange-500 bg-orange-900/30 text-orange-300",
  severe:   "border-red-500    bg-red-900/40    text-red-300",
};

export function PlaybookCard({ band }: { band: PlaybookBand }) {
  const cls = SEV_COLORS[band.severity] ?? "border-gray-500 bg-gray-900/30 text-gray-300";
  return (
    <div className={`border-l-4 rounded-lg p-4 ${cls}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-bold uppercase tracking-wide">
          {band.severity}
        </span>
        <span className="text-xs opacity-70">
          bloom {(band.bloom_prob_min * 100).toFixed(0)}–{(band.bloom_prob_max * 100).toFixed(0)} % · ERI {band.eri_class}
        </span>
      </div>
      <ul className="text-sm space-y-1 opacity-90">
        {band.actions.map((a, i) => (
          <li key={i} className="flex gap-2">
            <span className="opacity-50">•</span>
            <span>{a}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
