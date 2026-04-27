import Link from "next/link";
import { PlaybookCard } from "@/components/PlaybookCard";
import type { PlaybookBand } from "@/lib/api";

async function fetchPlaybook(): Promise<{ bands: PlaybookBand[] }> {
  const base = process.env.INTERNAL_API_URL ?? "http://api:8000";
  const res  = await fetch(`${base}/api/playbook`, { next: { revalidate: 3600 } });
  if (!res.ok) return { bands: [] };
  return res.json();
}

export default async function PlaybookPage() {
  const { bands } = await fetchPlaybook();

  return (
    <main className="min-h-screen bg-ocean-900 text-ocean-100 px-6 py-10 max-w-3xl mx-auto">
      <Link href="/" className="text-sm text-ocean-100/50 hover:text-white mb-6 inline-block">← Home</Link>
      <h1 className="text-3xl font-bold mb-2">Response Playbook</h1>
      <p className="text-ocean-100/60 text-sm mb-8">
        Context-aware actions for fisheries, aquaculture operators, and coastal authorities
        based on MM-MARAS bloom probability and Ecosystem Risk Index (ERI) forecasts.
      </p>

      <div className="flex flex-col gap-4">
        {bands.length === 0 && (
          <p className="text-ocean-100/40 text-sm">
            Playbook unavailable — API may be starting up.
          </p>
        )}
        {bands.map((band) => (
          <PlaybookCard key={band.severity} band={band} />
        ))}
      </div>

      <div className="mt-10 p-4 border border-ocean-100/20 rounded-lg text-xs text-ocean-100/40">
        <p className="font-semibold mb-1 text-ocean-100/60">Disclaimer</p>
        MM-MARAS is a research model. Bloom predictions carry inherent uncertainty.
        Always supplement automated forecasts with local field observations and consult
        district fisheries officers before major operational decisions.
      </div>
    </main>
  );
}
