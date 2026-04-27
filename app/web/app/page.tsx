import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen px-6 text-center bg-gradient-to-b from-ocean-900 to-ocean-700">
      <h1 className="text-4xl font-bold tracking-tight mb-3">
        MM-MARAS
      </h1>
      <p className="text-ocean-100 text-lg max-w-2xl mb-2">
        Multi-Modal Mask-Aware Regime-Adaptive Spatiotemporal Model
      </p>
      <p className="text-ocean-100/70 max-w-xl mb-10">
        5-day chlorophyll-a forecast, harmful algal bloom early warning, and
        Ecosystem Risk Index for the Bay of Bengal.
      </p>

      <div className="flex gap-4 flex-wrap justify-center">
        <Link
          href="/map"
          className="px-6 py-3 bg-ocean-500 hover:bg-ocean-700 rounded-lg font-semibold transition"
        >
          Open Forecast Map
        </Link>
        <Link
          href="/subscribe"
          className="px-6 py-3 border border-ocean-100 hover:bg-ocean-100/10 rounded-lg font-semibold transition"
        >
          Subscribe to Alerts
        </Link>
        <Link
          href="/playbook"
          className="px-6 py-3 border border-ocean-100/40 hover:bg-ocean-100/10 rounded-lg transition text-sm"
        >
          Response Playbook
        </Link>
        <Link
          href="/about"
          className="px-6 py-3 border border-ocean-100/40 hover:bg-ocean-100/10 rounded-lg transition text-sm"
        >
          About the Model
        </Link>
      </div>

      <p className="mt-16 text-xs text-ocean-100/40">
        Forecast accuracy: Gap RMSE 0.402 · Bloom Macro F1 0.824 · ERI Macro F1 0.926
      </p>
    </main>
  );
}
