import Link from "next/link";

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-ocean-900 text-ocean-100 px-6 py-10 max-w-2xl mx-auto">
      <Link href="/" className="text-sm text-ocean-100/50 hover:text-white mb-6 inline-block">← Home</Link>
      <h1 className="text-3xl font-bold mb-6">About MM-MARAS</h1>

      <section className="space-y-4 text-sm text-ocean-100/80 leading-relaxed">
        <p>
          <strong className="text-ocean-100">MM-MARAS</strong> (Multi-Modal Mask-Aware
          Regime-Adaptive Spatiotemporal Model) is a 46.4 M-parameter PyTorch model
          trained to reconstruct and forecast chlorophyll-a (Chl-a) concentrations in
          the Bay of Bengal from multi-source satellite and reanalysis data.
        </p>

        <h2 className="text-xl font-semibold text-ocean-100 mt-6">Inputs</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>CMEMS BGC Chl-a + biogeochemical auxiliary fields (0.25°/day)</li>
          <li>CMEMS ocean physics — SST, currents, MLD (0.083°/day)</li>
          <li>ERA5 surface winds, MSLP, precipitation (daily)</li>
          <li>GloFAS freshwater discharge (daily)</li>
          <li>CMEMS bathymetry (static)</li>
        </ul>

        <h2 className="text-xl font-semibold text-ocean-100 mt-6">Outputs (5-day horizon)</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>Gap-filled Chl-a reconstruction (aleatoric uncertainty)</li>
          <li>5-day Chl-a forecast per pixel</li>
          <li>Harmful algal bloom (HAB) probability per horizon</li>
          <li>Ecosystem Risk Index (ERI) — ordinal class 0–4 per pixel</li>
          <li>Composite ecosystem impact score 0–1</li>
        </ul>

        <h2 className="text-xl font-semibold text-ocean-100 mt-6">Performance (v3.5, 260 test patches)</h2>
        <table className="w-full text-xs mt-2 border-collapse">
          <tbody>
            {[
              ["Gap RMSE",           "0.402"],
              ["Valid RMSE",         "0.082"],
              ["Forecast +1d RMSE",  "0.103"],
              ["Forecast +5d RMSE",  "0.184"],
              ["Bloom Macro F1",     "0.824"],
              ["ERI Macro F1",       "0.926"],
            ].map(([metric, val]) => (
              <tr key={metric} className="border-t border-ocean-100/10">
                <td className="py-1.5 text-ocean-100/60">{metric}</td>
                <td className="py-1.5 text-right font-mono">{val}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h2 className="text-xl font-semibold text-ocean-100 mt-6">Architecture</h2>
        <p>
          Mixture-of-Experts spatiotemporal decoder (4 experts), mask-aware optical
          encoder, physics/discharge/BGC encoders, temporal ConvLSTM with cross-attention,
          ordinal-focal ERI head, heteroscedastic reconstruction head, binary bloom head.
        </p>

        <div className="mt-8 p-4 border border-ocean-100/20 rounded-lg text-xs text-ocean-100/40">
          <strong className="text-ocean-100/60">Disclaimer:</strong> MM-MARAS is a
          research prototype. Forecasts carry inherent model uncertainty. Do not use
          as the sole basis for regulatory or commercial decisions without corroborating
          field observations.
        </div>
      </section>
    </main>
  );
}
