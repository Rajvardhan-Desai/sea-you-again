"use client";

import { useState } from "react";
import Link from "next/link";
import { useLatestForecast, useAoiSummary } from "@/lib/queries";
import { MapView } from "@/components/MapView";

export default function MapPage() {
  const { data: forecast, isLoading, error } = useLatestForecast();
  const [aoiGeom, setAoiGeom] = useState<object | null>(null);
  const { data: aoiSummary } = useAoiSummary(aoiGeom);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-ocean-900 text-ocean-100">
        Loading latest forecast…
      </div>
    );
  }
  if (error || !forecast) {
    const isServiceDown = error && /^5\d\d/.test((error as Error).message);
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-ocean-900 text-red-400 gap-4">
        <p>{isServiceDown
          ? "Forecast service unavailable — try again shortly."
          : "No forecast data available yet."}</p>
        <Link href="/" className="text-ocean-100 underline text-sm">← Back</Link>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-ocean-900">
      {/* Nav */}
      <nav className="flex items-center justify-between px-4 h-16 bg-ocean-900/90 border-b border-ocean-100/10 z-50">
        <Link href="/" className="text-lg font-bold tracking-tight">MM-MARAS</Link>
        <div className="flex gap-4 text-sm text-ocean-100/70">
          <Link href="/subscribe" className="hover:text-white transition">Subscribe</Link>
          <Link href="/playbook"  className="hover:text-white transition">Playbook</Link>
          <Link href="/about"     className="hover:text-white transition">About</Link>
          <Link href="/admin"     className="hover:text-white transition">Admin</Link>
        </div>
      </nav>

      {/* Map */}
      <div className="flex-1 relative">
        <MapView forecast={forecast} onAOIDraw={setAoiGeom} />

        {/* AOI summary panel */}
        {aoiSummary && (
          <div className="absolute bottom-20 left-4 z-[1000] bg-ocean-900/90 rounded-lg p-4 w-72 text-sm">
            <p className="font-semibold mb-2 text-ocean-100">AOI Summary</p>
            <table className="w-full text-xs text-ocean-100/80">
              <thead>
                <tr>
                  <th className="text-left">Day</th>
                  <th>Bloom %</th>
                  <th>ERI</th>
                  <th>Chl-a</th>
                </tr>
              </thead>
              <tbody>
                {[1, 2, 3, 4, 5].map((d, i) => (
                  <tr key={d} className="border-t border-ocean-100/10">
                    <td>D+{d}</td>
                    <td className="text-center">
                      {(aoiSummary.max_bloom_prob_per_horizon[i] * 100).toFixed(0)}%
                    </td>
                    <td className="text-center">{aoiSummary.max_eri_per_horizon[i]}</td>
                    <td className="text-center">
                      {aoiSummary.mean_chl_per_horizon[i].toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-ocean-100/40">
              {aoiSummary.pixel_count} ocean pixels in AOI
            </p>
            <Link
              href="/subscribe"
              className="mt-3 block w-full text-center bg-ocean-500 hover:bg-ocean-700 rounded px-3 py-1.5 text-xs font-semibold transition"
            >
              Subscribe alerts for this zone →
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
