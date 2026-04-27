"use client";

import { useEffect, useState } from "react";
import type { ForecastOut } from "@/lib/api";
import type { LayerKey } from "./LayerSwitcher";
import { LayerSwitcher } from "./LayerSwitcher";
import { HorizonSlider } from "./HorizonSlider";

// Leaflet must be imported dynamically in Next.js (no SSR)
let L: typeof import("leaflet") | null = null;

interface Props {
  forecast:    ForecastOut;
  onAOIDraw?:  (geom: object) => void;
}

export function MapView({ forecast, onAOIDraw }: Props) {
  const [activeLayer, setActiveLayer] = useState<LayerKey>("bloom");
  const [horizonIdx,  setHorizonIdx]  = useState(0);
  const [mapReady,    setMapReady]    = useState(false);
  const [mapRef,      setMapRef]      = useState<import("leaflet").Map | null>(null);
  const [overlayRef,  setOverlayRef]  = useState<import("leaflet").ImageOverlay | null>(null);

  // Load Leaflet client-side only
  useEffect(() => {
    import("leaflet").then((leaflet) => {
      L = leaflet.default ?? leaflet;
      // Fix default marker icon paths for Next.js bundling
      delete (L.Icon.Default.prototype as any)._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconUrl:       "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        shadowUrl:     "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      });
      setMapReady(true);
    });
  }, []);

  // Initialise map
  useEffect(() => {
    if (!mapReady || !L) return;
    const container = document.getElementById("leaflet-map");
    if (!container || (container as any)._leaflet_id) return;

    const [minLon, minLat, maxLon, maxLat] = forecast.bbox;
    const map = L.map(container, {
      center: [(minLat + maxLat) / 2, (minLon + maxLon) / 2],
      zoom:   6,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors",
      maxZoom:     18,
    }).addTo(map);

    setMapRef(map);

    // ── AOI drawing (leaflet-draw) ──
    if (onAOIDraw) {
      import("leaflet-draw").then(() => {
        if (!L || !map) return;
        const drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);
        const drawControl = new (L as any).Control.Draw({
          edit:  { featureGroup: drawnItems, remove: true },
          draw:  {
            polygon:   true,
            rectangle: true,
            marker:    false,
            circle:    false,
            polyline:  false,
            circlemarker: false,
          },
        });
        map.addControl(drawControl);
        map.on((L as any).Draw.Event.CREATED, (e: any) => {
          drawnItems.clearLayers();
          drawnItems.addLayer(e.layer);
          const geoJSON = e.layer.toGeoJSON().geometry;
          onAOIDraw(geoJSON);
        });
      });
    }

    return () => { map.remove(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapReady]);

  // Update overlay when layer or horizon changes
  useEffect(() => {
    if (!mapRef || !L) return;
    const [minLon, minLat, maxLon, maxLat] = forecast.bbox;
    const bounds: [[number, number], [number, number]] = [[minLat, minLon], [maxLat, maxLon]];

    const urls = forecast.layers[activeLayer];
    if (!urls || urls.length === 0) return;
    const url = urls[Math.min(horizonIdx, urls.length - 1)];

    if (overlayRef) {
      overlayRef.remove();
    }
    const newOverlay = L.imageOverlay(url, bounds, { opacity: 0.75 });
    newOverlay.addTo(mapRef);
    setOverlayRef(newOverlay);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapRef, activeLayer, horizonIdx, forecast]);

  return (
    <div className="relative w-full map-full">
      {/* Leaflet container */}
      <div id="leaflet-map" className="absolute inset-0 z-0" />

      {/* Controls overlay */}
      <div className="absolute top-4 right-4 z-[1000] flex flex-col gap-3 pointer-events-auto">
        <LayerSwitcher activeLayer={activeLayer} onChange={setActiveLayer} />
        {forecast.horizons.length > 0 && (
          <HorizonSlider
            horizons={forecast.horizons}
            selectedHorizon={horizonIdx}
            onChange={setHorizonIdx}
          />
        )}
      </div>

      {/* Run date badge */}
      <div className="absolute bottom-6 left-4 z-[1000] bg-ocean-900/80 text-xs text-ocean-100/70 px-3 py-1.5 rounded-lg">
        Run: {forecast.run_date} · Horizon: {forecast.horizons[horizonIdx] ?? "—"}
      </div>
    </div>
  );
}
