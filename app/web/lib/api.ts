const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Type definitions mirroring FastAPI schemas ────────────────────────────

export interface LayerURLs {
  recon:    string[];
  forecast: string[];
  bloom:    string[];
  eri:      string[];
  impact:   string[];
}

export interface ForecastOut {
  run_id:           string;
  run_date:         string;
  horizons:         string[];
  layers:           LayerURLs;
  overlay_base_url: string;
  bbox:             [number, number, number, number];
  crs:              string;
  inference_metrics?: Record<string, unknown>;
}

export interface AOISummaryOut {
  max_bloom_prob_per_horizon: number[];
  max_eri_per_horizon:        number[];
  mean_chl_per_horizon:       number[];
  pixel_count:                number;
}

export interface RunSummary {
  id:                string;
  run_date:          string;
  status:            string;
  started_at?:       string;
  finished_at?:      string;
  triggered_by?:     string;
  ingest_summary?:   Record<string, unknown>;
  inference_metrics?: Record<string, unknown>;
  error_text?:       string;
}

export interface AlertOut {
  id:              string;
  run_id:          string;
  severity?:       number;
  max_bloom_prob?: number;
  max_eri?:        number;
  horizon_of_max?: number;
  sent_at?:        string;
  channels_sent?:  string[];
}

export interface PlaybookBand {
  severity:       string;
  bloom_prob_min: number;
  bloom_prob_max: number;
  eri_class:      number;
  actions:        string[];
}
