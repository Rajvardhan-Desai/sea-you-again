"use client";

import { useQuery, useMutation, QueryClient } from "@tanstack/react-query";
import {
  apiFetch,
  ForecastOut,
  AOISummaryOut,
  RunSummary,
  AlertOut,
  PlaybookBand,
} from "./api";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5 * 60 * 1000 } },
});

// ── Public ────────────────────────────────────────────────────────────────────

export function useLatestForecast() {
  return useQuery<ForecastOut>({
    queryKey: ["forecast", "latest"],
    queryFn:  () => apiFetch<ForecastOut>("/forecast/latest"),
  });
}

export function useAoiSummary(geometry: object | null) {
  return useQuery<AOISummaryOut>({
    queryKey:  ["aoi-summary", geometry],
    queryFn:   () =>
      apiFetch<AOISummaryOut>("/forecast/aoi-summary", {
        method: "POST",
        body:   JSON.stringify({ geometry }),
      }),
    enabled: geometry !== null,
  });
}

export function usePlaybook() {
  return useQuery<{ bands: PlaybookBand[] }>({
    queryKey: ["playbook"],
    queryFn:  () => apiFetch<{ bands: PlaybookBand[] }>("/playbook"),
    staleTime: Infinity,
  });
}

// ── Admin ─────────────────────────────────────────────────────────────────────

export function useAdminRuns(limit = 50) {
  return useQuery<RunSummary[]>({
    queryKey: ["admin", "runs"],
    queryFn:  () => apiFetch<RunSummary[]>(`/admin/runs?limit=${limit}`),
  });
}

export function useAdminRunDetail(id: string) {
  return useQuery<RunSummary>({
    queryKey: ["admin", "run", id],
    queryFn:  () => apiFetch<RunSummary>(`/admin/runs/${id}`),
  });
}

export function useTriggerRun() {
  return useMutation({
    mutationFn: (run_date?: string) =>
      apiFetch("/admin/runs/trigger", {
        method: "POST",
        body:   JSON.stringify({ run_date }),
      }),
  });
}

export function useRetryRun() {
  return useMutation({
    mutationFn: ({ id, phase }: { id: string; phase: string }) =>
      apiFetch(`/admin/runs/${id}/retry?phase=${phase}`, { method: "POST" }),
  });
}
