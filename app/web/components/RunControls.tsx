"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/lib/api";

export function TriggerRunForm() {
  const router = useRouter();
  const [runDate, setRunDate] = useState("");
  const [busy,    setBusy]    = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await apiFetch("/admin/runs/trigger", {
        method:      "POST",
        body:        JSON.stringify(runDate ? { run_date: runDate } : {}),
        credentials: "include",
      });
      router.refresh();
    } catch (err: any) {
      setError(err?.message ?? "Trigger failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="mt-6 flex gap-3 items-center">
      <input
        type="date"
        value={runDate}
        onChange={(e) => setRunDate(e.target.value)}
        className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm"
      />
      <button
        type="submit"
        disabled={busy}
        className="bg-ocean-500 hover:bg-ocean-700 disabled:opacity-50 rounded px-4 py-2 text-sm font-semibold transition"
      >
        {busy ? "Triggering…" : "Trigger Run"}
      </button>
      {error && <span className="text-red-400 text-xs">{error}</span>}
    </form>
  );
}

export function RetryButton({ runId, phase = "ingest" }: { runId: string; phase?: string }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  async function onClick() {
    setBusy(true);
    try {
      await apiFetch(`/admin/runs/${runId}/retry?phase=${phase}`, {
        method:      "POST",
        credentials: "include",
      });
      router.refresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <button
      onClick={onClick}
      disabled={busy}
      className="text-yellow-400 hover:text-yellow-200 disabled:opacity-50"
    >
      {busy ? "Retrying…" : "Retry"}
    </button>
  );
}
