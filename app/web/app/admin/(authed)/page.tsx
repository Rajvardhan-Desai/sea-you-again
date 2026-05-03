import Link from "next/link";
import { RunStatusPill } from "@/components/RunStatusPill";
import { TriggerRunForm, RetryButton } from "@/components/RunControls";
import type { RunSummary } from "@/lib/api";

export const dynamic = "force-dynamic";

async function fetchRuns(): Promise<RunSummary[]> {
  const base  = process.env.INTERNAL_API_URL ?? "http://api:8000";
  const token = process.env.ADMIN_TOKEN ?? "";
  try {
    const res = await fetch(`${base}/api/admin/runs?limit=50`, {
      headers: { Authorization: `Bearer ${token}` },
      cache:   "no-store",
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

function durationStr(start?: string, end?: string): string {
  if (!start || !end) return "—";
  const s = Math.round((new Date(end).getTime() - new Date(start).getTime()) / 1000);
  if (s < 60)   return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

export default async function AdminRunsPage() {
  const runs = await fetchRuns();

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Pipeline Runs</h1>
        <p className="text-xs text-gray-500">{runs.length} runs shown</p>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-800">
        <table className="w-full text-sm">
          <thead className="bg-gray-900 text-gray-400 text-xs uppercase">
            <tr>
              <th className="px-4 py-3 text-left">Date</th>
              <th className="px-4 py-3 text-left">Status</th>
              <th className="px-4 py-3 text-left">Duration</th>
              <th className="px-4 py-3 text-left">Triggered by</th>
              <th className="px-4 py-3 text-left">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {runs.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                  No runs yet. Trigger one below.
                </td>
              </tr>
            )}
            {runs.map((run) => (
              <tr key={run.id} className="hover:bg-gray-900/50 transition">
                <td className="px-4 py-3 font-mono">{run.run_date}</td>
                <td className="px-4 py-3"><RunStatusPill status={run.status} /></td>
                <td className="px-4 py-3 text-gray-400 font-mono text-xs">
                  {durationStr(run.started_at, run.finished_at)}
                </td>
                <td className="px-4 py-3 text-gray-400 text-xs">{run.triggered_by ?? "—"}</td>
                <td className="px-4 py-3 flex gap-3 text-xs">
                  <Link
                    href={`/admin/runs/${run.id}`}
                    className="text-ocean-400 hover:text-ocean-200 underline"
                  >
                    Detail
                  </Link>
                  {run.status === "failed" && (
                    <RetryButton runId={run.id} />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <TriggerRunForm />
    </div>
  );
}
