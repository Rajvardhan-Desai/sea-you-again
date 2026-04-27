import Link from "next/link";
import { RunStatusPill } from "@/components/RunStatusPill";
import type { RunSummary } from "@/lib/api";

async function fetchRun(id: string): Promise<RunSummary | null> {
  const base  = process.env.INTERNAL_API_URL ?? "http://api:8000";
  const token = process.env.ADMIN_TOKEN ?? "";
  const res   = await fetch(`${base}/api/admin/runs/${id}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache:   "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export default async function RunDetailPage({ params }: { params: { id: string } }) {
  const run = await fetchRun(params.id);

  if (!run) {
    return (
      <div>
        <Link href="/admin" className="text-gray-500 text-sm hover:text-white mb-4 inline-block">← Runs</Link>
        <p className="text-red-400">Run not found.</p>
      </div>
    );
  }

  return (
    <div>
      <Link href="/admin" className="text-gray-500 text-sm hover:text-white mb-4 inline-block">← Runs</Link>
      <div className="flex items-center gap-4 mb-6">
        <h1 className="text-2xl font-bold font-mono">{run.run_date}</h1>
        <RunStatusPill status={run.status} />
      </div>

      <dl className="grid grid-cols-2 gap-4 text-sm mb-8">
        {[
          ["Started",      run.started_at ?? "—"],
          ["Finished",     run.finished_at ?? "—"],
          ["Triggered by", run.triggered_by ?? "—"],
          ["Artifacts",    run.artifacts_path ?? "—"],
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-gray-500 text-xs mb-0.5">{k}</dt>
            <dd className="font-mono text-xs break-all">{v}</dd>
          </div>
        ))}
      </dl>

      {run.error_text && (
        <div className="mb-6 p-4 bg-red-950 border border-red-800 rounded-lg">
          <p className="text-red-400 text-xs font-semibold mb-1">Error</p>
          <pre className="text-red-300 text-xs whitespace-pre-wrap">{run.error_text}</pre>
        </div>
      )}

      {run.inference_metrics && (
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-2">Inference metrics</h2>
          <pre className="bg-gray-900 rounded-lg p-4 text-xs overflow-auto">
            {JSON.stringify(run.inference_metrics, null, 2)}
          </pre>
        </div>
      )}

      {run.ingest_summary && (
        <div>
          <h2 className="text-lg font-semibold mb-2">Ingest summary</h2>
          <pre className="bg-gray-900 rounded-lg p-4 text-xs overflow-auto">
            {JSON.stringify(run.ingest_summary, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
