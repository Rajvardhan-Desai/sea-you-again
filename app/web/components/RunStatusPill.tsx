const STATUS_STYLES: Record<string, string> = {
  succeeded: "bg-green-600 text-white",
  partial:   "bg-yellow-500 text-black",
  failed:    "bg-red-600 text-white",
  ingesting: "bg-blue-500 text-white animate-pulse",
  inferring: "bg-blue-500 text-white animate-pulse",
  alerting:  "bg-blue-400 text-white animate-pulse",
  pending:   "bg-gray-500 text-white",
};

export function RunStatusPill({ status }: { status: string }) {
  const cls = STATUS_STYLES[status] ?? "bg-gray-400 text-white";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {status}
    </span>
  );
}
