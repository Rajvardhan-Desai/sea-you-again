import Link from "next/link";

async function fetchSubscriptions() {
  const base  = process.env.INTERNAL_API_URL ?? "http://api:8000";
  const token = process.env.ADMIN_TOKEN ?? "";
  const res   = await fetch(`${base}/api/admin/subscriptions?limit=100`, {
    headers: { Authorization: `Bearer ${token}` },
    cache:   "no-store",
  });
  if (!res.ok) return { total: 0, items: [] };
  return res.json();
}

export default async function AdminSubscriptionsPage() {
  const { total, items } = await fetchSubscriptions();

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Subscriptions</h1>
        <p className="text-xs text-gray-500">{total} total</p>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-800">
        <table className="w-full text-sm">
          <thead className="bg-gray-900 text-gray-400 text-xs uppercase">
            <tr>
              <th className="px-4 py-3 text-left">Name</th>
              <th className="px-4 py-3 text-left">Email</th>
              <th className="px-4 py-3 text-left">Channels</th>
              <th className="px-4 py-3 text-left">Threshold</th>
              <th className="px-4 py-3 text-left">Confirmed</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {items.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                  No subscriptions yet.
                </td>
              </tr>
            )}
            {items.map((s: any) => (
              <tr key={s.id} className="hover:bg-gray-900/50 transition">
                <td className="px-4 py-3">{s.name}</td>
                <td className="px-4 py-3 text-gray-400 text-xs">{s.contact_email}</td>
                <td className="px-4 py-3 text-xs text-gray-400">{s.channels.join(", ")}</td>
                <td className="px-4 py-3 text-xs font-mono">
                  {(s.severity_threshold * 100).toFixed(0)}%
                </td>
                <td className="px-4 py-3 text-xs">
                  {s.confirmed_at
                    ? <span className="text-green-400">✓ {s.confirmed_at.slice(0, 10)}</span>
                    : <span className="text-yellow-500">Pending</span>
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
