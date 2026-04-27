import Link from "next/link";

export default function ConfirmPage({
  searchParams,
}: {
  searchParams: { ok?: string };
}) {
  const ok = searchParams.ok === "1";
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-ocean-900 text-ocean-100 px-6 text-center gap-6">
      {ok ? (
        <>
          <div className="text-5xl">✓</div>
          <h1 className="text-2xl font-bold">Subscription confirmed!</h1>
          <p className="max-w-sm text-ocean-100/70 text-sm">
            You will receive a bloom alert email (and SMS if configured) the next time
            MM-MARAS detects a risk event inside your area of interest.
          </p>
          <Link href="/map" className="bg-ocean-500 hover:bg-ocean-700 px-6 py-3 rounded-lg font-semibold transition text-sm">
            Open Forecast Map →
          </Link>
        </>
      ) : (
        <>
          <div className="text-5xl">✗</div>
          <h1 className="text-2xl font-bold">Confirmation failed</h1>
          <p className="text-ocean-100/70 text-sm">The link may have expired (24 h). Please subscribe again.</p>
          <Link href="/subscribe" className="underline text-sm text-ocean-100/50">← Subscribe</Link>
        </>
      )}
    </main>
  );
}
