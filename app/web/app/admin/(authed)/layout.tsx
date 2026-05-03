import Link from "next/link";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";

export default function AdminAuthedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  if (!cookies().get("mm_admin")) {
    redirect("/admin/login");
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <nav className="flex items-center gap-6 px-6 h-14 bg-gray-900 border-b border-gray-800 text-sm">
        <Link href="/" className="font-bold text-ocean-100">MM-MARAS</Link>
        <span className="text-gray-500">Admin</span>
        <Link href="/admin"              className="hover:text-white text-gray-400">Runs</Link>
        <Link href="/admin/subscriptions" className="hover:text-white text-gray-400">Subscriptions</Link>
        <Link href="/map"                className="ml-auto hover:text-white text-gray-400">← Map</Link>
      </nav>
      <main className="px-6 py-8 max-w-5xl mx-auto">{children}</main>
    </div>
  );
}
