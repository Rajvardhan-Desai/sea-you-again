import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MM-MARAS | Bay of Bengal Bloom Early Warning",
  description:
    "Real-time chlorophyll-a forecasting, harmful algal bloom alerts, and Ecosystem Risk Index for the Bay of Bengal.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-ocean-900 text-white antialiased">{children}</body>
    </html>
  );
}
