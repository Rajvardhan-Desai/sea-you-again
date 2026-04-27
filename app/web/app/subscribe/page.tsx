"use client";

import { useState } from "react";
import Link from "next/link";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiFetch } from "@/lib/api";

const schema = z.object({
  name:               z.string().min(1, "Name required"),
  contact_email:      z.string().email("Valid email required"),
  contact_phone:      z.string().optional(),
  severity_threshold: z.number().min(0).max(1),
  channels:           z.array(z.string()).min(1, "Select at least one channel"),
});

type FormValues = z.infer<typeof schema>;

export default function SubscribePage() {
  const [drawnGeom, setDrawnGeom] = useState<object | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [apiError,  setApiError]  = useState<string | null>(null);

  const { register, handleSubmit, watch, setValue, formState: { errors, isSubmitting } } =
    useForm<FormValues>({
      resolver: zodResolver(schema),
      defaultValues: {
        severity_threshold: 0.5,
        channels:           ["email"],
      },
    });

  async function onSubmit(values: FormValues) {
    if (!drawnGeom) {
      setApiError("Please draw your area of interest on the map below.");
      return;
    }
    setApiError(null);
    try {
      await apiFetch("/subscriptions", {
        method: "POST",
        body:   JSON.stringify({ ...values, geometry: drawnGeom }),
      });
      setSubmitted(true);
    } catch (err: any) {
      setApiError(err.message ?? "Submission failed.");
    }
  }

  if (submitted) {
    return (
      <main className="flex flex-col items-center justify-center min-h-screen bg-ocean-900 text-ocean-100 px-6 text-center gap-4">
        <h1 className="text-2xl font-bold">Check your email</h1>
        <p className="max-w-md text-ocean-100/70">
          A confirmation link has been sent. Click it to activate your bloom alert subscription.
        </p>
        <Link href="/" className="underline text-sm text-ocean-100/50 mt-4">← Home</Link>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-ocean-900 text-ocean-100 px-6 py-10 max-w-2xl mx-auto">
      <Link href="/" className="text-sm text-ocean-100/50 hover:text-white mb-6 inline-block">← Back</Link>
      <h1 className="text-3xl font-bold mb-2">Subscribe to Bloom Alerts</h1>
      <p className="text-ocean-100/60 mb-8 text-sm">
        Draw your farm or fishing zone on the map, set a risk threshold, and receive
        email/SMS notifications when a harmful algal bloom is forecast over your area.
      </p>

      <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-5">
        {/* Name */}
        <div>
          <label className="block text-sm mb-1">Name / Farm name</label>
          <input
            {...register("name")}
            className="w-full bg-ocean-900 border border-ocean-100/30 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ocean-500"
            placeholder="Rajvardhan Aqua Farms"
          />
          {errors.name && <p className="text-red-400 text-xs mt-1">{errors.name.message}</p>}
        </div>

        {/* Email */}
        <div>
          <label className="block text-sm mb-1">Email address</label>
          <input
            {...register("contact_email")}
            type="email"
            className="w-full bg-ocean-900 border border-ocean-100/30 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ocean-500"
            placeholder="you@example.com"
          />
          {errors.contact_email && <p className="text-red-400 text-xs mt-1">{errors.contact_email.message}</p>}
        </div>

        {/* Phone */}
        <div>
          <label className="block text-sm mb-1">Phone (optional, for SMS)</label>
          <input
            {...register("contact_phone")}
            type="tel"
            className="w-full bg-ocean-900 border border-ocean-100/30 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ocean-500"
            placeholder="+91 98765 43210"
          />
        </div>

        {/* Severity threshold */}
        <div>
          <label className="block text-sm mb-1">
            Alert threshold — {Math.round(watch("severity_threshold") * 100)}% bloom probability
          </label>
          <input
            type="range" min="0" max="1" step="0.05"
            {...register("severity_threshold", { valueAsNumber: true })}
            className="w-full accent-ocean-500"
          />
          <div className="flex justify-between text-xs text-ocean-100/40 mt-1">
            <span>0% (all alerts)</span><span>100% (severe only)</span>
          </div>
        </div>

        {/* Channels */}
        <div>
          <label className="block text-sm mb-2">Notification channels</label>
          <div className="flex gap-4 text-sm">
            {["email", "sms", "inapp"].map((ch) => (
              <label key={ch} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  value={ch}
                  className="accent-ocean-500"
                  {...register("channels")}
                />
                {ch.toUpperCase()}
              </label>
            ))}
          </div>
          {errors.channels && <p className="text-red-400 text-xs mt-1">{errors.channels.message}</p>}
        </div>

        {/* AOI notice */}
        <div className="bg-ocean-900 border border-ocean-100/20 rounded-lg p-4 text-sm text-ocean-100/70">
          {drawnGeom
            ? "✓ Area of interest drawn. Submit to activate."
            : "Draw your area of interest on the map at /map (use the AOI tool), then return here, or use the map embed below."}
        </div>

        {apiError && <p className="text-red-400 text-sm">{apiError}</p>}

        <button
          type="submit"
          disabled={isSubmitting}
          className="bg-ocean-500 hover:bg-ocean-700 disabled:opacity-50 rounded-lg px-6 py-3 font-semibold transition"
        >
          {isSubmitting ? "Submitting…" : "Subscribe →"}
        </button>
      </form>

      <p className="mt-10 text-xs text-ocean-100/30">
        You will receive a confirmation email. No password required.
        Unsubscribe at any time with the link in any alert email.
      </p>
    </main>
  );
}
