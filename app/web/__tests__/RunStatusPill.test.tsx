import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RunStatusPill } from "@/components/RunStatusPill";

describe("RunStatusPill", () => {
  const knownStatuses = ["succeeded", "failed", "partial", "ingesting", "inferring", "alerting", "pending"] as const;

  it.each(knownStatuses)("renders status text for '%s'", (status) => {
    render(<RunStatusPill status={status} />);
    expect(screen.getByText(status)).toBeInTheDocument();
  });

  it("renders unknown status text with fallback styling", () => {
    render(<RunStatusPill status="unknown-xyz" />);
    const pill = screen.getByText("unknown-xyz");
    expect(pill).toBeInTheDocument();
    expect(pill.className).toContain("bg-gray-400");
  });

  it("applies green class for succeeded", () => {
    render(<RunStatusPill status="succeeded" />);
    expect(screen.getByText("succeeded").className).toContain("bg-green-600");
  });

  it("applies red class for failed", () => {
    render(<RunStatusPill status="failed" />);
    expect(screen.getByText("failed").className).toContain("bg-red-600");
  });

  it("applies yellow class for partial", () => {
    render(<RunStatusPill status="partial" />);
    expect(screen.getByText("partial").className).toContain("bg-yellow-500");
  });
});
