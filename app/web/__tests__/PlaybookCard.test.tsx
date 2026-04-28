import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PlaybookCard } from "@/components/PlaybookCard";
import type { PlaybookBand } from "@/lib/api";

const makeBand = (overrides: Partial<PlaybookBand> = {}): PlaybookBand => ({
  severity:       "moderate",
  bloom_prob_min: 0.4,
  bloom_prob_max: 0.7,
  eri_class:      2,
  actions:        ["Reduce fishing activity", "Monitor water quality"],
  ...overrides,
});

describe("PlaybookCard", () => {
  it("renders the severity label", () => {
    render(<PlaybookCard band={makeBand({ severity: "high" })} />);
    expect(screen.getByText("high")).toBeInTheDocument();
  });

  it("renders all action bullet points", () => {
    const actions = ["Action A", "Action B", "Action C"];
    render(<PlaybookCard band={makeBand({ actions })} />);
    actions.forEach((a) => expect(screen.getByText(a)).toBeInTheDocument());
  });

  it("displays bloom probability range as percentages", () => {
    render(<PlaybookCard band={makeBand({ bloom_prob_min: 0.4, bloom_prob_max: 0.7 })} />);
    expect(screen.getByText(/40.+70/)).toBeInTheDocument();
  });

  it("displays ERI class", () => {
    render(<PlaybookCard band={makeBand({ eri_class: 3 })} />);
    expect(screen.getByText(/ERI 3/)).toBeInTheDocument();
  });

  it("applies correct colour class for 'severe'", () => {
    const { container } = render(<PlaybookCard band={makeBand({ severity: "severe" })} />);
    expect(container.firstChild as HTMLElement).toHaveClass("border-red-500");
  });

  it("applies fallback colour class for unknown severity", () => {
    const { container } = render(<PlaybookCard band={makeBand({ severity: "unknown" as any })} />);
    expect(container.firstChild as HTMLElement).toHaveClass("border-gray-500");
  });
});
