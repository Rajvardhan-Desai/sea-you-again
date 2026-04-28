import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { HorizonSlider } from "@/components/HorizonSlider";

const HORIZONS = ["2026-04-27", "2026-04-28", "2026-04-29"];

describe("HorizonSlider", () => {
  it("renders one button per horizon", () => {
    render(<HorizonSlider horizons={HORIZONS} selectedHorizon={0} onChange={() => {}} />);
    expect(screen.getAllByRole("button")).toHaveLength(HORIZONS.length);
  });

  it("labels each button with D+N prefix and the date", () => {
    render(<HorizonSlider horizons={HORIZONS} selectedHorizon={0} onChange={() => {}} />);
    expect(screen.getByText(/D\+1/)).toBeInTheDocument();
    expect(screen.getByText(/D\+2/)).toBeInTheDocument();
    expect(screen.getByText(/D\+3/)).toBeInTheDocument();
  });

  it("highlights the selected horizon button", () => {
    render(<HorizonSlider horizons={HORIZONS} selectedHorizon={1} onChange={() => {}} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons[1].className).toContain("bg-ocean-500");
    expect(buttons[0].className).not.toContain("bg-ocean-500");
  });

  it("calls onChange with the correct index on click", async () => {
    const onChange = vi.fn();
    render(<HorizonSlider horizons={HORIZONS} selectedHorizon={0} onChange={onChange} />);
    await userEvent.click(screen.getAllByRole("button")[2]);
    expect(onChange).toHaveBeenCalledOnce();
    expect(onChange).toHaveBeenCalledWith(2);
  });
});
