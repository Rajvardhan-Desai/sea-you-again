import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ocean: {
          50:  "#f0f9ff",
          100: "#e0f2fe",
          500: "#0ea5e9",
          700: "#0369a1",
          900: "#0c4a6e",
        },
        bloom: {
          low:      "#a6d96a",
          moderate: "#fdae61",
          high:     "#d7191c",
          severe:   "#7f0000",
        },
      },
    },
  },
  plugins: [],
};

export default config;
