/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Space Grotesk", "system-ui", "sans-serif"],
        serif: ["Fraunces", "serif"],
      },
      colors: {
        primary: {
          50: "#e9f6f3",
          100: "#cdebe5",
          200: "#9fd8cd",
          300: "#72c5b5",
          400: "#3eaa98",
          500: "#0c7c6d",
          600: "#0a6b5e",
          700: "#08594f",
          800: "#064840",
          900: "#053a34",
        },
        ink: {
          0: "#0f1f1a",
          1: "#2c3b33",
        },
        accent: {
          0: "#0c7c6d",
          1: "#f6b73c",
          2: "#0a4c9a",
        },
      }
    },
  },
  plugins: [],
}
