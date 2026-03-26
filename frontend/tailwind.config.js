/** @type {import('tailwindcss').Config} */
// Tell Tailwind which files to scan so it only ships CSS classes that are actually used.
// Without this glob, Tailwind produces megabytes of unused CSS in the bundle.
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

