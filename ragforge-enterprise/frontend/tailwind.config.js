/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html','./src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: { 950:'#060D1A',900:'#0A1628',800:'#111D35',700:'#162240',600:'#1E2D4A',500:'#263655' },
        gold: { DEFAULT:'#C9A84C', light:'#E2C97A', dark:'#8B6914' },
        legal: { success:'#2A7A4B', danger:'#8B2020', warning:'#8B6914', info:'#1A5A8B' },
      },
      fontFamily: {
        display: ['"Playfair Display"','Georgia','serif'],
        body: ['"IBM Plex Sans"','system-ui','sans-serif'],
        mono: ['"IBM Plex Mono"','monospace'],
      },
    }
  },
  plugins: [],
};
