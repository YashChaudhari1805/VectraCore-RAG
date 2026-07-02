/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        serif: ['Playfair Display', 'Georgia', 'serif'],
      },
      colors: {
        background: '#FDFBF7',
        surface: 'rgba(255, 255, 255, 0.6)',
        border: '#E8E4DD',
        foreground: '#1A1C23',
        muted: '#6B7280',
        accent: {
          DEFAULT: '#1c2024',
          fg: '#FFFFFF'
        }
      },
      boxShadow: {
        'glass': '0 8px 32px rgba(0, 0, 0, 0.04)',
        'float': '0 4px 24px rgba(0, 0, 0, 0.06)',
      }
    },
  },
  plugins: [],
}